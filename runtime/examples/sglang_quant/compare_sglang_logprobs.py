#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
EXAMPLES_DIR = THIS_FILE.parent
RUNTIME_DIR = EXAMPLES_DIR.parent.parent
APPROXMLIR_DIR = RUNTIME_DIR.parent
REPO_ROOT = APPROXMLIR_DIR.parent
LOCAL_TRITON_PYTHON = REPO_ROOT / "triton" / "python"
BOOTSTRAP_DIR = EXAMPLES_DIR / "bootstrap"


def _child_pythonpath() -> str:
    parts = [
        str(LOCAL_TRITON_PYTHON),
        str(EXAMPLES_DIR),
        str(RUNTIME_DIR),
        str(BOOTSTRAP_DIR),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _engine_kwargs() -> dict:
    kwargs = {
        "model_path": os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-2B"),
        "attention_backend": os.environ.get("ATTENTION_BACKEND", "triton"),
        "sampling_backend": os.environ.get("SAMPLING_BACKEND", "pytorch"),
        "disable_cuda_graph": os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "0") == "1",
        "log_level": "error",
    }
    mem_fraction = os.environ.get("SGLANG_MEM_FRACTION_STATIC")
    if mem_fraction:
        kwargs["mem_fraction_static"] = float(mem_fraction)
    return kwargs


def _prompts() -> list[str]:
    prompts = json.loads(os.environ.get("PROMPTS_JSON", "null") or "null")
    if prompts:
        return prompts
    prompt = os.environ.get("PROMPT", "The capital of France is")
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    return [prompt] * batch_size


def _normalize_results(results):
    if isinstance(results, list):
        return results
    return [results]


def _lp_only(seq) -> list[float]:
    return [float(x[0]) for x in seq]


def _topk_only(seq) -> list[list[tuple[float, int]]]:
    return [
        [(float(item[0]), int(item[1])) for item in token_topk]
        for token_topk in seq
    ]


def _topk_distribution(entries: list[tuple[float, int]]) -> dict:
    probs = {}
    total = 0.0
    for logprob, token_id in entries:
        prob = math.exp(logprob)
        probs[token_id] = prob
        total += prob
    probs["__other__"] = max(0.0, 1.0 - total)
    norm = sum(probs.values())
    if norm <= 0.0:
        return {"__other__": 1.0}
    return {k: v / norm for k, v in probs.items()}


def _js_divergence(p: dict, q: dict) -> float:
    eps = 1e-12
    keys = set(p.keys()) | set(q.keys())
    m = {}
    for key in keys:
        m[key] = 0.5 * (p.get(key, 0.0) + q.get(key, 0.0))
    def _kl(a: dict, b: dict) -> float:
        total = 0.0
        for key in keys:
            pa = a.get(key, 0.0)
            if pa <= 0.0:
                continue
            total += pa * math.log(pa / max(b.get(key, 0.0), eps))
        return total
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _teacher_forced_metrics(
    exact_logprobs: list[list[float]],
    approx_logprobs: list[list[float]],
    exact_topk: list[list[list[tuple[float, int]]]],
    approx_topk: list[list[list[tuple[float, int]]]],
) -> dict:
    flat_exact = [x for seq in exact_logprobs for x in seq]
    flat_approx = [x for seq in approx_logprobs for x in seq]
    if len(flat_exact) != len(flat_approx):
        raise ValueError(
            f"logprob length mismatch: exact={len(flat_exact)} approx={len(flat_approx)}"
        )

    deltas = [a - e for e, a in zip(flat_exact, flat_approx)]
    nll_exact = [-x for x in flat_exact]
    nll_approx = [-x for x in flat_approx]

    js_values = []
    top1_match = 0
    top1_total = 0
    ref_in_topk = 0
    ref_total = 0
    for exact_seq, approx_seq in zip(exact_topk, approx_topk):
        if len(exact_seq) != len(approx_seq):
            raise ValueError(
                f"top-k length mismatch: exact={len(exact_seq)} approx={len(approx_seq)}"
            )
        for exact_tok, approx_tok in zip(exact_seq, approx_seq):
            exact_dist = _topk_distribution(exact_tok)
            approx_dist = _topk_distribution(approx_tok)
            js_values.append(_js_divergence(exact_dist, approx_dist))
            if exact_tok and approx_tok:
                top1_total += 1
                top1_match += int(exact_tok[0][1] == approx_tok[0][1])
                ref_total += 1
                ref_token = exact_tok[0][1]
                ref_in_topk += int(any(tok == ref_token for _, tok in approx_tok))

    mean_exact_nll = statistics.mean(nll_exact) if nll_exact else 0.0
    mean_approx_nll = statistics.mean(nll_approx) if nll_approx else 0.0
    return {
        "num_sequences": len(exact_logprobs),
        "num_scored_tokens": len(flat_exact),
        "teacher_forced_mean_exact_nll": mean_exact_nll,
        "teacher_forced_mean_approx_nll": mean_approx_nll,
        "teacher_forced_mean_logprob_delta": statistics.mean(deltas) if deltas else 0.0,
        "teacher_forced_median_logprob_delta": statistics.median(deltas) if deltas else 0.0,
        "teacher_forced_perplexity_ratio": math.exp(mean_approx_nll - mean_exact_nll),
        "topk_js_mean": statistics.mean(js_values) if js_values else 0.0,
        "topk_js_median": statistics.median(js_values) if js_values else 0.0,
        "top1_agreement_rate": (top1_match / top1_total) if top1_total else 0.0,
        "reference_top1_in_approx_topk_rate": (ref_in_topk / ref_total) if ref_total else 0.0,
    }


def _run_worker(mode: str, payload_path: str) -> int:
    import sglang as sgl
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    prompts = _prompts()
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-2B")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "8"))
    topk = int(os.environ.get("TOP_LOGPROBS_NUM", "20"))
    sampling_seed = int(os.environ.get("SAMPLING_SEED", "0"))

    tokenizer = get_tokenizer(model_path)
    prompt_ids = [tokenizer.encode(prompt) for prompt in prompts]
    engine = sgl.Engine(**_engine_kwargs())
    try:
        if mode == "generate_ref":
            sampling_params = {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
                "sampling_seed": sampling_seed,
            }
            results = _normalize_results(
                engine.generate(
                    input_ids=prompt_ids,
                    sampling_params=sampling_params,
                    return_logprob=True,
                    top_logprobs_num=topk,
                )
            )
            payload = {
                "prompt_ids": prompt_ids,
                "outputs": [
                    {
                        "text": result["text"],
                        "output_ids": result["output_ids"],
                        "output_token_logprobs": _lp_only(
                            result["meta_info"]["output_token_logprobs"]
                        ),
                        "output_top_logprobs": _topk_only(
                            result["meta_info"].get("output_top_logprobs", [])
                        ),
                    }
                    for result in results
                ],
            }
        elif mode == "score_ref":
            ref_data = json.loads(
                Path(os.environ["REFERENCE_PAYLOAD_PATH"]).read_text(encoding="utf-8")
            )
            decode_only = os.environ.get("APPROX_SGLANG_DECODE_ONLY", "0") == "1"
            scored = []
            if decode_only:
                for prompt, ref_output in zip(prompt_ids, ref_data["outputs"]):
                    continuation_lps = []
                    continuation_topk = []
                    ref_output_ids = ref_output["output_ids"]
                    for step, target_token_id in enumerate(ref_output_ids):
                        prefix_ids = prompt + ref_output_ids[:step]
                        result = _normalize_results(
                            engine.generate(
                                input_ids=[prefix_ids],
                                sampling_params={
                                    "temperature": 0.0,
                                    "max_new_tokens": 1,
                                    "ignore_eos": True,
                                    "sampling_seed": sampling_seed,
                                },
                                return_logprob=True,
                                top_logprobs_num=topk,
                                token_ids_logprob=[target_token_id],
                            )
                        )[0]
                        continuation_lps.append(
                            float(result["meta_info"]["output_token_ids_logprobs"][0][0][0])
                        )
                        continuation_topk.append(
                            [(float(t[0]), int(t[1])) for t in result["meta_info"]["output_top_logprobs"][0]]
                        )
                    scored.append(
                        {
                            "continuation_token_logprobs": continuation_lps,
                            "continuation_top_logprobs": continuation_topk,
                        }
                    )
            else:
                eval_input_ids = [
                    prompt + output["output_ids"]
                    for prompt, output in zip(prompt_ids, ref_data["outputs"])
                ]
                results = _normalize_results(
                    engine.generate(
                        input_ids=eval_input_ids,
                        sampling_params={
                            "temperature": 0.0,
                            "max_new_tokens": 0,
                            "ignore_eos": True,
                            "sampling_seed": sampling_seed,
                        },
                        return_logprob=True,
                        logprob_start_len=0,
                        top_logprobs_num=topk,
                    )
                )
                for result, ref_output in zip(results, ref_data["outputs"]):
                    seq_len = len(ref_output["output_ids"])
                    raw_input_token_logprobs = result["meta_info"]["input_token_logprobs"][1:]
                    raw_input_top_logprobs = result["meta_info"].get("input_top_logprobs", [])[1:]
                    input_token_logprobs = _lp_only(raw_input_token_logprobs)
                    input_top_logprobs = _topk_only(raw_input_top_logprobs)
                    scored.append(
                        {
                            "continuation_token_logprobs": input_token_logprobs[-seq_len:],
                            "continuation_top_logprobs": input_top_logprobs[-seq_len:],
                        }
                    )
            payload = {"scored": scored, "evaluation_mode": "stepwise_decode" if decode_only else "prefill_teacher_forced"}
        else:
            raise ValueError(f"unsupported worker mode: {mode}")
    finally:
        engine.shutdown()

    Path(payload_path).write_text(json.dumps(payload), encoding="utf-8")
    return 0


def _run_child(worker_mode: str, payload_path: str, quant_enabled: bool) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = _child_pythonpath()
    env["SGLANG_DISABLE_CUDA_GRAPH"] = os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "0")
    if quant_enabled:
        env["APPROX_SGLANG_QUANT"] = "1"
    else:
        env["APPROX_SGLANG_QUANT"] = "0"
        env.pop("APPROX_SGLANG_MODE", None)
        env.pop("APPROX_SGLANG_TARGET", None)
        env.pop("APPROX_SGLANG_BACKEND", None)
        env.pop("APPROX_SGLANG_SQ_ARTIFACT_PATH", None)
        env.pop("APPROX_SGLANG_SQ_GROUP_SIZE", None)
        env.pop("APPROX_SGLANG_SQ_BLOCK_K", None)
        env.pop("APPROX_SGLANG_BLOCK_N", None)
    subprocess.run(
        [sys.executable, str(THIS_FILE), "--worker", worker_mode, payload_path],
        check=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    return json.loads(Path(payload_path).read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        if len(sys.argv) != 4:
            raise SystemExit("usage: compare_sglang_logprobs.py --worker <mode> <payload_path>")
        return _run_worker(sys.argv[2], sys.argv[3])

    with tempfile.TemporaryDirectory(prefix="approx_sglang_logprob_") as tmpdir:
        tmpdir = Path(tmpdir)
        ref_path = str(tmpdir / "ref.json")
        exact_score_path = str(tmpdir / "exact_score.json")
        approx_score_path = str(tmpdir / "approx_score.json")

        ref = _run_child("generate_ref", ref_path, quant_enabled=False)
        os.environ["REFERENCE_PAYLOAD_PATH"] = ref_path
        exact_score = _run_child("score_ref", exact_score_path, quant_enabled=False)
        approx_score = _run_child("score_ref", approx_score_path, quant_enabled=True)

        exact_logprobs = [
            item["continuation_token_logprobs"] for item in exact_score["scored"]
        ]
        approx_logprobs = [
            item["continuation_token_logprobs"] for item in approx_score["scored"]
        ]
        exact_topk = [
            item["continuation_top_logprobs"] for item in exact_score["scored"]
        ]
        approx_topk = [
            item["continuation_top_logprobs"] for item in approx_score["scored"]
        ]
        metrics = _teacher_forced_metrics(
            exact_logprobs=exact_logprobs,
            approx_logprobs=approx_logprobs,
            exact_topk=exact_topk,
            approx_topk=approx_topk,
        )

        summary = {
            "evaluation_mode": exact_score.get("evaluation_mode", "unknown"),
            "model_path": os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-2B"),
            "num_prompts": len(ref["prompt_ids"]),
            "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "8")),
            "top_logprobs_num": int(os.environ.get("TOP_LOGPROBS_NUM", "20")),
            "reference_texts": [item["text"] for item in ref["outputs"]],
            "reference_output_lengths": [len(item["output_ids"]) for item in ref["outputs"]],
            "metrics": metrics,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
