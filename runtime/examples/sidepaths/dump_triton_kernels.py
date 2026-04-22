#!/usr/bin/env python3
"""Stage 1 dump harness: direct Triton kernel menu -> TTIR dump.

Replaces the HF + torch.compile path on machines where the PyTorch/Triton
Inductor ABI has drifted. The kernel menu in :mod:`kernels.llm_like` uses
Qwen2.5-0.5B-derived shapes so the resulting TTIR is representative of the
serving workload without depending on a compiled HF forward pass.

Environment variables
---------------------
OUT_DIR        Directory to write dumps into (default: /tmp/triton_ttir_dump)
DUMP_VERBOSE   "1" to log each dumped kernel (default: "0")

The harness scopes Triton's compile cache to ``$OUT_DIR/triton_cache`` via
``TRITON_CACHE_DIR`` and wipes only that scoped dir, so the user-wide
``~/.triton/cache`` is never touched.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


_EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))


def _setup_dlopen_flags() -> None:
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    except AttributeError:
        pass


def main() -> int:
    _setup_dlopen_flags()

    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/triton_ttir_dump"))
    verbose = os.environ.get("DUMP_VERBOSE", "0") == "1"

    # Scoped cache: re-runs always re-compile (hook fires) without touching ~/.triton/cache.
    cache_dir = out_dir / "triton_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    print(f"[dump-kernels] scoped TRITON_CACHE_DIR={cache_dir}")

    import torch
    from triton import knobs

    sys.path.insert(0, str(Path(__file__).parent))
    from kernels.llm_like import MENU

    import approx_runtime as ar

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this dump harness")

    print(f"[dump-kernels] out_dir={out_dir}")

    hook = ar.make_triton_dump_hook(
        out_dir=out_dir,
        source="direct_triton_menu",
        verbose=verbose,
    )
    recorder = ar.make_launch_recorder(out_dir=out_dir, source="direct_triton_menu")
    # Start fresh so a rerun doesn't mix with an older log.
    if recorder.path.exists():
        recorder.path.unlink()
    knobs.runtime.add_stages_inspection_hook = hook

    for entry in MENU:
        print(f"[dump-kernels] --- {entry.name} ---")
        for shape in entry.shapes():
            tag = shape.get("tag", "")
            print(f"[dump-kernels]   launching {entry.name} {tag} {shape}")
            entry.launcher(shape, recorder=recorder)

    torch.cuda.synchronize()
    knobs.runtime.add_stages_inspection_hook = None

    jsons = sorted(p for p in out_dir.glob("*.json") if p.name != "launches.jsonl")
    print(f"[dump-kernels] dumped {len(jsons)} distinct kernels into {out_dir}")
    by_name: dict[str, int] = {}
    for p in jsons:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        by_name[rec["func_name"]] = by_name.get(rec["func_name"], 0) + 1
    for name, count in sorted(by_name.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d} shape variants  {name}")

    launches_path = out_dir / "launches.jsonl"
    if launches_path.exists():
        with launches_path.open(encoding="utf-8") as f:
            launches = [json.loads(line) for line in f if line.strip()]
        print(f"[dump-kernels] recorded {len(launches)} launches in {launches_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
