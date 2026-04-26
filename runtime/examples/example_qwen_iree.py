#!/usr/bin/env python3
"""Deploy Qwen3.5-2B on IREE CUDA via torchax.

Workflow:
1. Load Qwen3.5-2B from HuggingFace (PyTorch)
2. Convert to JAX via torchax (model.to('jax') + functional_call)
3. Use StaticCache for fixed-shape KV buffers (IREE-compatible)
4. Validate eager-mode correctness (prefill + autoregressive decode)
5. Export to StableHLO MLIR via jax.export
6. Compile to IREE VM flatbuffers targeting CUDA
7. Run full prefill + decode loop via IREE with timing

No approximation is applied (empty config / vanilla compilation).

Usage:
    python example_qwen_iree.py
"""

import os
import re
import shlex
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.export as jax_export
from jax.tree_util import register_pytree_node, tree_flatten_with_path

import numpy as np
import torch
import torchax
import torchax.interop
from torchax.ops import jaten as _torchax_jaten  # noqa: F401 - registers base ops first.
from torchax.ops import ops_registry
from torchax.view import View

from transformers import AutoTokenizer
from transformers import cache_utils


def _aten_copy_accept_view_rhs(x, y, memory_format=None, env=None):
    if getattr(y, "device", None) is not None and y.device.type == "cpu":
        y = env.to_xla(y)

    if isinstance(x, View):
        x.update(y)
        return x

    y_elem = y.jax() if isinstance(y, View) else y._elem
    if x.ndim == 1 and y.ndim == 0:
        x._elem = jnp.array([y_elem.astype(x._elem.dtype)])
    else:
        x._elem = y_elem.astype(x._elem.dtype)
    return x


ops_registry.register_torch_dispatch_op(
    torch.ops.aten.copy_,
    _aten_copy_accept_view_rhs,
    is_jax_function=False,
    is_view_op=True,
    needs_env=True,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-2B")
BACKEND = os.environ.get("APPROX_BACKEND", "cuda")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "100"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "256"))
PROMPT = os.environ.get("PROMPT", "What is the capital of France?")
SKIP_EAGER_DECODE = os.environ.get("SKIP_EAGER_DECODE", "0") == "1"

print("=" * 70)
print("Qwen IREE Deployment (no approximation)")
print("=" * 70)
print(f"Model         : {MODEL_NAME}")
print(f"IREE backend  : {BACKEND}")
print(f"Max seq len   : {MAX_SEQ}")
print(f"JAX devices   : {jax.devices()}")
print(f"Prompt        : {PROMPT}")

IREE_EXTRA_ARGS = shlex.split(os.environ.get("IREE_EXTRA_ARGS", ""))
if IREE_EXTRA_ARGS:
    print(f"IREE extra args: {IREE_EXTRA_ARGS}")

# ===================================================================
# Step 1: Load model
# ===================================================================
print(f"\n--- Step 1: Loading model ---")
from transformers import Qwen3_5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class Qwen35TextOnlyCausalLM(torch.nn.Module):
    """Text-only wrapper around the Qwen3.5 multimodal checkpoint."""

    def __init__(self, conditional_model):
        super().__init__()
        self.config = conditional_model.config.text_config
        self.generation_config = getattr(conditional_model, "generation_config", None)
        self.model = conditional_model.model.language_model
        self.lm_head = conditional_model.lm_head

    def forward(
        self,
        input_ids,
        cache_position=None,
        past_key_values=None,
        return_dict=False,
        use_cache=True,
    ):
        position_ids = None
        if cache_position is not None:
            position_ids = cache_position.reshape(1, -1)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return logits, outputs.past_key_values


conditional_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)
model = Qwen35TextOnlyCausalLM(conditional_model)
del conditional_model

def _conv1d_forward_cast_input(self, input):
    return torch.nn.functional.conv1d(
        input.to(self.weight.dtype),
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )

for module in model.modules():
    if module.__class__.__name__ == "Qwen3_5GatedDeltaNet" and hasattr(module, "conv1d"):
        module.conv1d = module.conv1d.to(torch.float32)
        module.conv1d.forward = _conv1d_forward_cast_input.__get__(
            module.conv1d, module.conv1d.__class__
        )

model.eval()

# Untie lm_head / embed_tokens weights so torchax sees all params.
if model.config.tie_word_embeddings:
    model.config.tie_word_embeddings = False
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Loaded {MODEL_NAME} ({n_params:.0f}M params)")

# Collect all stop token IDs.
eos_ids = getattr(model.generation_config, "eos_token_id", None)
if eos_ids is None:
    eos_ids = tokenizer.eos_token_id
if isinstance(eos_ids, int):
    eos_ids = [eos_ids]
stop_token_ids = {tok for tok in eos_ids if tok is not None}
print(f"Stop tokens: {stop_token_ids}")

# ===================================================================
# Step 2: Convert to JAX via torchax + StaticCache
# ===================================================================
print(f"\n--- Step 2: Converting to JAX via torchax ---")

# Register StaticCache as JAX pytree. Transformers versions before Qwen3.5
# expose key_cache/value_cache directly; newer versions store per-layer cache
# objects that can mix full-attention and linear-attention state.
def _flatten_static_cache(cache):
    if hasattr(cache, "key_cache"):
        return (cache.key_cache, cache.value_cache), (
            "legacy",
            getattr(cache, "max_cache_len", MAX_SEQ),
            getattr(cache, "max_batch_size", 1),
        )

    children = []
    layer_aux = []
    for layer in cache.layers:
        layer_name = layer.__class__.__name__
        if hasattr(layer, "max_cache_len"):
            initialized = getattr(layer, "is_initialized", False)
            if initialized:
                children.append((layer.keys, layer.values, layer.cumulative_length))
            else:
                children.append((layer.cumulative_length,))
            layer_aux.append(
                (
                    "kv",
                    layer_name,
                    getattr(layer, "max_cache_len", MAX_SEQ),
                    initialized,
                )
            )
            continue

        names = []
        values = []
        for name in ("conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            if value is not None:
                names.append(name)
                values.append(value)
        children.append(tuple(values))
        layer_aux.append(
            (
                "linear",
                layer_name,
                tuple(names),
                getattr(layer, "is_conv_states_initialized", False),
                getattr(layer, "is_recurrent_states_initialized", False),
                getattr(layer, "has_previous_state", False),
            )
        )

    return tuple(children), (
        "layers",
        tuple(layer_aux),
        getattr(cache, "offloading", False),
    )

def _unflatten_static_cache(aux, children):
    def _device_or_jax(value):
        return getattr(value, "device", "jax")

    layout = aux[0]
    cache = object.__new__(cache_utils.StaticCache)
    if layout == "legacy":
        cache.key_cache = list(children[0])
        cache.value_cache = list(children[1])
        _, cache.max_cache_len, cache.max_batch_size = aux
        return cache

    _, layer_aux, offloading = aux
    cache.layers = []
    cache.layer_class_to_replicate = None
    cache.offloading = offloading
    for layer_info, layer_children in zip(layer_aux, children):
        kind, layer_name = layer_info[:2]
        layer_cls = getattr(cache_utils, layer_name)
        layer = object.__new__(layer_cls)

        if kind == "kv":
            _, _, max_cache_len, initialized = layer_info
            layer.max_cache_len = max_cache_len
            layer.is_initialized = initialized
            if initialized:
                layer.keys, layer.values, layer.cumulative_length = layer_children
                layer.dtype = layer.keys.dtype
                layer.device = _device_or_jax(layer.keys)
                layer.max_batch_size, layer.num_heads = layer.keys.shape[:2]
                layer.k_head_dim = layer.keys.shape[-1]
                layer.v_head_dim = layer.values.shape[-1]
            else:
                (layer.cumulative_length,) = layer_children
            cache.layers.append(layer)
            continue

        _, _, names, conv_initialized, recurrent_initialized, has_previous = layer_info
        values = dict(zip(names, layer_children))
        layer.conv_states = values.get("conv_states")
        layer.recurrent_states = values.get("recurrent_states")
        layer.is_conv_states_initialized = conv_initialized
        layer.is_recurrent_states_initialized = recurrent_initialized
        layer.has_previous_state = has_previous
        sample = layer.conv_states if layer.conv_states is not None else layer.recurrent_states
        if sample is not None:
            layer.dtype = sample.dtype
            layer.device = _device_or_jax(sample)
            layer.max_batch_size = sample.shape[0]
            layer.conv_kernel_size = sample.shape[-1]
        cache.layers.append(layer)

    return cache

register_pytree_node(
    cache_utils.StaticCache,
    _flatten_static_cache,
    _unflatten_static_cache,
)

def _move_static_cache_to_jax(cache):
    if hasattr(cache, "key_cache"):
        return cache
    for layer in cache.layers:
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = layer.cumulative_length.to("jax")
    return cache

# Move model to torchax ('jax' device)
env = torchax.default_env()
with env:
    model.to('jax')

    # Create StaticCache with fixed shape
    cache = cache_utils.StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=MAX_SEQ,
        device='jax',
        dtype=torch.bfloat16,
    )
    cache = _move_static_cache_to_jax(cache)

    model_weights = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())

if hasattr(cache, "key_cache"):
    cache_summary = f"K[0].shape={cache.key_cache[0].shape}"
else:
    cache_summary = f"{len(cache.layers)} cache layers"
print(f"Model on JAX device. StaticCache allocated: {cache_summary}")

# ===================================================================
# Step 3: Define decode_one via torch.func.functional_call
# ===================================================================
# A single forward step: takes token + cache_position, returns logits + updated cache.
# Works for both prefill (seq_len tokens) and decode (1 token) since
# StaticCache has fixed shape — only cache_position changes.

def decode_one(weights, buffers, input_ids, cache_position, past_key_values):
    logits, updated_cache = torch.func.functional_call(
        model,
        (weights, buffers),
        (input_ids,),
        dict(
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        ),
    )
    return logits, updated_cache

# jax_view converts torch callable -> JAX callable (handles torchax tensor conversion)
jax_decode = torchax.interop.jax_view(decode_one)
jitted_decode = torchax.interop.jax_jit(decode_one)

# ===================================================================
# Step 4: Eager-mode validation (CPU via torchax)
# ===================================================================
print(f"\n--- Step 4: Eager-mode validation ---")

messages = [{"role": "user", "content": PROMPT}]
prompt_inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
)
if hasattr(prompt_inputs, "input_ids"):
    prompt_ids = prompt_inputs.input_ids
elif isinstance(prompt_inputs, dict):
    prompt_ids = prompt_inputs["input_ids"]
else:
    prompt_ids = prompt_inputs
SEQ_LEN = prompt_ids.shape[1]
print(f"Prompt tokens: {SEQ_LEN}")

if SEQ_LEN > MAX_SEQ:
    raise ValueError(f"Prompt length {SEQ_LEN} exceeds MAX_SEQ {MAX_SEQ}")

with env:
    prompt_jax = prompt_ids.to('jax')
    cache_position = torch.arange(SEQ_LEN, device='jax')

    # Prefill
    t0 = time.perf_counter()
    logits, kv = jitted_decode(
        model_weights, model_buffers, prompt_jax, cache_position, cache
    )
    t1 = time.perf_counter()
    print(f"Prefill  : {t1-t0:.3f}s  logits.shape={logits.shape}")

    # Greedy autoregressive decode
    next_token_id = int(logits[0, -1].argmax())
    generated = [next_token_id]

    if not SKIP_EAGER_DECODE:
        for step in range(MAX_NEW_TOKENS - 1):
            cur_token = torch.tensor([[next_token_id]], device='jax')
            pos = torch.tensor([SEQ_LEN + step], device='jax')
            logits, kv = jitted_decode(
                model_weights, model_buffers, cur_token, pos, kv
            )
            next_token_id = int(logits[0, -1].argmax())
            generated.append(next_token_id)
            if next_token_id in stop_token_ids:
                break

    t2 = time.perf_counter()

text = tokenizer.decode(generated, skip_special_tokens=True)
decode_label = "Decode" if not SKIP_EAGER_DECODE else "First tok"
print(f"{decode_label:8s}: {len(generated)} tokens in {t2-t1:.3f}s")
print(f"Output   : {text}")

# ===================================================================
# Step 5: Export to StableHLO via jax.export
# ===================================================================
print(f"\n--- Step 5: Exporting to StableHLO ---")

# torchax tensors wrap JAX arrays in ._elem; extract the underlying JAX array.
def _unwrap_torchax(x):
    return x._elem if hasattr(x, '_elem') else x

def _to_jax_shape(x):
    j = _unwrap_torchax(x)
    return jax.ShapeDtypeStruct(j.shape, j.dtype)

weights_shapes = jax.tree.map(_to_jax_shape, model_weights)
buffers_shapes = jax.tree.map(_to_jax_shape, model_buffers)
prefill_kv_shapes = jax.tree.map(_to_jax_shape, cache)
decode_kv_shapes = jax.tree.map(_to_jax_shape, kv)

jitted_jax_decode = jax.jit(jax_decode)

def export_mlir(name, seq_len, filename, kv_shapes):
    """Export a single StableHLO module and write to file."""
    shapes = (
        weights_shapes,
        buffers_shapes,
        jax.ShapeDtypeStruct((1, seq_len), jnp.int32),
        jax.ShapeDtypeStruct((seq_len,), jnp.int32),
        kv_shapes,
    )
    exported = jax_export.export(jitted_jax_decode)(*shapes)
    mlir = str(exported.mlir_module())
    with open(filename, "w") as f:
        f.write(mlir)
    print(f"{name:9s}: {len(mlir):,} chars -> {filename}")
    return mlir

prefill_mlir = export_mlir("Prefill", SEQ_LEN, "qwen_prefill.mlir", prefill_kv_shapes)
decode_mlir = export_mlir("Decode", 1, "qwen_decode.mlir", decode_kv_shapes)

# ===================================================================
# Step 6: Compile to IREE and load
# ===================================================================
import approx_runtime as ar
import iree.runtime as ireert

if BACKEND != "cuda":
    raise ValueError(f"Unsupported backend {BACKEND!r}; this example is CUDA-only")
iree_backend = "cuda"
runtime_backend = "cuda"


def compile_and_load(mlir_text, name, backend, rt_backend):
    """Compile StableHLO -> IREE vmfb and load."""
    t_start = time.perf_counter()
    vmfb = ar.compile_to_iree(
        mlir_text,
        backend=backend,
        input_type="stablehlo",
        extra_args=IREE_EXTRA_ARGS,
    )
    t_compile = time.perf_counter()
    modules, device = ar.load_module(vmfb, backend=rt_backend)
    t_load = time.perf_counter()
    print(f"  {name}: compile {t_compile-t_start:.1f}s, "
          f"load {t_load-t_compile:.1f}s, "
          f"{len(vmfb):,} bytes, device={device} (backend={backend})")
    return modules, device


print(f"\n--- Step 6: Compiling with IREE (backend={iree_backend}) ---")
prefill_mod, _ = compile_and_load(prefill_mlir, "prefill", iree_backend, runtime_backend)
decode_mod, decode_device = compile_and_load(
    decode_mlir, "decode", iree_backend, runtime_backend
)

# ===================================================================
# Step 7: Run full inference via IREE-compiled modules
# ===================================================================
print(f"\n--- Step 7: IREE inference ---")

iree_prefill = prefill_mod.jit_call_torch["main"]
iree_decode = decode_mod.jit_call_torch["main"]

def _path_key(part):
    if hasattr(part, "key"):
        return f"[{part.key!r}]"
    if hasattr(part, "idx"):
        return f"[{part.idx}]"
    if hasattr(part, "name"):
        return f".{part.name}"
    return f"[{part}]"

def _flatten_arg_pairs(arg_index, tree, convert):
    leaves = tree_flatten_with_path(jax.tree.map(_unwrap_torchax, tree))[0]
    return [
        (f"args[{arg_index}]" + "".join(_path_key(part) for part in path),
         convert(value))
        for path, value in leaves
    ]

def _live_arg_paths(mlir_text):
    matches = re.findall(r'%arg([0-9]+):[^,\n]*?loc\("(args[^"]+)"\)', mlir_text)
    return [path for _, path in sorted(matches, key=lambda match: int(match[0]))]

def _result_paths(mlir_text):
    return re.findall(r'jax\.result_info = "(result[^"]+)"', mlir_text)

def _select_live_args(name, live_paths, pairs):
    values_by_path = dict(pairs)
    missing = [path for path in live_paths if path not in values_by_path]
    if missing:
        raise ValueError(f"{name} live args missing from Python inputs: {missing[:5]}")
    pruned_count = len(pairs) - len(live_paths)
    if pruned_count:
        print(f"{name:8s}: using {len(live_paths)} live inputs "
              f"({pruned_count} pruned by export)")
    return [values_by_path[path] for path in live_paths]

def _cache_output_pairs(result_paths, outputs):
    return [
        (path.replace("result[1]", "args[4]", 1), outputs[idx])
        for idx, path in enumerate(result_paths)
        if path.startswith("result[1]")
    ]

input_ids_np = np.asarray(prompt_ids, dtype=np.int32)
cache_pos_np = np.arange(SEQ_LEN, dtype=np.int32)

weights_np_pairs = _flatten_arg_pairs(0, model_weights, np.asarray)
buffers_np_pairs = _flatten_arg_pairs(1, model_buffers, np.asarray)
cache_np_pairs = _flatten_arg_pairs(4, cache, np.asarray)

prefill_live_paths = _live_arg_paths(prefill_mlir)
decode_live_paths = _live_arg_paths(decode_mlir)
prefill_result_paths = _result_paths(prefill_mlir)
decode_result_paths = _result_paths(decode_mlir)
prefill_mlir = decode_mlir = None

# -- IREE Prefill (warmup + timed) --
all_prefill_pairs = (
    weights_np_pairs
    + buffers_np_pairs
    + [("args[2]", input_ids_np), ("args[3]", cache_pos_np)]
    + cache_np_pairs
)
all_prefill_inputs = _select_live_args("prefill", prefill_live_paths, all_prefill_pairs)
_ = iree_prefill(*all_prefill_inputs)  # warmup
del _

t0 = time.perf_counter()
prefill_out = iree_prefill(*all_prefill_inputs)
t1 = time.perf_counter()

# Output structure: (logits, kv_cache_leaves...)
iree_logits = np.asarray(prefill_out[0].to_host())
print(f"Prefill  : {t1-t0:.4f}s  logits.shape={iree_logits.shape}")

first_tok = int(np.argmax(iree_logits[0, -1, :]))
print(f"First tok: '{tokenizer.decode([first_tok])}'")

# -- IREE Decode loop --
# Static weights/buffers stay fixed; cache buffers are threaded through outputs.
cache_buffer_pairs = _cache_output_pairs(prefill_result_paths, prefill_out)
iree_generated = [first_tok]
static_np_pairs = weights_np_pairs + buffers_np_pairs  # constant across steps

t_static_upload_start = time.perf_counter()
decode_live_path_set = set(decode_live_paths)
static_inputs_by_path = {
    path: ireert.asdevicearray(decode_device, value)
    for path, value in static_np_pairs
    if path in decode_live_path_set
}
t_static_upload_end = time.perf_counter()
print(f"Static inputs uploaded to device in "
      f"{t_static_upload_end-t_static_upload_start:.4f}s")

t_decode_start = time.perf_counter()
for step in range(MAX_NEW_TOKENS - 1):
    next_tok_np = np.array([[iree_generated[-1]]], dtype=np.int32)
    pos_np = np.array([SEQ_LEN + step], dtype=np.int32)

    decode_pairs = (
        list(static_inputs_by_path.items())
        + [("args[2]", next_tok_np), ("args[3]", pos_np)]
        + cache_buffer_pairs
    )
    decode_inputs = _select_live_args("decode", decode_live_paths, decode_pairs)
    decode_out = iree_decode(*decode_inputs)

    tok = int(np.argmax(np.asarray(decode_out[0].to_host())[0, -1, :]))
    iree_generated.append(tok)
    cache_buffer_pairs = _cache_output_pairs(decode_result_paths, decode_out)

    if tok in stop_token_ids:
        break

t_decode_end = time.perf_counter()

iree_text = tokenizer.decode(iree_generated, skip_special_tokens=True)
print(f"Decode   : {len(iree_generated)} tokens in "
      f"{t_decode_end-t_decode_start:.4f}s")
print(f"Output   : {iree_text}")

# ===================================================================
# Summary
# ===================================================================
print(f"\n--- Summary ---")
print(f"Eager output : {text}")
print(f"IREE  output : {iree_text}")
print("Prefill MLIR : OK")
print("Decode  MLIR : OK")
print("Prefill IREE : OK")
print("Decode  IREE : OK")

print("\n" + "=" * 70)
print("Done!")
