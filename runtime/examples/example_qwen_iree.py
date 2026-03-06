#!/usr/bin/env python3
"""Deploy Qwen2.5-0.5B-Instruct prefill+decode on IREE via torchax.

Workflow:
1. Load Qwen2.5-0.5B-Instruct from HuggingFace (PyTorch)
2. Convert to JAX via torchax (model.to('jax') + functional_call)
3. Use StaticCache for fixed-shape KV buffers (IREE-compatible)
4. Validate eager-mode correctness (prefill + autoregressive decode)
5. Export to StableHLO MLIR via jax.export
6. Compile to IREE VM flatbuffers (targeting CUDA)
7. Run full prefill + decode loop via IREE with timing

No approximation is applied (empty config / vanilla compilation).

Usage:
    # JAX traces on CPU; IREE deploys on CUDA (default)
    JAX_PLATFORMS=cpu python example_qwen_iree.py

    # Override model / backend / prompt:
    QWEN_MODEL=Qwen/Qwen2.5-1.5B APPROX_BACKEND=llvm-cpu \
        JAX_PLATFORMS=cpu python example_qwen_iree.py
"""

import os
import time

# JAX traces on CPU (avoids cuDNN version issues). IREE compilation
# targets CUDA by default — these are independent.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.export as jax_export
from jax.tree_util import register_pytree_node

import numpy as np
import torch
import torchax
import torchax.interop

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import cache_utils

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BACKEND = os.environ.get("APPROX_BACKEND", "cuda")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "100"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "256"))
PROMPT = os.environ.get("PROMPT", "What is the capital of France?")

print("=" * 70)
print("Qwen IREE Deployment (no approximation)")
print("=" * 70)
print(f"Model         : {MODEL_NAME}")
print(f"IREE backend  : {BACKEND}")
print(f"Max seq len   : {MAX_SEQ}")
print(f"JAX devices   : {jax.devices()}")
print(f"Prompt        : {PROMPT}")

# ===================================================================
# Step 1: Load model
# ===================================================================
print(f"\n--- Step 1: Loading model ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)
model.eval()

# Untie lm_head / embed_tokens weights so torchax sees all params.
if model.config.tie_word_embeddings:
    model.config.tie_word_embeddings = False
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Loaded {MODEL_NAME} ({n_params:.0f}M params)")

# Collect all stop token IDs from generation config.
eos_ids = model.generation_config.eos_token_id
if isinstance(eos_ids, int):
    eos_ids = [eos_ids]
stop_token_ids = set(eos_ids)
print(f"Stop tokens: {stop_token_ids}")

# ===================================================================
# Step 2: Convert to JAX via torchax + StaticCache
# ===================================================================
print(f"\n--- Step 2: Converting to JAX via torchax ---")

# Patch StaticCache.get_seq_length to avoid unsupported .any(dim=...)
# torchax doesn't support the dim keyword, but positional works.
def _patched_get_seq_length(self, layer_idx=0):
    return (self.key_cache[layer_idx][0, 0].any(-1)).sum()

cache_utils.StaticCache.get_seq_length = _patched_get_seq_length

# Patch torchax's index_copy to use dynamic_update_slice instead of scatter.
# The original uses x.at[indexes].set(source) which generates stablehlo.scatter
# that IREE can't compile (dimension map bug). For contiguous indices (like KV
# cache updates), dynamic_update_slice is equivalent and IREE handles it fine.
# Patch torchax's index_put to use dynamic_update_slice instead of scatter.
# The original x.at[indexes].set(values) generates stablehlo.scatter which
# IREE can't compile (dimension map bug). For contiguous slice updates (like
# KV cache), dynamic_update_slice is equivalent and IREE handles it fine.
import torchax.ops.jaten as _jaten  # trigger op registration
from torchax.ops.ops_registry import all_aten_ops as _all_ops

def _patched_index_put(self, indexes, values, accumulate=False):
    indexes = [slice(None, None, None) if i is None else i for i in indexes]
    # Check if this is a single-index scatter we can convert to dynamic_update_slice
    if not accumulate:
        # Find which dims are indexed by tensors vs slices
        tensor_dims = []
        for i, idx in enumerate(indexes):
            if not isinstance(idx, slice):
                tensor_dims.append((i, idx))
        # Single dim indexed by a single-element tensor -> dynamic_update_slice
        if len(tensor_dims) == 1:
            dim, idx_tensor = tensor_dims[0]
            starts = [0] * self.ndim
            starts[dim] = idx_tensor[0] if idx_tensor.ndim > 0 else idx_tensor
            return jax.lax.dynamic_update_slice(self, values, starts)
    # Fallback to original scatter-based approach
    indexes = tuple(indexes)
    if accumulate:
        return self.at[indexes].add(values)
    return self.at[indexes].set(values)

_all_ops[torch.ops.aten.index_put].func = _patched_index_put

# Register StaticCache as JAX pytree
def _flatten_static_cache(cache):
    return (cache.key_cache, cache.value_cache), (
        getattr(cache, 'max_cache_len', MAX_SEQ),
        getattr(cache, 'max_batch_size', 1),
    )

def _unflatten_static_cache(aux, children):
    cache = object.__new__(cache_utils.StaticCache)
    cache.key_cache = list(children[0])
    cache.value_cache = list(children[1])
    cache.max_cache_len, cache.max_batch_size = aux
    return cache

register_pytree_node(
    cache_utils.StaticCache,
    _flatten_static_cache,
    _unflatten_static_cache,
)

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

    model_weights = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())

print(f"Model on JAX device. StaticCache allocated: "
      f"K[0].shape={cache.key_cache[0].shape}")

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
print(f"\n--- Step 3: Eager-mode validation ---")

messages = [{"role": "user", "content": PROMPT}]
prompt_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
)
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
print(f"Decode   : {len(generated)} tokens in {t2-t1:.3f}s")
print(f"Output   : {text}")

# ===================================================================
# Step 5: Export to StableHLO via jax.export
# ===================================================================
print(f"\n--- Step 4: Exporting to StableHLO ---")

# Build abstract shape specs for jax.export.
# torchax tensors wrap JAX arrays in ._elem; extract to get JAX shapes/dtypes.
def _to_jax_shape(x):
    if hasattr(x, '_elem'):
        j = x._elem
        return jax.ShapeDtypeStruct(j.shape, j.dtype)
    return jax.ShapeDtypeStruct(x.shape, x.dtype)

weights_shapes = jax.tree.map(_to_jax_shape, model_weights)
buffers_shapes = jax.tree.map(_to_jax_shape, model_buffers)
kv_shapes = jax.tree.map(_to_jax_shape, cache)

# Export prefill (full prompt)
prefill_mlir = None
try:
    prefill_shapes = (
        weights_shapes,
        buffers_shapes,
        jax.ShapeDtypeStruct((1, SEQ_LEN), jnp.int32),
        jax.ShapeDtypeStruct((SEQ_LEN,), jnp.int32),
        kv_shapes,
    )
    exported = jax_export.export(jax.jit(jax_decode))(*prefill_shapes)
    prefill_mlir = str(exported.mlir_module())
    with open("qwen_prefill.mlir", "w") as f:
        f.write(prefill_mlir)
    print(f"Prefill  : {len(prefill_mlir):,} chars -> qwen_prefill.mlir")
except Exception as e:
    print(f"Prefill export failed: {e}")
    import traceback; traceback.print_exc()

# Export decode (single token)
decode_mlir = None
try:
    decode_shapes = (
        weights_shapes,
        buffers_shapes,
        jax.ShapeDtypeStruct((1, 1), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        kv_shapes,
    )
    exported = jax_export.export(jax.jit(jax_decode))(*decode_shapes)
    decode_mlir = str(exported.mlir_module())
    with open("qwen_decode.mlir", "w") as f:
        f.write(decode_mlir)
    print(f"Decode   : {len(decode_mlir):,} chars -> qwen_decode.mlir")
except Exception as e:
    print(f"Decode export failed: {e}")
    import traceback; traceback.print_exc()

# ===================================================================
# Step 6: Compile to IREE and load
# ===================================================================
import approx_runtime as ar

iree_backend = "cuda" if BACKEND == "cuda" else "llvm-cpu"
runtime_backend = BACKEND if BACKEND == "cuda" else "cpu"


def compile_and_load(mlir_text, name, backend, rt_backend):
    """Compile StableHLO -> IREE vmfb and load."""
    t_start = time.perf_counter()
    vmfb = ar.compile_to_iree(
        mlir_text,
        backend=backend,
        input_type="stablehlo",
    )
    t_compile = time.perf_counter()
    modules, device = ar.load_module(vmfb, backend=rt_backend)
    t_load = time.perf_counter()
    print(f"  {name}: compile {t_compile-t_start:.1f}s, "
          f"load {t_load-t_compile:.1f}s, "
          f"{len(vmfb):,} bytes, device={device} (backend={backend})")
    return modules


prefill_mod = decode_mod = None
if prefill_mlir or decode_mlir:
    print(f"\n--- Step 5: Compiling with IREE (backend={iree_backend}) ---")

# Track which backend each module uses (must match for buffer sharing)
actual_backend = iree_backend
actual_rt_backend = runtime_backend

if prefill_mlir:
    try:
        prefill_mod = compile_and_load(prefill_mlir, "prefill", iree_backend, runtime_backend)
    except Exception as e:
        print(f"  Prefill IREE {iree_backend} failed, falling back to llvm-cpu...")
        actual_backend = "llvm-cpu"
        actual_rt_backend = "cpu"
        try:
            prefill_mod = compile_and_load(prefill_mlir, "prefill", "llvm-cpu", "cpu")
        except Exception as e2:
            print(f"  Prefill IREE llvm-cpu also failed: {e2}")

if decode_mlir:
    try:
        decode_mod = compile_and_load(decode_mlir, "decode", actual_backend, actual_rt_backend)
    except Exception as e:
        if actual_backend != "llvm-cpu":
            print(f"  Decode IREE {actual_backend} failed, falling back to llvm-cpu...")
            # Both must use same backend for buffer sharing. Re-compile prefill too.
            actual_backend = "llvm-cpu"
            actual_rt_backend = "cpu"
            if prefill_mlir:
                print(f"  Re-compiling prefill on llvm-cpu for buffer compatibility...")
                try:
                    prefill_mod = compile_and_load(prefill_mlir, "prefill", "llvm-cpu", "cpu")
                except Exception as e3:
                    print(f"  Prefill IREE llvm-cpu failed: {e3}")
            try:
                decode_mod = compile_and_load(decode_mlir, "decode", "llvm-cpu", "cpu")
            except Exception as e2:
                print(f"  Decode IREE llvm-cpu also failed: {e2}")
        else:
            print(f"  Decode IREE llvm-cpu failed: {e}")

# ===================================================================
# Step 7: Run full inference via IREE-compiled modules
# ===================================================================
if prefill_mod and decode_mod:
    print(f"\n--- Step 6: IREE inference ---")

    def get_main(modules):
        for k, mod in modules.items():
            if k != "hal":
                return mod["main"]

    iree_prefill = get_main(prefill_mod)
    iree_decode = get_main(decode_mod)

    # Flatten all inputs to numpy arrays in pytree order.
    # Extract underlying JAX arrays from torchax tensors first.
    def _to_jax(x):
        return x._elem if hasattr(x, '_elem') else x

    weights_flat, _ = jax.tree.flatten(jax.tree.map(_to_jax, model_weights))
    buffers_flat, _ = jax.tree.flatten(jax.tree.map(_to_jax, model_buffers))
    cache_flat, _ = jax.tree.flatten(jax.tree.map(_to_jax, cache))

    weights_np = [np.array(w) for w in weights_flat]
    buffers_np = [np.array(b) for b in buffers_flat]
    cache_np = [np.array(c) for c in cache_flat]

    input_ids_np = np.array(prompt_ids, dtype=np.int32)
    cache_pos_np = np.arange(SEQ_LEN, dtype=np.int32)

    # -- IREE Prefill (warmup + timed) --
    all_prefill_inputs = weights_np + buffers_np + [input_ids_np, cache_pos_np] + cache_np
    prefill_out = iree_prefill(*all_prefill_inputs)  # warmup

    t0 = time.perf_counter()
    prefill_out = iree_prefill(*all_prefill_inputs)
    t1 = time.perf_counter()

    # Output structure: (logits, kv_cache_leaves...)
    # logits is first, then StaticCache key/value tensors
    iree_logits = np.array(prefill_out[0].to_host())
    print(f"Prefill  : {t1-t0:.4f}s  logits.shape={iree_logits.shape}")

    first_tok = int(np.argmax(iree_logits[0, -1, :]))
    print(f"First tok: '{tokenizer.decode([first_tok])}'")

    # -- IREE Decode loop --
    # StaticCache has fixed shape, so we can reuse the decode module
    # for every step — only the token and cache_position change.
    kv_buffers = [prefill_out[i] for i in range(1, len(prefill_out))]
    iree_generated = [first_tok]

    t_decode_start = time.perf_counter()
    for step in range(MAX_NEW_TOKENS - 1):
        next_tok_np = np.array([[iree_generated[-1]]], dtype=np.int32)
        pos_np = np.array([SEQ_LEN + step], dtype=np.int32)

        all_decode_inputs = weights_np + buffers_np + [next_tok_np, pos_np] + kv_buffers
        decode_out = iree_decode(*all_decode_inputs)

        decode_logits = np.array(decode_out[0].to_host())
        tok = int(np.argmax(decode_logits[0, -1, :]))
        iree_generated.append(tok)

        # Update KV buffers for next step
        kv_buffers = [decode_out[i] for i in range(1, len(decode_out))]

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
if prefill_mod and decode_mod:
    print(f"IREE  output : {iree_text}")
print(f"Prefill MLIR : {'OK' if prefill_mlir else 'FAILED'}")
print(f"Decode  MLIR : {'OK' if decode_mlir else 'FAILED'}")
print(f"Prefill IREE : {'OK' if prefill_mod else 'FAILED'}")
print(f"Decode  IREE : {'OK' if decode_mod else 'FAILED'}")

print("\n" + "=" * 70)
print("Done!")
