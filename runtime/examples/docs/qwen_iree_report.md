# Report: Deploying HuggingFace Transformer Models via torchax + JAX Export + IREE

## Summary

Successfully built an end-to-end pipeline: **PyTorch (HuggingFace Qwen2.5-0.5B) -> torchax -> JAX -> StableHLO MLIR -> IREE compilation -> IREE inference** with full autoregressive decode loop. Both prefill and decode produce correct output ("The capital of France is Paris.") through IREE-compiled modules.

The pipeline works on `llvm-cpu`. Two IREE bugs block the `cuda` backend.

## Pipeline Architecture

```
HuggingFace Model (PyTorch)
    | torchax (model.to('jax') + functional_call)
JAX function (via torchax.interop.jax_view)
    | jax.export
StableHLO MLIR (.mlir files)
    | iree.compiler.compile_str
IREE VM Flatbuffer (.vmfb)
    | iree.runtime
Inference (prefill + decode loop)
```

Key design choices:
- **StaticCache** (not DynamicCache): DynamicCache grows the KV sequence dimension each decode step, making shapes incompatible with static IREE compilation. StaticCache pre-allocates `(batch, n_heads, max_seq_len, head_dim)` buffers.
- **`torchax.interop.jax_view`**: Wraps the torch function so JAX tracers are properly converted to torchax tensors during `jax.export` tracing. Using raw `jax.jit(torch_func)` fails because `torch.func.functional_call` rejects JAX tracers.
- **`torch.func.functional_call`**: Makes the model's forward pass a pure function (weights as inputs), required for JAX tracing.

## IREE Bugs Found

### Bug 1: `iree_linalg_ext.scatter` dimension map invalid (BLOCKING for StaticCache on all backends)

**Symptom**: IREE compilation fails with:
```
'iree_linalg_ext.scatter' op dimension map is invalid.
element (2) at index#0 is out of bounds
```

**Root cause**: HuggingFace's `StaticCache.update()` uses `index_copy_(dim=2, ...)` which torchax lowers to `x.at[indexes].set(values)`, generating this StableHLO scatter:
```mlir
%95 = "stablehlo.scatter"(%arg294, %94, %88) <{
    scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0, 1, 3],
        inserted_window_dims = [2],
        scatter_dims_to_operand_dims = [2],
        index_vector_dim = 1>
}> : (tensor<1x2x256x64xbf16>, tensor<1x1xi32>, tensor<1x2x1x64xbf16>)
   -> tensor<1x2x256x64xbf16>
```

This is a valid StableHLO scatter (inserting 1 KV entry along dim 2 of a 4D tensor), but IREE's `ConvertStableHloToIreeInputDialect` pass produces an invalid `iree_linalg_ext.scatter` from it. Specifically, the dimension mapping `scatter_dims_to_operand_dims = [2]` triggers `element (2) at index#0 is out of bounds`.

**Workaround**: Patch torchax's `aten.index_put` handler to use `jax.lax.dynamic_update_slice` instead of `x.at[indexes].set(values)` for single-tensor-indexed dims. This generates `stablehlo.dynamic_update_slice` which IREE handles correctly.

**Fix suggestion**: The StableHLO-to-IREE-input scatter conversion should handle the case where `scatter_dims_to_operand_dims` references a dimension index >= number of scatter batch dims. This scatter pattern is semantically equivalent to `dynamic_update_slice` and could be canonicalized.

### Bug 2: CUDA `failed to distribute` on small batch matmul (BLOCKING for decode on CUDA)

**Symptom**: IREE CUDA compilation fails with:
```
'func.func' op failed to distribute
```
at the attention matmul in decode mode.

**Root cause**: The decode-step attention computes `(14, 1, 64) @ (14, 64, 256) -> (14, 1, 256)` -- a batched matmul with the query dimension being 1. IREE's GPU tiling/distribution heuristics can't find a valid workgroup mapping for this very narrow shape. Prefill works because the query dimension is larger (e.g., 36).

**Impact**: Decode can't compile for CUDA. The `llvm-cpu` backend compiles fine.

**Fix suggestion**: The GPU codegen should have a fallback distribution strategy for narrow matmuls (M=1 batched GEMMs), which are the standard case for autoregressive LLM decode.

### Known Issue: bf16 `to_host()` (potential issue on CUDA)

There is a known IREE bug where bf16 device buffers fail to convert to numpy via `.to_host()`. We didn't hit this because the working configuration uses `llvm-cpu`, but it would be a problem on CUDA. The Qwen model uses bf16 throughout.

## What Works Today

| Component | Status |
|-----------|--------|
| torchax model conversion | Works (with `get_seq_length` patch for torchax) |
| StaticCache with fixed KV shapes | Works (with `index_put` -> `dynamic_update_slice` patch) |
| StableHLO export via `jax.export` | Works for both prefill and decode |
| IREE compilation (llvm-cpu) | Works for both prefill and decode |
| IREE compilation (cuda) | Prefill works, decode blocked by Bug 2 |
| IREE inference (llvm-cpu) | Full decode loop works, correct output |
| IREE inference (cuda) | Prefill only (decode blocked) |

## Performance (llvm-cpu, Qwen2.5-0.5B, 36-token prompt)

| Phase | Eager (torchax CPU) | IREE (llvm-cpu) |
|-------|-------------------|-----------------|
| Prefill (36 tokens) | 1.1s | 7.8s |
| Decode (8 tokens) | 2.0s | 6.0s |

Note: IREE llvm-cpu is slower than eager because (a) no GPU, (b) generic CPU target (no AVX/SSE flags), (c) no warmup amortization. With CUDA working, performance should be significantly better.

## What IREE Could Do

1. **Fix scatter dimension mapping** (Bug 1): The `stablehlo.scatter` -> `iree_linalg_ext.scatter` conversion needs to handle 4D operands with `scatter_dims_to_operand_dims = [2]`. Alternatively, add a canonicalization pass that converts scatter-as-slice-update patterns to `dynamic_update_slice` before lowering.

2. **Fix CUDA codegen for M=1 batched matmul** (Bug 2): The GPU distribution strategy needs a fallback for `(B, 1, K) @ (B, K, N)` shapes. This is the standard decode-step attention pattern for all transformer models.

3. **Fix bf16 `to_host()`**: Enable bf16 buffer -> numpy conversion on CUDA.

4. **Consider a torchax -> IREE fast path**: The current pipeline goes through StableHLO MLIR text, which is ~550K chars for a 0.5B model. A binary serialization or direct API would be faster.

## Files

- `example_qwen_iree.py` -- the working end-to-end example
- `qwen_prefill.mlir` / `qwen_decode.mlir` -- exported StableHLO (generated at runtime)

## Reproducing

```bash
# From approxMLIR/runtime/examples/
JAX_PLATFORMS=cpu python example_qwen_iree.py

# Force llvm-cpu backend (skips CUDA attempts):
JAX_PLATFORMS=cpu APPROX_BACKEND=llvm-cpu python example_qwen_iree.py
```

## Versions Used

- torchax 0.0.7
- PyTorch 2.7.0
- JAX 0.5.3 / jaxlib 0.5.3
- transformers 4.51.3
- iree-base-compiler / iree-base-runtime (pip)
- IREE built from source (LLVM 22.0.0git)
