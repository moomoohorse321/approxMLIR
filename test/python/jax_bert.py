from approxMLIR import ToolBox

import jax
import jax.numpy as jnp
from flax.linen import Module
from transformers import FlaxBertForMaskedLM
from jax import export

from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Returns prettyprint of StableHLO module without large constants
def get_stablehlo_asm(module_str):
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
    return stablehlo_module.operation.get_asm(large_elements_limit=20)

# Disable logging for better tutorial rendering
import logging
logging.disable(logging.WARNING)

# Load a Flax (JAX-native) BERT model
model = FlaxBertForMaskedLM.from_pretrained("bert-base-uncased")

# Create example input
sample_input = jnp.ones((2, 128), dtype=jnp.int32)

# Define a function to trace
def forward_fn(input_ids):
    return model(input_ids).logits

# JIT compile and export
def compile():
  jitted_fn = jax.jit(forward_fn)
  input_shape = jax.ShapeDtypeStruct(sample_input.shape, sample_input.dtype)
  exported = export.export(jitted_fn)(input_shape)
  stablehlo_module = exported.mlir_module()

  with open("pretty_jax_bert.mlir", "w") as f:
      f.write(get_stablehlo_asm(stablehlo_module))

# with open("jax_bert.mlir", "w") as f:
#     f.write(stablehlo_module)
    
loaded_module = ToolBox.load_mlir_from_file("jax_bert.mlir").jit_forward_fn

bert_infer_func = loaded_module.main
print(bert_infer_func(sample_input).to_host())