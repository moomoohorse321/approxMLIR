import jax
import jax.numpy as jnp
from gemma import gm
from jax.export import export
import os


# --- Setup: Set memory allocation and choose model ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# --- 1. Load the Model and Parameters for 270M ---
print("Loading Gemma 270M model architecture...")
model = gm.nn.Gemma3_270M()

print("Loading 270M model parameters (will download to default cache if not found)...")

params = gm.ckpts.load_params(
    gm.ckpts.CheckpointPath.GEMMA3_270M_IT,
)
print("Model and parameters loaded successfully.")

# --- 2. Define the function to export ---
# This part remains the same.
def forward_pass(tokens):
    return model.apply({'params': params}, tokens)

# --- 3. Define the concrete shape and type of our inputs ---
# This part remains the same.
BATCH_SIZE = 1
SEQ_LEN = 128
input_tokens_shape = jax.ShapeDtypeStruct((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
jitted_pass = jax.jit(forward_pass)
# --- 4. Use jax.export to get the StableHLO ---
print("Exporting the model to StableHLO...")
exported_model = export(jitted_pass)(input_tokens_shape)

# --- 5. Store the StableHLO IR to a file ---
stablehlo_ir = exported_model.mlir_module()
with open("gemma_270m_latest.mlir", "w") as f:
    f.write(stablehlo_ir)

print("\n✅ Successfully saved StableHLO to gemma_270m_latest.mlir")