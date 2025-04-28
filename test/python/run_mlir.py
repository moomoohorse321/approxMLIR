from iree import compiler as ireec

# Compile a module.
INPUT_MLIR = """
module @arithmetic {
  func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

# Compile using the vmvx (reference) target:
compiled_flatbuffer = ireec.tools.compile_str(
    INPUT_MLIR,
    target_backends=["vmvx"])

from iree import runtime as ireert
import numpy as np

# Register the module with a runtime context.
# Use the "local-task" CPU driver, which can load the vmvx executable:
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
ctx.add_vm_module(vm_module)

# Invoke the function and print the result.
print("INVOKE simple_mul")
arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
f = ctx.modules.arithmetic["simple_mul"]
results = f(arg0, arg1).to_host()
print("Results:", results)