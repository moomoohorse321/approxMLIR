from approxMLIR import ToolBox

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import jax.lax as lax

from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir


def get_stablehlo_asm(module_str):
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
    return stablehlo_module.operation.get_asm(large_elements_limit=20)

# Create a JIT-transformed function
@jax.jit
def plus(x,y):
  return jnp.add(x,y)

@jax.jit
def matmul(x, y):
    A, B = x.shape
    B, C = y.shape
    z = jnp.zeros((A, C), dtype=x.dtype)  # Initialize output matrix
    
    # Define the outer loop body
    def outer_body(i, z):
        # Define the inner loop body
        def inner_body(j, z):
            # Define the innermost loop body
            def dot_body(k, s):
                return s + x[i, k] * y[k, j]
            
            # Compute dot product for position (i, j)
            s = lax.fori_loop(0, B, dot_body, 0.0)
            # Update the matrix at position (i, j)
            z = z.at[i, j].set(s)
            return z
        
        # Run inner loop over columns
        z = lax.fori_loop(0, C, inner_body, z)
        return z
    
    # Run outer loop over rows
    z = lax.fori_loop(0, A, outer_body, z)
    return z


def conv2d(input, kernel, stride=1, padding=0):
    """
    2D convolution with NHWC format (batch, height, width, channels)
    
    Args:
        input: Input tensor of shape (N, H, W, C_in)
        kernel: Kernel tensor of shape (K_h, K_w, C_in, C_out)
        stride: Stride for convolution (integer, same for both dimensions)
        padding: Padding size (integer, same for all sides)
    
    Returns:
        Output tensor of shape (N, H_out, W_out, C_out)
    """
    N, H, W, C_in = input.shape
    K_h, K_w, _, C_out = kernel.shape
    
    # Apply padding if needed
    if padding > 0:
        input = jnp.pad(input, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        H, W = H + 2 * padding, W + 2 * padding
    
    # Calculate output dimensions
    H_out = (H - K_h) // stride + 1
    W_out = (W - K_w) // stride + 1
    
    # Initialize output tensor
    output = jnp.zeros((N, H_out, W_out, C_out), dtype=input.dtype)
    
    # Define batch loop body
    def batch_body(n, output):
        # Define output height loop body
        def h_body(oh, output):
            # Define output width loop body
            def w_body(ow, output):
                # Define output channel loop body
                def c_out_body(oc, output):
                    # Initialize accumulator for this output position
                    acc = 0.0
                    
                    # Define kernel height loop body
                    def kh_body(kh, acc):
                        # Define kernel width loop body
                        def kw_body(kw, acc):
                            # Define input channel loop body
                            def c_in_body(ic, acc):
                                # Calculate input position
                                ih = oh * stride + kh
                                iw = ow * stride + kw
                                
                                # Accumulate the convolution
                                acc = acc + input[n, ih, iw, ic] * kernel[kh, kw, ic, oc]
                                return acc
                            
                            # Loop over input channels
                            acc = lax.fori_loop(0, C_in, c_in_body, acc)
                            return acc
                        
                        # Loop over kernel width
                        acc = lax.fori_loop(0, K_w, kw_body, acc)
                        return acc
                    
                    # Loop over kernel height
                    acc = lax.fori_loop(0, K_h, kh_body, acc)
                    
                    # Set the output value
                    output = output.at[n, oh, ow, oc].set(acc)
                    return output
                
                # Loop over output channels
                output = lax.fori_loop(0, C_out, c_out_body, output)
                return output
            
            # Loop over output width
            output = lax.fori_loop(0, W_out, w_body, output)
            return output
        
        # Loop over output height
        output = lax.fori_loop(0, H_out, h_body, output)
        return output
    
    # Loop over batch
    output = lax.fori_loop(0, N, batch_body, output)
    return output

# Create abstract x shapes
inputs = (np.int32(1), np.int32(1),)
input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in inputs]

# Export the function to StableHLO
stablehlo_add = export.export(plus)(*input_shapes).mlir_module()
print(stablehlo_add)

# create a StableHLO module for matrix multiplication
inputs = (np.zeros((2, 3), dtype=np.float32), np.zeros((3, 4), dtype=np.float32))
input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in inputs]
stablehlo_matmul = export.export(matmul)(*input_shapes).mlir_module()
print(stablehlo_matmul)


# Test conv2d function with a simple example
# Input: batch=1, 4x4 image, 2 input channels
# Kernel: 3x3, 2 input channels, 3 output channels
# Stride: 1, Padding: 1
input_conv = np.random.randn(1, 4, 4, 2).astype(np.float32)
kernel_conv = np.random.randn(3, 3, 2, 3).astype(np.float32)

input_shapes_conv = [
    jax.ShapeDtypeStruct(input_conv.shape, input_conv.dtype),
    jax.ShapeDtypeStruct(kernel_conv.shape, kernel_conv.dtype)
]

# Create a wrapped function for export with default stride and padding
# Use partial to make stride and padding static
from functools import partial
conv2d_default = jax.jit(partial(conv2d, stride=1, padding=1))

stablehlo_conv = export.export(conv2d_default)(*input_shapes_conv).mlir_module()
print("=== Conv2D function StableHLO ===")
with open("conv2d.mlir", "w") as f:
    f.write(stablehlo_conv)
with open("pretty_conv2d.mlir", "w") as f:
    f.write(get_stablehlo_asm(stablehlo_conv))
print(get_stablehlo_asm(stablehlo_conv))
print()

conv_modules = ToolBox.load_mlir_from_file("conv2d.mlir").jit__unnamed_wrapped_function_
# print(conv_modules.keys())
conv_func = conv_modules.main

# Test the exported StableHLO modules
print(conv_func(input_conv, kernel_conv).to_host())
