// RUN: triton-opt --emit-approx --emit-management --config-approx --transform-approx --finalize-approx %s | FileCheck %s
//
// Test: function substitution on a Triton kernel that calls tt.extern_elementwise
// for libdevice math (e.g., __nv_exp). The approx pipeline should replace it with
// an approximate version.
//
// TODO: This is a skeleton test. Fill in once tt.extern_elementwise pattern matching
// is implemented in transform-approx.

// module {
//   tt.func @my_kernel(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
//     // This would be the Triton IR for tl.exp(x):
//     %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", symbol = "__nv_expf", pure = true} : (tensor<1024xf32>) -> tensor<1024xf32>
//     tt.return %0 : tensor<1024xf32>
//   }
// }
