// Input: Standard MLIR code
module {
  func.func @_sum(%sum_iter: f32, %t: f32) -> (f32) {
    %sum_next = arith.addf %sum_iter, %t : f32
    return %sum_next : f32
  }

  func.func @for_with_yield(%buffer: memref<1024xf32>) -> (f32) {
    %sum_0 = arith.constant 0.0 : f32
    %sum = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_0) -> (f32) {
      %t = affine.load %buffer[%i] : memref<1024xf32>
      // Call the function to be approximated
      %sum_next = func.call @_sum(%sum_iter, %t) : (f32, f32) -> f32
      affine.yield %sum_next : f32
    }
    return %sum : f32
  }
}

// ---------------------------------------------------------
// Gold Intermeidate (With knob and try emitted): Transformed with approx.knob inside _sum
// ---------------------------------------------------------
// module {
//   // Runtime Recovery Function
//   // Arguments match the context required: %sum_iter (state), %t (input)
//   func.func private @recover_impl(%sum_iter: f32, %t: f32) -> f32
// 
//   // Runtime Checker Function
//   func.func private @checker_impl(%candidate: f32, %sum_iter: f32, %t: f32) -> i1
// 
//   // Transformed _sum function
//   func.func @_sum(%sum_iter: f32, %t: f32) -> f32 {
//     %final_result = "approx.knob"(....) ({
//       
//       // --- COMPUTATION (Inside Knob) ---
//       // Implicit capture of %sum_iter and %t
//       %attempt = arith.addf %sum_iter, %t : f32
// 
//       "approx.try"(%attempt, %sum_iter, %t) ({
//         
//         // --- CHECK REGION ---
//         // Determines if %attempt is valid.
//         // when this block is lowered, it will look at the yielded result, taking it as the input variables
//         // the yield will be overwritten such that it yields the emitted if-else (for try-check-recover) result. 
//         ^bb0(%c: f32, %s: f32, %p: f32):
//            %valid = func.call @checker_impl(%c, %s, %p) : (f32, f32, f32) -> i1
//            "approx.yield"(%valid) : (i1) -> ()
//       
//       }) {
//         // --- RECOVER ATTRIBUTE ---
//         recover = "@recover_impl"
//       } : (f32, f32, f32) -> (f32)
//       
//       // will be overwirtten when try is lowered
//       "approx.yield"(%attempt) : (f32) -> ()
//     }) {
//       // --- KnobOp Configuration ---
//       id = 1 : i32,
//       ....
//     } : (f32) -> (f32)
// 
//     return %final_result : f32
//   }
// 
//   func.func @for_with_yield(%buffer: memref<1024xf32>) -> (f32) {
//     %sum_0 = arith.constant 0.0 : f32
//     %sum = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_0) -> (f32) {
//       %t = affine.load %buffer[%i] : memref<1024xf32>
//       
//       // The call remains standard; the complexity is hidden inside @_sum
//       %sum_next = func.call @_sum(%sum_iter, %t) : (f32, f32) -> f32
//       
//       affine.yield %sum_next : f32
//     }
//     return %sum : f32
//   }
// }