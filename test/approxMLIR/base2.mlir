// Input: Standard MLIR code
module {
  // Runtime functions (declared externally or defined elsewhere)
  func.func private @recover_impl(%arg0: f32, %arg1: f32, %state: i32) -> f32
  func.func private @checker_impl(%result: f32, %arg0: f32, %arg1: f32, %state: i32) -> i1

  // Safety contract annotation
  "approx.util.annotation.try"() <{
    func_name = "_sum", recover = "@recover_impl", checker = "@checker_impl"
  }> : () -> ()

  "approx.util.annotation.knob"() <{func_name = "_sum"}> : () -> ()

  // Function with existing KnobOp (created by emit-approx pass)
  func.func @_sum(%sum_iter: f32, %t: f32, %state: i32) -> f32 {
      %attempt = arith.addf %sum_iter, %t : f32
      return %attempt : f32
  }
}

