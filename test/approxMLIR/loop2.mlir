// Input: Standard MLIR code
module {
  // Runtime functions (declared externally or defined elsewhere)
  func.func private @recover_impl(%arg0: f32, %arg1: f32) -> f32
  func.func private @checker_impl(%result: f32, %arg0: f32, %arg1: f32) -> i1

  // Safety contract annotation
  "approx.util.annotation.try"() <{
    func_name = "_sum", recover = "@recover_impl", checker = "@checker_impl"
  }> : () -> ()

  "approx.util.annotation.knob"() <{func_name = "_sum"}> : () -> ()
  
  "approx.util.annotation.decision_tree"() <{decision_values = array<i32: 0, 1>, decisions = array<i32: 0, 1>, func_name = "_sum", num_thresholds = 1 : i32, thresholds = array<i32: 2>, thresholds_lowers = array<i32: 0>, thresholds_uppers = array<i32: 4>, transform_type = "task_skipping"}> : () -> ()

  // Function with existing KnobOp (created by emit-approx pass)
  func.func @_sum(%sum_iter: f32, %t: f32, %state: i32) -> f32 {
      %attempt = arith.addf %sum_iter, %t : f32
      return %attempt : f32
  }
}

