// Input: Standard MLIR code
module {
  // Runtime functions (declared externally or defined elsewhere)
  func.func private @recover_impl(%arg0: f32, %arg1: f32, %state: i32) -> f32
  func.func private @checker_impl(%result: f32, %arg0: f32, %arg1: f32, %state: i32) -> i1

  // Safety contract annotation
  "approx.util.annotation.try"() <{
    func_name = "_sum",
    recover = "@recover_impl",
    checker = "@checker_impl"
  }> : () -> ()

  // Function with existing KnobOp (created by emit-approx pass)
  func.func @_sum(%sum_iter: f32, %t: f32, %state: i32) -> f32 {
    %result = "approx.knob"(%state, %sum_iter, %t, %state) <{
      id = 0 : i32,
      rf = 0 : i32,
      QoS_in = array<i32>, QoS_out = array<i32>,
      transform_type = "try_check_recover"
    }> ({
      // Approximate computation
      %attempt = arith.addf %sum_iter, %t : f32
      
      // Original yield (TryOp will be injected BEFORE this)
      "approx.yield"(%attempt) : (f32) -> ()
    }) : (i32, f32, f32, i32) -> f32
    
    return %result : f32
  }
}

