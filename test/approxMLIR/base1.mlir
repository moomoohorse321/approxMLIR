module {

// Annotation 1: Create knobOp wrapping the function
"approx.util.annotation.knob"() <{func_name = "base"}> : () -> ()

// Annotation 2: Inject decideOp into the knob (assumes knob already exists)
"approx.util.annotation.decision_tree"() <{
    func_name = "base",
    transform_type = "loop_perforate",
    state_indices = array<i64: 0>,
    state_function = "getState",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 100>,
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 50>,
    decisions = array<i32: 1, 2>
}> : () -> ()

func.func @getState(%arg0: i32) -> i32 {
  return %arg0 : i32
}

func.func @base(%arg0: i32, %arg1: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %n = arith.index_cast %arg0 : i32 to index
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> f32 {
    %v = memref.load %arg1[%i] : memref<?xf32>
    %new = arith.addf %acc, %v : f32
    scf.yield %new : f32
  }
  return %sum : f32
}

}