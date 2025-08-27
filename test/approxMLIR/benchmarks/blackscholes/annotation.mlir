// Knob — CNDF substitution (proper func_substitute)
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "compute_cndf",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 2>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 1>,
  decisions = array<i32: 0, 1>
}> : () -> ()


// Required for func_substitute
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "compute_cndf"}> : () -> ()

// Knob — Black-Scholes approximation (func_substitute)
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "BlkSchlsEqEuroNoDiv",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 2>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 1>,
  decisions = array<i32: 0, 1>
}> : () -> ()

// Required for func_substitute
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "BlkSchlsEqEuroNoDiv"}> : () -> ()
