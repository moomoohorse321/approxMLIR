func.func @approx_state_identity(%arg0: i32) -> i32 {
  return %arg0 : i32
}

"approx.util.annotation.decision_tree"() <{
    func_name = "parse_embedding",
    transform_type = "func_substitute",
    state_indices = array<i64: 2>,
    state_function = "approx_state_identity",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 10>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 0, 1, 2>,
    thresholds = array<i32: 1>,
    decisions = array<i32: 0, 1>
}> : () -> ()

// Required for func_substitute
"approx.util.annotation.convert_to_call"() <{func_name = "parse_embedding"}> : () -> ()


"approx.util.annotation.decision_tree"() <{
func_name = "compute_similarities_with_state",
transform_type = "func_substitute",
state_indices = array<i64: 0>,
state_function = "approx_state_identity",
num_thresholds = 1 : i32,
thresholds_uppers = array<i32: 5>,
thresholds_lowers = array<i32: 1>,
decision_values = array<i32: 0, 1, 2>,
thresholds = array<i32: 2>,
decisions = array<i32: 0, 1>
}> : () -> ()

"approx.util.annotation.convert_to_call"() <{func_name = "compute_similarities_with_state"}> : () -> ()
