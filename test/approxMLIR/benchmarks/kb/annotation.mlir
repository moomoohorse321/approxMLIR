"approxMLIR.util.annotation.decision_tree"() <{
    func_name = "parse_embedding",
    transform_type = "func_substitute",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 4>,
    thresholds_lowers = array<i32: 1>,
    decision_values = array<i32: 0, 1, 2>,
    thresholds = array<i32: 1>,
    decisions = array<i32: 0, 1>
}> : () -> ()

// Required for func_substitute
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "parse_embedding"}> : () -> ()


"approxMLIR.util.annotation.decision_tree"() <{
func_name = "compute_similarities_with_state",
transform_type = "func_substitute",
num_thresholds = 1 : i32,
thresholds_uppers = array<i32: 5>,
thresholds_lowers = array<i32: 1>,
decision_values = array<i32: 0, 1, 2, 3>,
thresholds = array<i32: 2>,
decisions = array<i32: 0, 1>
}> : () -> ()

"approxMLIR.util.annotation.convert_to_call"() <{func_name = "compute_similarities_with_state"}> : () -> ()