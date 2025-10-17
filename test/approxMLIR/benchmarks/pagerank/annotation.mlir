"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "update_node_rank",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 50>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2>,
  thresholds = array<i32: 24>,
  decisions = array<i32: 0, 1>
}> : () -> ()

// Required by func_substitute
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "update_node_rank"}> : () -> ()


// Knob C - pagerank_worker_impl
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "pagerank_worker_impl",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 5>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2>,
  thresholds = array<i32: 3>,
  decisions = array<i32: 0, 1>
}> : () -> ()

// Required by func_substitute
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "pagerank_worker_impl"}> : () -> ()
