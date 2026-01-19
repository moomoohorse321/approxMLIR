

// Knob — choosing nearest centroid (loop_perforate over centroids)
func.func @approx_state_identity(%arg0: i32) -> i32 {
  return %arg0 : i32
}

"approx.util.annotation.decision_tree"() <{
  func_name = "choose_cluster",
  transform_type = "loop_perforate",
  state_indices = array<i64: 5>,
  state_function = "approx_state_identity",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 20>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2, 3, 4>,
  thresholds = array<i32: 8>,
  decisions = array<i32: 1, 2>
}> : () -> ()

// Knob — assigning points & accumulating (loop_perforate over points)
"approx.util.annotation.decision_tree"() <{
  func_name = "assign_points_and_accumulate",
  transform_type = "loop_perforate",
  state_indices = array<i64: 9>,
  state_function = "approx_state_identity",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 20>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2, 3, 4>,
  thresholds = array<i32: 10000>,
  decisions = array<i32: 1, 2>
}> : () -> ()

// Knob — run_kmeans_iterations (loop_perforate over iterations)
"approx.util.annotation.decision_tree"() <{
  func_name = "run_kmeans_iterations",
  transform_type = "loop_perforate",
  state_indices = array<i64: 9>,
  state_function = "approx_state_identity",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 1000000>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2, 3, 4>,
  thresholds = array<i32: 10000>,
  decisions = array<i32: 1, 2>
}> : () -> ()
