
// Knob — distance computation (func_substitute)
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "compute_distance_sq",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 64>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 16>,
  decisions = array<i32: 0, 1>
}> : () -> ()
// Required for func_substitute
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "compute_distance_sq"}> : () -> ()

// Knob — choosing nearest centroid (loop_perforate over centroids)
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "choose_cluster",
  transform_type = "loop_perforate",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 16>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 8>,
  decisions = array<i32: 1, 2>
}> : () -> ()

// Knob — assigning points & accumulating (loop_perforate over points)
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "assign_points_and_accumulate",
  transform_type = "loop_perforate",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 2000000>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 10000>,
  decisions = array<i32: 1, 2>
}> : () -> ()