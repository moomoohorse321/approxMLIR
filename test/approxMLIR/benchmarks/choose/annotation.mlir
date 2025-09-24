// Knob for model_choose function, enabling task skipping.
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "model_choose",
  transform_type = "task_skipping",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 4>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1, 2>, // Corresponds to different models/tasks
  thresholds = array<i32: 2>,
  decisions = array<i32: 1, 2>
}> : () -> ()
