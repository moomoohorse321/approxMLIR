// B) self-box inner loop perforation (loop_perforate)
// decisions: 1 = step 1 (exact), 2 = step 2 (perforated)
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "self_box_accumulate",
  transform_type = "loop_perforate",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 100>,   // state in [0,100] (= |q_i| * 100)
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 1, 2, 3, 4>,
  thresholds = array<i32: 60>,           // |q_i| >= 0.60 => step=1 else step=2
  decisions = array<i32: 1, 2>
}> : () -> ()

// C) neighbor-box perforation (loop_perforate over k/j)
// decisions: 1 = step 1 (exact), 2 = step 2 (perforated)
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "neighbor_box_accumulate",
  transform_type = "loop_perforate",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 10>,    // state = nn in [0,26]
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 1, 2, 3, 4>,
  thresholds = array<i32: 6>,           // nn >= 13 => step=1 else step=2
  decisions = array<i32: 1, 2>
}> : () -> ()
