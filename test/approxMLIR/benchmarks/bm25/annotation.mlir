// === Knob 1: TF counting (func_substitute) ================================
// Exact func name: tf_count_whole_word
// Approx func name present in C: approx_tf_count_whole_word
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "tf_count_whole_word",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 12>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 6>,
  decisions = array<i32: 0, 1>
}> : () -> ()
// Required for func_substitute:
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "tf_count_whole_word"}> : () -> ()

// === Knob 2: DF membership test (func_substitute) =========================
// Exact func name: df_contains_whole_word
// Approx func name present in C: approx_df_contains_whole_word
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "df_contains_whole_word",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 10>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 4>,
  decisions = array<i32: 0, 1>
}> : () -> ()
// Required for func_substitute:
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "df_contains_whole_word"}> : () -> ()

// === Knob 3: Per-term scoring loop over documents (loop_perforate) ========
// Exact func name: score_term_over_docs
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "score_term_over_docs",
  transform_type = "loop_perforate",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 1000000>,
  thresholds_lowers = array<i32: 0>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 2000>,      // e.g., perforate when many docs
  decisions = array<i32: 0, 1>
}> : () -> ()
