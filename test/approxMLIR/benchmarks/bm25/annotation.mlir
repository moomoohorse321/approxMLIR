// === Knob 1: Per-term scoring loop over documents  ========
// Exact func name: score_term_over_docs
func.func @approx_state_identity(%arg0: i32) -> i32 {
  return %arg0 : i32
}

"approx.util.annotation.decision_tree"() <{
  func_name = "score_term_over_docs",
  transform_type = "func_substitute",
  state_indices = array<i64: 7>,
  state_function = "approx_state_identity",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 40>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2>,
  thresholds = array<i32: 2000>,      // e.g., perforate when many docs
  decisions = array<i32: 0, 1>
}> : () -> ()

"approx.util.annotation.convert_to_call"() <{func_name = "score_term_over_docs"}> : () -> ()


// === Knob 2: lowering_corpus (func_substitute) ================================
// Exact func name: lowering_corpus
// Approx func name present in C: approx_lowering_corpus
"approx.util.annotation.decision_tree"() <{
  func_name = "lowering_corpus",
  transform_type = "func_substitute",
  state_indices = array<i64: 5>,
  state_function = "approx_state_identity",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 5>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2, 3>,
  thresholds = array<i32: 6>,
  decisions = array<i32: 0, 1>
}> : () -> ()
// Required for func_substitute:
"approx.util.annotation.convert_to_call"() <{func_name = "lowering_corpus"}> : () -> ()

// === Knob 3: count_and_lower_words (func_substitute) =========================
// Exact func name: count_and_lower_words
// Approx func name present in C: approx_count_and_lower_words
"approx.util.annotation.decision_tree"() <{
  func_name = "count_and_lower_words",
  transform_type = "func_substitute",
  state_indices = array<i64: 2>,
  state_function = "approx_state_identity",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 1000>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 4>,
  decisions = array<i32: 0, 1>
}> : () -> ()
// Required for func_substitute:
"approx.util.annotation.convert_to_call"() <{func_name = "count_and_lower_words"}> : () -> ()
