// === Knob 1: Per-term scoring loop over documents  ========
// Exact func name: score_term_over_docs
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "score_term_over_docs",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 40>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2>,
  thresholds = array<i32: 2000>,      // e.g., perforate when many docs
  decisions = array<i32: 0, 1>
}> : () -> ()

"approxMLIR.util.annotation.convert_to_call"() <{func_name = "score_term_over_docs"}> : () -> ()


// === Knob 2: lowering_corpus (func_substitute) ================================
// Exact func name: lowering_corpus
// Approx func name present in C: approx_lowering_corpus
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "lowering_corpus",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 5>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1, 2, 3>,
  thresholds = array<i32: 6>,
  decisions = array<i32: 0, 1>
}> : () -> ()
// Required for func_substitute:
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "lowering_corpus"}> : () -> ()

// === Knob 3: count_and_lower_words (func_substitute) =========================
// Exact func name: count_and_lower_words
// Approx func name present in C: approx_count_and_lower_words
"approxMLIR.util.annotation.decision_tree"() <{
  func_name = "count_and_lower_words",
  transform_type = "func_substitute",
  num_thresholds = 1 : i32,
  thresholds_uppers = array<i32: 1000>,
  thresholds_lowers = array<i32: 1>,
  decision_values = array<i32: 0, 1>,
  thresholds = array<i32: 4>,
  decisions = array<i32: 0, 1>
}> : () -> ()
// Required for func_substitute:
"approxMLIR.util.annotation.convert_to_call"() <{func_name = "count_and_lower_words"}> : () -> ()
