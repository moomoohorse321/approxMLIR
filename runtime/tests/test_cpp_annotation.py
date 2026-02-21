"""Tests for cpp_annotation.py - C++ comment parsing and MLIR generation."""

import pytest

from approx_runtime.cpp_annotation import (
    AnnotationSyntaxError,
    parse_cpp_annotations,
    generate_cpp_annotation_mlir,
)


def test_parse_block_annotation_identity_default():
    source = """
// @approx:decision_tree {
//   transform_type: func_substitute
//   thresholds: [5, 10]
//   decisions: [0, 1, 2]
// }
int my_kernel(float* data, int n, int state) { return n; }
"""
    anns = parse_cpp_annotations(source)
    assert len(anns) == 1
    ann = anns[0]
    assert ann.func_name == "my_kernel"
    assert ann.transform_type == "func_substitute"
    assert ann.state_indices == [2]
    assert ann.state_function == "__identity_my_kernel"
    assert ann.thresholds == [5, 10]
    assert ann.decisions == [0, 1, 2]


def test_parse_single_line_annotation_with_state_indices():
    source = """
// @approx: loop_perforate state=[-1] thresh=[1] dec=[0,1] state_fn=getState
double foo(const double* x, int len, int state) { return 0.0; }
"""
    anns = parse_cpp_annotations(source)
    ann = anns[0]
    assert ann.func_name == "foo"
    assert ann.state_indices == [2]
    assert ann.state_function == "getState"


def test_parse_negative_indices_multiple():
    source = """
// @approx:decision_tree {
//   transform_type: task_skipping
//   thresholds: [2]
//   decisions: [0, 1]
//   state_indices: [-2, -1]
// }
void bar(int a, int b, int state) { }
"""
    anns = parse_cpp_annotations(source)
    ann = anns[0]
    assert ann.state_indices == [1, 2]


def test_invalid_transform_type_raises():
    source = """
// @approx:decision_tree { transform_type: invalid thresholds: [1] decisions: [0,1] }
int bad(int x, int state) { return x; }
"""
    with pytest.raises(AnnotationSyntaxError):
        parse_cpp_annotations(source)


def test_thresholds_decisions_mismatch_raises():
    source = """
// @approx:decision_tree { transform_type: loop_perforate thresholds: [1,2] decisions: [0,1] }
int bad2(int x, int state) { return x; }
"""
    with pytest.raises(AnnotationSyntaxError):
        parse_cpp_annotations(source)


def test_generate_mlir_includes_convert_to_call():
    source = """
// @approx:decision_tree {
//   transform_type: func_substitute
//   thresholds: [1]
//   decisions: [0, 1]
// }
int kernel(int x, int state) { return x; }
"""
    anns = parse_cpp_annotations(source)
    mlir = generate_cpp_annotation_mlir(anns)
    assert "approx.util.annotation.decision_tree" in mlir
    assert "approx.util.annotation.convert_to_call" in mlir
    assert 'state_function = "__identity_kernel"' in mlir


def test_parse_multiline_signature_nested_types():
    source = """
// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [3]
//   decisions: [0, 1]
// }
int
complex_fn(
    int (*cb)(int, int),
    const char *name,
    int state
) { return cb(1, 2); }
"""
    anns = parse_cpp_annotations(source)
    assert len(anns) == 1
    ann = anns[0]
    assert ann.func_name == "complex_fn"
    assert ann.state_indices == [2]


def test_parse_block_annotation_injects_annotations_text():
    source = """
// @approx:decision_tree {
//   transform_type: task_skipping
//   thresholds: [2]
//   decisions: [0, 1]
// }
void f(int a, int state) { }
"""
    anns = parse_cpp_annotations(source)
    mlir = generate_cpp_annotation_mlir(anns)
    assert "approx.util.annotation.decision_tree" in mlir
    assert "func_name = \"f\"" in mlir


def test_parse_multiple_annotations_in_file():
    source = """
// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [1]
//   decisions: [0, 1]
// }
int a(int x, int state) { return x; }

// @approx:decision_tree {
//   transform_type: task_skipping
//   thresholds: [2]
//   decisions: [0, 1]
// }
void b(int x, int state) { }
"""
    anns = parse_cpp_annotations(source)
    assert len(anns) == 2
    names = [a.func_name for a in anns]
    assert "a" in names
    assert "b" in names


def test_single_line_annotation_parsing():
    source = """
// @approx: func_substitute state=[-1] thresh=[4] dec=[0,1] state_fn=getState
int foo(int x, int y, int state) { return x + y; }
"""
    anns = parse_cpp_annotations(source)
    ann = anns[0]
    assert ann.transform_type == "func_substitute"
    assert ann.state_indices == [2]
    assert ann.state_function == "getState"
    assert ann.thresholds == [4]
    assert ann.decisions == [0, 1]


def test_threshold_bounds_lengths_must_match():
    source = """
// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [1, 2]
//   decisions: [0, 1, 2]
//   thresholds_lower: [0]
// }
int bad_bounds(int x, int state) { return x; }
"""
    with pytest.raises(AnnotationSyntaxError):
        parse_cpp_annotations(source)


def test_state_indices_out_of_bounds():
    source = """
// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [1]
//   decisions: [0, 1]
//   state_indices: [5]
// }
int bad_idx(int x, int state) { return x; }
"""
    with pytest.raises(AnnotationSyntaxError):
        parse_cpp_annotations(source)


def test_identity_helpers_are_emitted_once():
    source = """
// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [1]
//   decisions: [0, 1]
// }
int f1(int x, int state) { return x; }

// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [2]
//   decisions: [0, 1]
// }
int f2(int x, int state) { return x; }
"""
    anns = parse_cpp_annotations(source)
    mlir = generate_cpp_annotation_mlir(anns)
    assert "func.func @__identity_f1" in mlir
    assert "func.func @__identity_f2" in mlir


def test_parse_real_bm25_score_term_over_docs():
    source = """
// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [10]
//   decisions: [0, 1]
// }
void score_term_over_docs(
    const char *lower_term,
    char **lower_corpus,
    const double *doc_lengths,
    double avg_doc_len,
    double idf,
    DocumentScore *scores,
    int num_docs,
    int state
){
    for (int i = 0; i < num_docs; ++i) {
        scores[i].score += idf;
    }
}
"""
    anns = parse_cpp_annotations(source)
    ann = anns[0]
    assert ann.func_name == "score_term_over_docs"
    assert ann.state_indices == [7]


def test_parse_real_lavamd_neighbor_box_accumulate():
    source = """
// @approx:decision_tree {
//   transform_type: func_substitute
//   thresholds: [5]
//   decisions: [0, 1]
// }
void neighbor_box_accumulate(int pi_idx, int bx, fp a2,
                                    box_str* box, FOUR_VECTOR* rv, fp* qv,
                                    FOUR_VECTOR* fv_particle, int state){
    int k;
    for (k = 0; k < box[bx].nn; k++) {
        int nb  = box[bx].nei[k].number;
        int off = box[nb].offset;
        (void)off;
    }
}
"""
    anns = parse_cpp_annotations(source)
    ann = anns[0]
    assert ann.func_name == "neighbor_box_accumulate"
    assert ann.state_indices == [7]


# ---- Pragma-based annotation tests ----


def test_pragma_single_line():
    source = """
#pragma approx decision_tree transform_type=func_substitute state_indices=[-1] thresholds=[4] decisions=[0,1] state_fn=getState
int foo(int x, int y, int state) { return x + y; }
"""
    anns = parse_cpp_annotations(source)
    assert len(anns) == 1
    ann = anns[0]
    assert ann.func_name == "foo"
    assert ann.transform_type == "func_substitute"
    assert ann.state_indices == [2]
    assert ann.state_function == "getState"
    assert ann.thresholds == [4]
    assert ann.decisions == [0, 1]


def test_pragma_multiline_backslash():
    source = """
#pragma approx decision_tree \\
    transform_type=func_substitute \\
    state_indices=[7] \\
    state_function=approx_state_identity \\
    thresholds=[2000] \\
    thresholds_lower=[1] \\
    thresholds_upper=[40] \\
    decisions=[0,0] \\
    decision_values=[0,1,2]
void score_term_over_docs(
    const char *lower_term,
    char **lower_corpus,
    const double *doc_lengths,
    double avg_doc_len,
    double idf,
    int *scores,
    int num_docs,
    int state
){ }
"""
    anns = parse_cpp_annotations(source)
    assert len(anns) == 1
    ann = anns[0]
    assert ann.func_name == "score_term_over_docs"
    assert ann.transform_type == "func_substitute"
    assert ann.state_indices == [7]
    assert ann.state_function == "approx_state_identity"
    assert ann.thresholds == [2000]
    assert ann.thresholds_lower == [1]
    assert ann.thresholds_upper == [40]
    assert ann.decisions == [0, 0]
    assert ann.decision_values == [0, 1, 2]


def test_pragma_loop_perforate():
    source = """
#pragma approx decision_tree transform_type=loop_perforate state_indices=[5] thresholds=[8] decisions=[0,0] decision_values=[0,1,2]
int choose_cluster(const double *point, double **centroids, int k, int dim, int dist_state, int state) {
    return 0;
}
"""
    anns = parse_cpp_annotations(source)
    ann = anns[0]
    assert ann.func_name == "choose_cluster"
    assert ann.transform_type == "loop_perforate"
    assert ann.state_indices == [5]


def test_pragma_task_skipping():
    source = """
#pragma approx decision_tree transform_type=task_skipping state_indices=[2] thresholds=[2] decisions=[1,2] decision_values=[0,1,2]
void model_choose(int input, int* output, int state) { }
"""
    anns = parse_cpp_annotations(source)
    ann = anns[0]
    assert ann.func_name == "model_choose"
    assert ann.transform_type == "task_skipping"
    assert ann.decisions == [1, 2]


def test_pragma_and_comment_mixed():
    source = """
#pragma approx decision_tree transform_type=loop_perforate state_indices=[-1] thresholds=[1] decisions=[0,1]
int a(int x, int state) { return x; }

// @approx:decision_tree {
//   transform_type: task_skipping
//   thresholds: [2]
//   decisions: [0, 1]
// }
void b(int x, int state) { }
"""
    anns = parse_cpp_annotations(source)
    assert len(anns) == 2
    assert anns[0].func_name == "a"
    assert anns[0].transform_type == "loop_perforate"
    assert anns[1].func_name == "b"
    assert anns[1].transform_type == "task_skipping"


def test_pragma_generates_same_mlir_as_comment():
    pragma_source = """
#pragma approx decision_tree transform_type=func_substitute thresholds=[1] decisions=[0,1]
int kernel(int x, int state) { return x; }
"""
    comment_source = """
// @approx:decision_tree {
//   transform_type: func_substitute
//   thresholds: [1]
//   decisions: [0, 1]
// }
int kernel(int x, int state) { return x; }
"""
    pragma_anns = parse_cpp_annotations(pragma_source)
    comment_anns = parse_cpp_annotations(comment_source)
    pragma_mlir = generate_cpp_annotation_mlir(pragma_anns)
    comment_mlir = generate_cpp_annotation_mlir(comment_anns)
    assert pragma_mlir == comment_mlir
