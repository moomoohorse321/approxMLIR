configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(APPROXMLIR_OPT_TEST_DEPENDS
        FileCheck count not
        approxMLIR-opt
        )

add_lit_testsuite(check-approxMLIR-opt "Running the approxMLIR-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${APPROXMLIR_OPT_TEST_DEPENDS}
        )
set_target_properties(check-approxMLIR-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(APPROXMLIR_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${APPROXMLIR_OPT_TEST_DEPENDS})
