configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(APPROX_OPT_TEST_DEPENDS
        FileCheck count not
        approx-opt
        )

add_lit_testsuite(check-approx-opt "Running the approx-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${APPROX_OPT_TEST_DEPENDS}
        )
set_target_properties(check-approx-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(APPROX_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${APPROX_OPT_TEST_DEPENDS})
