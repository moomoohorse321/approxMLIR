add_mlir_dialect_library(MLIRapproxMLIR
        approxMLIRDialect.cpp
        approxMLIROps.cpp

        ADDITIONAL_HEADER_DIRS
        ${APPROX_SOURCE_DIR}/include/approxMLIR

        DEPENDS
        MLIRapproxMLIROpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
)

# --- BEGIN EXPLICIT INTERFACE INCLUDE DIRECTORY ---
# This explicitly tells CMake that any target linking against MLIRapproxMLIR
# (like polygeist-opt) should get these include directories.
message(STATUS "Setting INTERFACE_INCLUDE_DIRECTORIES for MLIRapproxMLIR to: ${APPROX_SOURCE_DIR}/include")
target_include_directories(MLIRapproxMLIR INTERFACE
    "$<BUILD_INTERFACE:${APPROX_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${APPROX_BINARY_DIR}/include>"
    # If you also have public headers from this library that are generated into the build directory:
    # For example, if Dialect.h.inc was public and in ${APPROX_BINARY_DIR}/include/approxMLIR/
    # and you wanted to allow #include "approxMLIR/Dialect.h.inc"
    # then you'd add: "$<BUILD_INTERFACE:${APPROX_BINARY_DIR}/include>"
    # For now, let's stick to the source includes.
)
# --- END EXPLICIT INTERFACE INCLUDE DIRECTORY ---

add_subdirectory(Passes)