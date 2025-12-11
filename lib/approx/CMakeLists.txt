add_mlir_dialect_library(MLIRapprox
        approxDialect.cpp
        approxOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${APPROX_SOURCE_DIR}/include/approx

        DEPENDS
        MLIRapproxOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
)

# --- BEGIN EXPLICIT INTERFACE INCLUDE DIRECTORY ---
# This explicitly tells CMake that any target linking against MLIRapprox
# (like polygeist-opt) should get these include directories.
message(STATUS "Setting INTERFACE_INCLUDE_DIRECTORIES for MLIRapprox to: ${APPROX_SOURCE_DIR}/include")
target_include_directories(MLIRapprox INTERFACE
    "$<BUILD_INTERFACE:${APPROX_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${APPROX_BINARY_DIR}/include>"
    # If you also have public headers from this library that are generated into the build directory:
    # For example, if Dialect.h.inc was public and in ${APPROX_BINARY_DIR}/include/approx/
    # and you wanted to allow #include "approx/Dialect.h.inc"
    # then you'd add: "$<BUILD_INTERFACE:${APPROX_BINARY_DIR}/include>"
    # For now, let's stick to the source includes.
)
# --- END EXPLICIT INTERFACE INCLUDE DIRECTORY ---

add_subdirectory(Passes)