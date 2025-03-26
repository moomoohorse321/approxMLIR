add_mlir_dialect_library(MLIRapproxMLIR
        approxMLIRDialect.cpp
        approxMLIROps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/approxMLIR

        DEPENDS
        MLIRapproxMLIROpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
