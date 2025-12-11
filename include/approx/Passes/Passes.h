#ifndef APPROX_DIALECT_APPROX_PASSES_H
#define APPROX_DIALECT_APPROX_PASSES_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "approx/Dialect.h"

#include <memory>

/**
 * Pass.td defines Ops required for approx dialect
 * 
 * Passes achieve 2 goals:
 * 1. Codegen from exact Ops [Represent approximate transformation]
 * 2. Lower approxOps to Exact Ops [perform approximate transformation]
 */

namespace mlir {
    class PatternRewriter;
    class RewritePatternSet;
    class DominanceInfo;
    namespace approx {
        std::unique_ptr<Pass> createEmitApproxPass();
        std::unique_ptr<Pass> createConfigApproxPass();
        std::unique_ptr<Pass> createPreEmitTransformationPass();
        std::unique_ptr<Pass> createTransformApproxPass();
        // void registerEmitApproxPass();
        // void registerConfigApproxPass();
   } // namespace approx
} // namespace mlir

void fully2ComposeAffineMapAndOperands(
    mlir::PatternRewriter &rewriter, mlir::AffineMap *map,
    llvm::SmallVectorImpl<mlir::Value> *operands, mlir::DominanceInfo &DI);
bool isValidIndex(mlir::Value val);

namespace mlir {
    // Forward declaration from Dialect.h
    template <typename ConcreteDialect>
    void registerDialect(DialectRegistry &registry);
    
    namespace arith {
    class ArithDialect;
    } // end namespace arith
    
    namespace omp {
    class OpenMPDialect;
    } // end namespace omp
    
    namespace scf {
    class SCFDialect;
    } // end namespace scf
    
    namespace cf {
    class ControlFlowDialect;
    } // end namespace cf
    
    namespace math {
    class MathDialect;
    } // end namespace math
    
    namespace memref {
    class MemRefDialect;
    } // end namespace memref
    
    namespace func {
    class FuncDialect;
    }
    
    namespace affine {
    class AffineDialect;
    }
    
    namespace LLVM {
    class LLVMDialect;
    }

    #define GEN_PASS_DECL
    #define GEN_PASS_REGISTRATION
    #include "approx/Passes/Passes.h.inc"
} // end namespace mlir

#endif // APPROX_DIALECT_APPROX_PASSES_H