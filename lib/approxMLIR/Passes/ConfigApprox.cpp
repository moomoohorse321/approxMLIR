#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "approxMLIR/Passes/Passes.h"
#include "approxMLIR/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
// set
#include <set>

using namespace mlir;
using namespace approxMLIR;

namespace {

    struct ConifgureNN4Func : public OpRewritePattern<func::FuncOp> 
    {
        using OpRewritePattern<func::FuncOp>::OpRewritePattern;
        
        /**
         * For each function, we look at the module to find its approximate version (a NN model). 
         * Currently the NN model will be named as approx_<original_func_name>.
         * We simply erase the body and replace it with a call. Then inline the call.
         */
        LogicalResult
        matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const final {

        }
    };

    struct ConfigApproxPass
    : public ConfigApproxPassBase<ConfigApproxPass> {

        void runOnOperation() override {
            RewritePatternSet patterns(&getContext());
            // patterns.add<>(&getContext());
            (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)));
        }
    };
}

std::unique_ptr<Pass> mlir::approxMLIR::createConfigApproxPass() {
    return std::make_unique<ConfigApproxPass>();
}

namespace mlir{
    namespace approxMLIR {
        void registerConfigApproxPass() {
            PassRegistration<ConfigApproxPass>();
        }
    } // namespace approxMLIR
} // namespace mlir
