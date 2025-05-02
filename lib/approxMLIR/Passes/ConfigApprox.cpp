
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
// #include "approxMLIR/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
// queue
#include <queue>

#include "PassDetails.h"


namespace mlir {
    using namespace approxMLIR;
    
    namespace {
        #define GEN_PASS_DEF_CONFIGAPPROXPASS
        #include "approxMLIR/Passes/Passes.h.inc"

        struct ConifgureNN4Func : public OpRewritePattern<approxMLIR::transformOp> 
        {
            using OpRewritePattern<approxMLIR::transformOp>::OpRewritePattern;
            
        
            static void erase_region(Region* region, PatternRewriter &rewriter) {
                std::queue<Block*> blocksToErase;
                auto try_delete_block = [&] (Block* block) {
                    // the make_early_inc_range is used to ensure that we can safely erase ops
                    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
                        // for an op to be deleted, all its uses must be deleted.
                        if (!op.use_empty()) {
                            // failed, there are some ops that has uses outside the block.
                            return false;
                        } else {
                            rewriter.eraseOp(&op);
                        }
                    }
                    return true;
                };
                // Note: you must first put it in a list, otherwise you will damage iterator
                for (Block &block : llvm::reverse(region->getBlocks())) {
                    blocksToErase.push(&block);
                }

                // when we dequee, we reset.
                // Otherwise we decrement. Once it's zero, it means an infinite loop.
                int errCounter = blocksToErase.size();

                while(!blocksToErase.empty()) {
                    Block* block = blocksToErase.front();
                    blocksToErase.pop();
                    if (try_delete_block(block)) {
                        rewriter.eraseBlock(block);
                        errCounter = blocksToErase.size();
                    } else {
                        blocksToErase.push(block);
                        errCounter--;
                    }
                    if(!blocksToErase.empty() && errCounter <= 0) {
                        block->dump();
                        llvm::errs() << "Error: Infinite loop detected while erasing blocks.\n";
                        return;
                    }
                }

            }

            /**
             * For each function, we look at the module to find its approximate version (a NN model). 
             * Currently the NN model will be named as approx_<original_func_name>.
             * We simply erase the body and inline the body of the approximate function. (The approx function shouldn't be moved)
             */
            LogicalResult
            matchAndRewrite(approxMLIR::transformOp transformOp, PatternRewriter &rewriter) const final {
                
                // auto inserted = rewriter.create<approxMLIR::transformOp>(funcOp.getLoc(), StringRef("NNsubstitute"), 1);
                auto funcOp = transformOp->getParentOfType<func::FuncOp>();
                if (!funcOp) {
                    // rewriter.eraseOp(transformOp);
                    llvm::errs() << "Error: transformOp is not inside a function.\n";
                    return failure();
                }
                auto approxFuncName = "approx_" + funcOp.getName().str();
                // we go to the definition (in `BuiltinOps.h.inc`) to see how to dump the region of ModuleOp
                Region &moduleRegion = funcOp->getParentOfType<ModuleOp>().getBodyRegion();
                
                func::FuncOp approxFunc = nullptr;
                for(Block &block : moduleRegion.getBlocks()) {
                    for(Operation &op : block.getOperations()) {
                        auto funcOp = dyn_cast<func::FuncOp>(op);
                        if (!funcOp) {
                            continue;
                        }
                        if (funcOp.getName() == approxFuncName) {
                            approxFunc = funcOp;
                            break;
                        }
                    }
                }
                if (!approxFunc) {
                    // rewriter.eraseOp(transformOp);
                    llvm::errs() << "Error: Approximate function " << approxFuncName << " not found.\n";
                    return failure(); // No approximate function found, nothing to do.
                }
                llvm::outs() << "Found approximate function: " << approxFunc.getName() << "\n";

                Region &replacedRegion = funcOp.getBody();
                erase_region(&replacedRegion, rewriter);

                rewriter.cloneRegionBefore(approxFunc.getBody(), replacedRegion, funcOp.getBody().end());
                
                // rewriter.eraseOp(transformOp);

                return success();
            }
        };

        struct ConfigApproxPass
        : public impl::ConfigApproxPassBase<ConfigApproxPass> {
            using ConfigApproxPassBase::ConfigApproxPassBase;

            void runOnOperation() override {
                RewritePatternSet patterns(&getContext());
                patterns.add<ConifgureNN4Func>(&getContext());
                GreedyRewriteConfig config;
                config.maxIterations = 1; // to debug
                (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)), 
                    config); // apply the patterns to the operation
            }
        };
    }

    std::unique_ptr<Pass> createConfigApproxPass() {
        return std::make_unique<ConfigApproxPass>();
    }
}


// namespace mlir{
//     namespace approxMLIR {
//         void registerConfigApproxPass() {
//             PassRegistration<ConfigApproxPass>();
//         }
//     } // namespace approxMLIR
// } // namespace mlir
