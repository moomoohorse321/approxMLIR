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
    static void moveMarkedOpsToNewBlock(Operation* op, ConversionPatternRewriter &rewriter) {
        Operation* clonedOp = rewriter.clone(*op);
        rewriter.insert(clonedOp);
        rewriter.eraseOp(op);
        return;
    }

    /**
     * https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
     */
    static void track_producers(Operation &op, std::vector<std::pair<Type, Location>> &producers, std::vector<Value> &producerVals) {
        for (Value operand : op.getOperands()) {
            bool found = false;
            
            for(auto _producerVal: producerVals) {
                if(_producerVal == operand) {
                    found = true;
                    break;
                }
            }
            
            if(found) {
                continue;
            }

            producerVals.push_back(operand);
            if (Operation *producer = operand.getDefiningOp()) {
                producers.push_back(std::make_pair(operand.getType(), producer->getLoc()));
            } else {
                // If there is no defining op, the Value is necessarily a Block
                // argument.
                auto blockArg = operand.cast<BlockArgument>();
                producers.push_back(std::make_pair(blockArg.getType(), blockArg.getLoc()));
            }
        }
    }

    static void track_users(Operation &op, std::vector<Operation*> &users, std::vector<Value> &results) {
        for (auto indexedResult : llvm::enumerate(op.getResults())) {
            Value result = indexedResult.value();
            results.push_back(result);
        }
    }

    static void generate_approx_outs(std::vector<Operation*> inRegionOps, std::vector<Operation*> &users, std::vector<Value> &results, std::vector<Type> &resultTypes) {
        std::vector<Value> _results;
        
        for(auto result : results) {
            bool usedOutsideRegion = false;
            if(result.use_empty()) {
                continue;
            }
            for(Operation* inRegionOp : inRegionOps) {
                for (Operation *userOp : result.getUsers()) {
                    if(userOp != inRegionOp) {
                        usedOutsideRegion = true;
                        break;
                    }
                }
                if(usedOutsideRegion) {
                    break;
                }
            }
            if(usedOutsideRegion) {
                _results.push_back(result);
            }
        }

        results.clear();
        for(auto result : _results) {
            results.push_back(result);
            resultTypes.push_back(result.getType());
        }
    } 

    static void dump_region(Region* region) {
        for (Block &block : region->getBlocks()) {
            block.dump();
        }
    }

    struct AnnocationOpConversion : public OpConversionPattern<func::CallOp> 
    {
        using OpConversionPattern<func::CallOp>::OpConversionPattern;
        
        /**
         * For each function, we scan through it, finding annotations that mark the region to insert in our knobOp (in this case, approxForOp).
         */
        LogicalResult
        matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor, 
            ConversionPatternRewriter &rewriter) const final {
            // std::vector<Operation*> opsToMove;
            // std::vector<std::pair<Type, Location>> producers;
            // std::vector<Value> producerVals;
            // std::vector<Operation*> users;
            // std::vector<Value> results;
            // std::vector<Type> resultTypes;
            
            // Operation* start_knob = nullptr;
            // Operation* end_knob = nullptr;


            // // decision tree arguments
            // Value state;
            // Value num_thresholds;
            // Value thresholds_uppers;
            // Value thresholds_lowers;
            // Value decision_values;
            // Value thresholds;
            // Value decisions;

            // bool in_knob = false;
            // Region* region = nullptr;
            
            // callOp.dump();
            
            // StringRef callee = callOp.getCallee();
            
            // llvm::outs() << callee << "\n";

            // // We only look at the start and get the parent region.
            // if(callee.compare(StringRef("knob_start")) != 0) {
            //     llvm::outs() << "not knob_start\n";
            //     return failure();
            // }
            
            // // callOp.dump();
            
            
            // region = callOp->getParentRegion();

            // dump_region(region);
            

            // // step 1: mark the ops to move (we only look at the first knob_start and knob_end)
            // for (Block &block : region->getBlocks()) {
            //     for (Operation &op : block.getOperations()) {
            //         // check if hit the user annotation "func.call @knob_start() : () -> ()"
            //         if (!in_knob) {
            //             auto _callOp = dyn_cast<func::CallOp>(op);
            //             if (!_callOp) {
            //                 continue;
            //             }
            //             if (_callOp.getLoc() == callOp.getLoc()) {
            //                 in_knob = true;
            //                 start_knob = &op; // we will later replace this will the approxOp
            //             }

            //             if (_callOp.getCallee().compare(StringRef("knob_end")) == 0) {
            //                 in_knob = false;
            //                 end_knob = &op; // we will later replace this will the approxOp
            //             }
            //             continue;
            //         }
            //         if(in_knob) {
            //             opsToMove.push_back(&op);
            //         }
            //     }
            // }
            
            // llvm::outs() << "here\n";

            // auto savedInsertPoint = rewriter.saveInsertionPoint();
            
            // if(start_knob == nullptr || end_knob == nullptr) {
            //     return failure();
            // }

            // start_knob->dump();
            // end_knob->dump();

            // // First emit Ops to the new block, then move the new block to the body of the approxForOp
            // Block* tempBlock = rewriter.createBlock(region, {}, std::nullopt, std::nullopt); // side-effect: change insert point to the end of the created block

            // generate_approx_outs(opsToMove, users, results, resultTypes);

            // // step 2: move Ops to the new block
            // for (auto* _op : opsToMove) {
            //     Operation & op = *_op;
            //     if(dyn_cast<func::CallOp>(op) && dyn_cast<func::CallOp>(op).getCallee().compare(StringRef("decision_tree")) == 0) {
            //         // parse func.call @decision_tree(%i_f32, %num_thresholds, %thresholds_uppers, %thresholds_lowers, %decision_values, %thresholds, %decisions) : (f32, i32, tensor<3xf32>, tensor<3xf32>, tensor<3xi32>, tensor<3xf32>, tensor<3xi32>) -> () 
            //         // then create a decideOp, inserting it into the tempBlock
            //         auto callOp = dyn_cast<func::CallOp>(op);
            //         // parse the inputs
            //         state = callOp.getOperand(0);
            //         num_thresholds = callOp.getOperand(1);
            //         thresholds_uppers = callOp.getOperand(2);
            //         thresholds_lowers = callOp.getOperand(3);
            //         decision_values = callOp.getOperand(4);
            //         thresholds = callOp.getOperand(5);
            //         decisions = callOp.getOperand(6);
                    
            //         // create the decideOp
            //         // then, we insert the decideOp based on the annotation func.call @decision_tree(%i_f32, %num_thresholds, %thresholds_uppers, %thresholds_lowers, %decision_values, %thresholds, %decisions) : (f32, i32, tensor<3xf32>, tensor<3xf32>, tensor<3xi32>, tensor<3xf32>, tensor<3xi32>) -> ()
            //         rewriter.replaceOpWithNewOp<approxMLIR::decideOp>(_op, 
            //             state, num_thresholds, thresholds_uppers, thresholds_lowers, decision_values, thresholds, decisions);
                            
            //     }
            //     // then we keep track of all the uses for the internal Ops
            //     // also we keep track of all the defs for the internal Ops
            //     // such uses and defs will be used to create the KnobOp
            //     track_users(op, users, results);
            //     track_producers(op, producers, producerVals);
            // }
            
            // for (auto* op : opsToMove) {
            //     // here we use rewriter to move the Ops between the 2 annotations in the new block
            //     moveMarkedOpsToNewBlock(op, rewriter);   
            // }
            // moveMarkedOpsToNewBlock(end_knob, rewriter);
            
            // rewriter.replaceOpWithNewOp<scf::YieldOp>(end_knob, results);

            // // step 3: emit approxOp (producers, state, rf, QoS_in, QoS_out -> users) by replacing the start_knob
            // for(auto arg: producers) {
            //     tempBlock->addArgument(arg.first, arg.second);
            // }
            // rewriter.replaceOpWithNewOp<approxMLIR::approxForOp>(start_knob, TypeRange(ArrayRef<Type>(resultTypes)), state, state, state, state, producerVals); // temporarily set rf, QoS_in, and QoS_out to 0 (todo)
            
            // // step 4: move the tempBlock to the approxOp
            // Region & approxRegion = start_knob->getRegion(0);
            // Block* approxBlock = & approxRegion.back();
            // rewriter.mergeBlocks(tempBlock, approxBlock, std::nullopt);
            
            // // todo: insert the checkerOp
            
            // // step 5: finish
            // rewriter.restoreInsertionPoint(savedInsertPoint);
            return success();
        }
    };

    struct EmitApproxPass
    : public EmitApproxPassBase<EmitApproxPass> {


        void runOnOperation() override {
            std::vector<Operation*> funcsToDo;
            getOperation()->walk([&](Operation *op) {
              if (isa<func::FuncOp>(op)) {
                funcsToDo.push_back(op);
              }
            });

            RewritePatternSet patterns(&getContext());
            patterns.add<AnnocationOpConversion>(&getContext());
            GreedyRewriteConfig config;
            config.maxIterations = 1;
            if(failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config))) {
                return signalPassFailure();
            }
        }
    };
}

std::unique_ptr<Pass> mlir::approxMLIR::createEmitApproxPass() {
    return std::make_unique<EmitApproxPass>();
}

namespace mlir{
    namespace approxMLIR {
        void registerEmitApproxPass() {
            PassRegistration<EmitApproxPass>();
        }
    } // namespace approxMLIR
} // namespace mlir
