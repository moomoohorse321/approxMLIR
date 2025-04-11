// #include "PassDetails.h"

// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/LLVMIR/LLVMTypes.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/Support/LogicalResult.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// #include "approxMLIR/Passes/Passes.h"
// #include "approxMLIR/Passes/Utils.h"
// #include "llvm/ADT/STLExtras.h"
// #include <memory>

// using namespace mlir;
// using namespace approxMLIR;

// namespace {

//     struct ConvertSCFToApproxPass
//     : public ConvertSCFToApproxPassBase<ConvertSCFToApproxPass> {

//         LogicalResult buildForErrorKnob()

//         void runOnOperation() override {
//             std::vector<Operation*> funcsToDo;
//             getOperation()->walk([&](Operation *op) {
//               if (isa<func::FuncOp>(op)) {
//                 funcsToDo.push_back(op);
//               }
//             });

//         }
//     };
// }
// std::unique_ptr<Pass> mlir::approxMLIR::createConvertSCFToApproxPass() {
//     return std::make_unique<ConvertSCFToApproxPass>();
// }

// namespace mlir{
//     namespace approxMLIR {
//         void registerConvertSCFToApproxPass() {
//             PassRegistration<ConvertSCFToApproxPass>();
//         }
//     } // namespace approxMLIR
// } // namespace mlir