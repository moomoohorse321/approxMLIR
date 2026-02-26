/// ApproxTritonPlugin.cpp — Triton pass plugin entry point
///
/// Registers approxMLIR's dialect and passes so they can be loaded by Triton
/// via TRITON_PASS_PLUGIN_PATH.
///
/// Passes (the full approx pipeline minus legalize-to-stablehlo):
///   1. emit-approx
///   2. emit-management
///   3. config-approx
///   4. pre-emit-transform
///   5. transform-approx
///   6. finalize-approx
///
/// Dialect: "approx" (approxDialect) — registered via the dialect plugin API
/// so Triton's MLIRContext can parse approx.* ops.

#include "approx/Dialect.h"
#include "approx/Ops.h"
#include "approx/Passes/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/Config/llvm-config.h"

// ---------------------------------------------------------------------------
// Triton plugin entry points
//
// Triton's pass plugin loader calls these three C functions (defined in
// triton/Tools/PluginUtils.h) from the shared library:
//   - tritonRegisterPluginPass(passName)
//   - tritonAddPluginPass(pm, passName)
//   - tritonEnumeratePluginPasses(count, names)
// ---------------------------------------------------------------------------

namespace {

/// Pass name constants — these are the names users reference when adding
/// passes to Triton's pipeline.
constexpr const char *kPassNames[] = {
    "emit-approx",
    "emit-management",
    "config-approx",
    "pre-emit-transform",
    "transform-approx",
    "finalize-approx",
};
constexpr uint32_t kNumPasses = sizeof(kPassNames) / sizeof(kPassNames[0]);

std::unique_ptr<mlir::Pass> createPassByName(const char *name) {
  llvm::StringRef n(name);
  if (n == "emit-approx")
    return mlir::approx::createEmitApproxPass();
  if (n == "emit-management")
    return mlir::approx::createEmitManagementPass();
  if (n == "config-approx")
    return mlir::approx::createConfigApproxPass();
  if (n == "pre-emit-transform")
    return mlir::approx::createPreEmitTransformationPass();
  if (n == "transform-approx")
    return mlir::approx::createTransformApproxPass();
  if (n == "finalize-approx")
    return mlir::approx::createFinalizeApproxPass();
  return nullptr;
}

} // namespace

// ---------------------------------------------------------------------------
// Plugin API (exported as C symbols from the shared library)
// ---------------------------------------------------------------------------

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName) {
  auto pass = createPassByName(passName);
  if (!pass)
    return TP_GENERIC_FAILURE;
  // Register the pass so it can be found by name in the pipeline.
  mlir::registerPass([passName]() { return createPassByName(passName); });
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName) {
  auto pass = createPassByName(passName);
  if (!pass)
    return TP_GENERIC_FAILURE;
  pm->addPass(std::move(pass));
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames) {
  if (!passCount)
    return TP_GENERIC_FAILURE;
  *passCount = kNumPasses;
  if (passNames) {
    for (uint32_t i = 0; i < kNumPasses; ++i)
      passNames[i] = kPassNames[i];
  }
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonEnumeratePluginDialects(uint32_t *dialectCount,
                              const char **dialectNames) {
  if (!dialectCount)
    return TP_GENERIC_FAILURE;
  *dialectCount = 1;
  if (!dialectNames)
    return TP_SUCCESS;
  dialectNames[0] = "approx";
  return TP_SUCCESS;
}

TRITON_PLUGIN_API_TYPE(::mlir::DialectPluginLibraryInfo)
tritonGetDialectPluginInfo(const char * /*name*/) {
  return {MLIR_PLUGIN_API_VERSION, "ApproxPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<mlir::approx::approxDialect>();
            mlir::func::registerInlinerExtension(*registry);
          }};
}
