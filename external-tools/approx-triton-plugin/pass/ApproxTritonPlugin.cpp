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

using namespace mlir::triton;

// ---------------------------------------------------------------------------
// Pass creation helper
// ---------------------------------------------------------------------------

namespace {

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
// Callback functions for each pass
// ---------------------------------------------------------------------------

#define DEFINE_PASS_CALLBACKS(funcName, passName)                               \
  static void add_##funcName(mlir::PassManager *pm,                            \
                             const std::vector<std::string> &) {               \
    pm->addPass(createPassByName(passName));                                    \
  }                                                                            \
  static void register_##funcName() {                                          \
    mlir::registerPass([]() { return createPassByName(passName); });           \
  }

DEFINE_PASS_CALLBACKS(emit_approx,      "emit-approx")
DEFINE_PASS_CALLBACKS(emit_management,  "emit-management")
DEFINE_PASS_CALLBACKS(config_approx,    "config-approx")
DEFINE_PASS_CALLBACKS(pre_emit_transform, "pre-emit-transform")
DEFINE_PASS_CALLBACKS(transform_approx, "transform-approx")
DEFINE_PASS_CALLBACKS(finalize_approx,  "finalize-approx")

#undef DEFINE_PASS_CALLBACKS

// ---------------------------------------------------------------------------
// Dialect registration callback
// ---------------------------------------------------------------------------

static void registerApproxDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::approx::approxDialect>();
  mlir::func::registerInlinerExtension(*registry);
}

// ---------------------------------------------------------------------------
// Plugin entry point (new Triton PluginInfo API)
// ---------------------------------------------------------------------------

static const char *PLUGIN_NAME = "ApproxPlugin";
static const char *VERSION = "0.1.0";

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {
      {"emit-approx",       VERSION, add_emit_approx,       register_emit_approx},
      {"emit-management",   VERSION, add_emit_management,   register_emit_management},
      {"config-approx",     VERSION, add_config_approx,     register_config_approx},
      {"pre-emit-transform", VERSION, add_pre_emit_transform, register_pre_emit_transform},
      {"transform-approx",  VERSION, add_transform_approx,  register_transform_approx},
      {"finalize-approx",   VERSION, add_finalize_approx,   register_finalize_approx},
  };

  static plugin::DialectInfo dialects[] = {
      {"approx", VERSION, registerApproxDialect},
  };

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      PLUGIN_NAME,
      VERSION,
      passes,
      sizeof(passes) / sizeof(passes[0]),
      dialects,
      sizeof(dialects) / sizeof(dialects[0]),
      nullptr, // no custom ops
      0,
  };
  return &info;
}
