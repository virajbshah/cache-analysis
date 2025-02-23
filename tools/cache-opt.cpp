#include "lib/Analysis/LoopCacheAnalysis.h"
#include "lib/Analysis/LoopCounter.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry Registry;
  mlir::registerAllDialects(Registry);
  mlir::registerAllPasses();

  mlir::PassRegistration<mlir::foo::LoopCacheAnalysis>();
  mlir::PassRegistration<mlir::foo::LoopCounterPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Cache Analysis Driver", Registry));
}