#ifndef LIB_ANALYSIS_LOOP_COUNTER_H_
#define LIB_ANALYSIS_LOOP_COUNTER_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::foo {

class LoopCounterPass
    : public PassWrapper<LoopCounterPass, OperationPass<ModuleOp>> {
public:
  [[nodiscard]] StringRef getArgument() const final { return "loop-counter"; }
  [[nodiscard]] StringRef getDescription() const final {
    return "Counts the number of loops in the program.";
  }

  void runOnOperation() override;
};

} // namespace mlir::foo

#endif // LIB_ANALYSIS_LOOP_COUNTER_H_
