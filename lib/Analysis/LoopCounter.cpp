#include "LoopCounter.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "loop-counter"

namespace mlir::foo {

void LoopCounterPass::runOnOperation() {
  int LoopCount = 0;

  // Traverse the operations in the module.
  getOperation()->walk([&](Operation *Op) {
    if (isa<affine::AffineForOp>(Op) || isa<scf::ForOp>(Op)) {
      ++LoopCount;
    }
  });

  llvm::outs() << "Number of loops in the program: " << LoopCount << "\n";
}

} // namespace mlir::foo
