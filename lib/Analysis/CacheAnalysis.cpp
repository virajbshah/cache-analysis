#include "CacheAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "llvm/Support/Casting.h"

namespace mlir::foo {

void CacheAnalysis::runOnOperation() {
  getOperation()->walk<WalkOrder::PreOrder>([&](Operation *Op) {
    if (auto LoadOp = llvm::dyn_cast<affine::AffineLoadOp>(Op)) {
      AnalysisGraph.getOrAddNode(LoadOp, &IndexComputations);
    } else if (auto StoreOp = llvm::dyn_cast<affine::AffineStoreOp>(Op)) {
      AnalysisGraph.getOrAddNode(StoreOp, &IndexComputations);
    } else if (auto LoadOp = llvm::dyn_cast<memref::LoadOp>(Op)) {
      AnalysisGraph.getOrAddNode(LoadOp, &IndexComputations);
    } else if (auto StoreOp = llvm::dyn_cast<memref::StoreOp>(Op)) {
      AnalysisGraph.getOrAddNode(StoreOp, &IndexComputations);
    }
  });
}

} // namespace mlir::foo