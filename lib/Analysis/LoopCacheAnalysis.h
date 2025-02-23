#ifndef LIB_ANALYSIS_LOOP_CACHE_H_
#define LIB_ANALYSIS_LOOP_CACHE_H_

#include <optional>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace mlir::foo {

///
class LoopCacheAnalysis
    : public PassWrapper<LoopCacheAnalysis, OperationPass<func::FuncOp>> {
public:
  [[nodiscard]] StringRef getArgument() const final {
    return "loop-cache-analysis";
  }
  [[nodiscard]] StringRef getDescription() const final {
    return "Analyses the cache-friendliness of loop nests";
  }

  void runOnOperation() override;

  static constexpr int CacheLineSizeSymbolIndex = 0;
  static constexpr int CacheSizeSymbolIndex = 1;

private:
  struct CacheCostResult {
    affine::AffineForOp LoopOp;
    llvm::SmallVector<AffineExpr> CacheCost;
  };

  //
  static llvm::Expected<llvm::SmallVector<AffineExpr>>
  getLoopCacheCost(affine::AffineForOp *LoopOp);
  // Checks if this operation represents one of the loop types amenable to our
  // analysis, e.g. `scf::ForOp`, `affine::AffineForOp`.
  static bool isLoopOp(const Operation *Op);

  static std::optional<affine::AffineForOp> getLoopOp(Operation *Op) {
    if (auto LoopOp = llvm::dyn_cast<affine::AffineForOp>(Op)) {
      return LoopOp;
    }

    return std::nullopt;
  }

  //
  static llvm::Expected<SmallVector<affine::AffineForOp>>
  getLoopNestAsList(affine::AffineForOp *LoopOp);
};

} // namespace mlir::foo

#endif // LIB_ANALYSIS_LOOP_CACHE_H_
