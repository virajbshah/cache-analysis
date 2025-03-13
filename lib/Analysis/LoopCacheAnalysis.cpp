#include "LoopCacheAnalysis.h"

#include <cassert>
#include <cstdint>
#include <system_error>
#include <tuple>
#include <vector>

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "loop-cache-analysis"

namespace mlir::foo {

void LoopCacheAnalysis::runOnOperation() {
  SmallVector<CacheCostResult> LoopCacheCosts;

  getOperation()->walk<WalkOrder::PreOrder>([&](Operation *Op) {
    if (auto LoopOp = getLoopOp(Op)) {
      auto CacheCost = getLoopCacheCost(&*LoopOp);
      if (llvm::Error Err = CacheCost.takeError()) {
        llvm::outs() << "Skipping loop " << (*LoopOp)->getName() << "@"
                     << (*LoopOp)->getLoc() << ": " << Err << '\n';
        return WalkResult::skip();
      }

      LoopCacheCosts.push_back({*LoopOp, *CacheCost});
      return WalkResult::skip(); // We only want to visit outermost loops.
    }
    return WalkResult::advance();
  });

  for (const auto &Result : LoopCacheCosts) {
    llvm::outs() << Result.LoopOp->getName() << "@" << Result.LoopOp->getLoc()
                 << " has cache costs:\n";
    for (auto CacheCostElem : Result.CacheCost) {
      llvm::outs() << CacheCostElem << '\n';
    }
  }
}

namespace {

template <typename LoadOrStoreOpTy>
AffineMap getAffineMap(LoadOrStoreOpTy LoadOrStoreOp) {
  return AffineMap::getMultiDimIdentityMap(
      LoadOrStoreOp.getMemRefType().getShape().size(),
      LoadOrStoreOp.getContext());
}

template <> AffineMap getAffineMap(affine::AffineLoadOp LoadOrStoreOp) {
  return LoadOrStoreOp.getAffineMap();
}

template <> AffineMap getAffineMap(affine::AffineStoreOp LoadOrStoreOp) {
  return LoadOrStoreOp.getAffineMap();
}

} // namespace

llvm::Expected<llvm::SmallVector<AffineExpr>>
LoopCacheAnalysis::getLoopCacheCost(affine::AffineForOp *LoopOp) {
  auto LoopNest = getLoopNestAsList(LoopOp);
  if (llvm::Error Err = LoopNest.takeError()) {
    return Err;
  }
  auto &InnerMostLoop = LoopNest->back();

  llvm::SmallVector<AffineExpr> CacheCosts;

  llvm::DenseMap<Value, unsigned> IndVarIndex;
  for (auto [Level, Loop] : llvm::enumerate(*LoopNest)) {
    IndVarIndex[Loop.getInductionVar()] = Level;
  }

  llvm::SmallVector<AffineExpr> InnerMostStep = getAffineConstantExprs(
      std::vector<int64_t>(LoopNest->size(), 0), InnerMostLoop->getContext());
  InnerMostStep.back() = getAffineConstantExpr(InnerMostLoop.getStepAsInt(),
                                               InnerMostLoop->getContext());

  llvm::DenseMap<void *, llvm::SmallVector<AffineExpr>> RefGroups;
  auto ProcessAffineLoadOrStoreOp = [&](auto LoadOrStoreOp) {
    // Let's skip more interesting layouts for now.
    if (!LoadOrStoreOp.getMemRefType().getLayout().isIdentity()) {
      LLVM_DEBUG(llvm::outs() << "unsupported layout");
      return;
    }

    // Add symbols representing the cache line size and total cache size
    // at positions 0 and 1 respectively.
    AffineMap MappedIndices = getAffineMap(LoadOrStoreOp).shiftSymbols(2);
    const auto &[Strides, Offset] =
        LoadOrStoreOp.getMemRefType().getStridesAndOffset();

    llvm::SmallVector<unsigned> Levels;
    for (Value Index :
         LoadOrStoreOp.getIndices().take_front(MappedIndices.getNumDims())) {
      unsigned Level = IndVarIndex[Index];
      Levels.push_back(Level);
    }

    auto LevelToIndexMap = AffineMap::getMultiDimMapWithTargets(
        LoopNest->size(), Levels, InnerMostLoop->getContext());
    auto MappedLevels = MappedIndices.compose(LevelToIndexMap);

    AffineExpr LinearizedIndex =
        getAffineConstantExpr(0, LoadOrStoreOp->getContext());
    for (const auto &[MappedLevel, Stride] :
         llvm::zip_equal(MappedLevels.getResults(), Strides)) {
      LinearizedIndex = LinearizedIndex + MappedLevel * Stride;
    }

    void *BasePointer = LoadOrStoreOp.getMemref().getAsOpaquePointer();
    // Assuming the base pointer is aligned.
    AffineExpr CacheLineExpr = LinearizedIndex.floorDiv(getAffineSymbolExpr(
        CacheLineSizeSymbolIndex, LoadOrStoreOp->getContext()));
    if (RefGroups.contains(BasePointer)) {
      bool IsCacheLineReuse = false;
      for (auto &RefGroup : RefGroups[BasePointer]) {
        AffineExpr Distance = CacheLineExpr - RefGroup;
        // TODO: Consider cases where it might be zero based on some
        // constraint on the values of symbols.
        if (Distance == 0) {
          IsCacheLineReuse = true;
          break;
        }
      }
      if (IsCacheLineReuse) {
        // Do not consider this access in the cost computation.
        return;
      }
    } else {
      RefGroups[BasePointer] = llvm::SmallVector<AffineExpr>{CacheLineExpr};
    }

    auto TotalDistance =
        LinearizedIndex.replaceDims(InnerMostStep) *
        (LoadOrStoreOp.getMemRefType().getElementTypeBitWidth() / 8);
    CacheCosts.push_back(TotalDistance);
  };

  for (auto &Op : InnerMostLoop.getOps()) {
    // For now, treat loads and stores similarly.
    if (auto AffineLoadOp = dyn_cast<affine::AffineLoadOp>(Op)) {
      ProcessAffineLoadOrStoreOp(AffineLoadOp);
    } else if (auto AffineStoreOp = dyn_cast<affine::AffineStoreOp>(Op)) {
      ProcessAffineLoadOrStoreOp(AffineStoreOp);
    } else if (auto LoadOp = dyn_cast<memref::LoadOp>(Op)) {
      ProcessAffineLoadOrStoreOp(LoadOp);
    } else if (auto StoreOp = dyn_cast<memref::StoreOp>(Op)) {
      ProcessAffineLoadOrStoreOp(StoreOp);
    }
  }

  return CacheCosts;
}

llvm::Expected<llvm::SmallVector<affine::AffineForOp>>
LoopCacheAnalysis::getLoopNestAsList(affine::AffineForOp *LoopOp) {
  llvm::SmallVector<affine::AffineForOp> LoopNest{*LoopOp};
  for (auto [Depth, ChildLoops] =
           std::tuple(2, LoopOp->getOps<affine::AffineForOp>());
       !ChildLoops.empty();
       ++Depth, ChildLoops = LoopNest.back().getOps<affine::AffineForOp>()) {
    LoopNest.append(ChildLoops.begin(), ChildLoops.end());
    if (LoopNest.size() > Depth) {
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("loop at depth {0} has siblings", Depth),
          std::make_error_code(std::errc::not_supported),
          /* PrintMsgOnly = */ true);
    }
  }

  return LoopNest;
}

} // namespace mlir::foo
