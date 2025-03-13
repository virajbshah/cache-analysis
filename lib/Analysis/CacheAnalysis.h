#ifndef LIB_ANALYSIS_CACHEANALYSIS_H_
#define LIB_ANALYSIS_CACHEANALYSIS_H_

#include "CacheAnalysisGraph.h"
#include "IndexComputationGraph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::foo {

class CacheAnalysis
    : public PassWrapper<CacheAnalysis, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "cache-analysis"; }
  StringRef getDescription() const override {
    return "Soon to be cache analysis";
  }

  void runOnOperation() override;

private:
  CacheAnalysisGraph AnalysisGraph;
  IndexComputationGraph IndexComputations;
};

} // namespace mlir::foo

#endif // LIB_ANALYSIS_CACHEANALYSIS_H_
