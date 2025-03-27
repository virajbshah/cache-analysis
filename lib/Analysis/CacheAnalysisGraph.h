#ifndef LIB_ANALYSIS_CACHEANALYSISGRAPH_H_
#define LIB_ANALYSIS_CACHEANALYSISGRAPH_H_

#include "IndexComputationGraph.h"
#include "Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>

namespace mlir::foo {

struct AnalysisGraphNode;

/// An edge in a `CacheAnalysisGraph`. Edges encode relationships between nodes
/// based on known spatial and temporal distances between accesses, such as
/// those between loads from and stores to nearby addresses.
struct AnalysisGraphEdge {
  AnalysisGraphNode *Receiver;
  AffineExpr DistanceExpr;
};

/// A node in a `CacheAnalysisGraph`. Nodes represent operations to be subjected
/// to analysis, such as load and store operations on memrefs.
struct AnalysisGraphNode {
  /// The memory access operation that this node represents.
  Operation *LoadOrStoreOp;
  /// Whether or not `LoadOrStoreOp` is affine.
  bool IsAffineAccess;

  /// An expression representing the linearized access index of `LoadOrStoreOp`
  /// off of the base pointer.
  AffineExpr AccessExpr;

  /// The list of outgoing edges sent from this node.
  llvm::SmallVector<AnalysisGraphEdge> Edges;
};

class CacheAnalysisGraph {
public:
  CacheAnalysisGraph() = default;

  /// Inserts an `Operation` into the graph for analysis.
  template <typename LoadOrStoreOpTy>
  AnalysisGraphNode *getOrAddNode(LoadOrStoreOpTy LoadOrStoreOp,
                                  IndexComputationGraph *IndexComputations);

  ~CacheAnalysisGraph() {
    for (auto *Node : Nodes) {
      delete Node;
    }
  }

private:
  template <typename LoadOrStoreOpTy>
  AffineExpr findAccessExpr(LoadOrStoreOpTy LoadOrStoreOp,
                            IndexComputationGraph *IndexComputations);

  template <typename LoadOrStoreOpTy>
  void findAndAddEdges(LoadOrStoreOpTy LoadOrStoreOp,
                       IndexComputationGraph *IndexComputations);

  template <typename LoadOrStoreOpTy>
  void findRecursiveSelfRelation(LoadOrStoreOpTy LoadOrStoreOp,
                                 IndexComputationGraph *IndexComputations);

  // List of nodes belonging to this graph.
  llvm::SmallVector<AnalysisGraphNode *> Nodes;
  // One-to-one mapping between `Operation *`s and indices into `Nodes`.
  llvm::DenseMap<Operation *, std::size_t> NodeIndices;
  // Mapping from base pointers to nodes uniquely identifying isolated subgraphs
  // comprised of potentially related references off of the same base pointer.
  llvm::DenseMap<Value, AnalysisGraphNode *> ReferenceGroups;
};

template <typename LoadOrStoreOpTy>
AnalysisGraphNode *
CacheAnalysisGraph::getOrAddNode(LoadOrStoreOpTy LoadOrStoreOp,
                                 IndexComputationGraph *IndexComputations) {
  if (auto Itr = NodeIndices.find(LoadOrStoreOp); Itr != NodeIndices.end()) {
    return Nodes[Itr->second];
  }

  auto *Node = new AnalysisGraphNode{LoadOrStoreOp};
  Node->IsAffineAccess = is_affine_access<LoadOrStoreOpTy>::value;
  Node->AccessExpr =
      findAccessExpr<LoadOrStoreOpTy>(LoadOrStoreOp, IndexComputations);

  NodeIndices[LoadOrStoreOp] = Nodes.size();
  Nodes.push_back(Node);

  findAndAddEdges(LoadOrStoreOp, IndexComputations);

  return Node;
}

template <typename LoadOrStoreOpTy>
AffineExpr
CacheAnalysisGraph::findAccessExpr(LoadOrStoreOpTy LoadOrStoreOp,
                                   IndexComputationGraph *IndexComputations) {
  auto *Context = LoadOrStoreOp->getContext();

  const auto &Indices = LoadOrStoreOp.getIndices();
  llvm::SmallVector<AffineExpr> IndexExprs(Indices.size());
  std::transform(
      Indices.begin(), Indices.end(), IndexExprs.begin(), [&](auto Index) {
        return IndexComputations->getOrAddNode(Index)->PartialComputationExpr;
      });

  if constexpr (is_affine_access<LoadOrStoreOpTy>::value) {
    auto IndexMap = LoadOrStoreOp.getAffineMap();

    llvm::SmallVector<AffineExpr> DimReplacements;
    for (Value Index : Indices.take_front(IndexMap.getNumDims())) {
      DimReplacements.push_back(
          IndexComputations->getOrAddNode(Index)->PartialComputationExpr);
    }
    llvm::SmallVector<AffineExpr> SymReplacements;
    for (Value Index : Indices.take_back(IndexMap.getNumSymbols())) {
      SymReplacements.push_back(
          IndexComputations->getOrAddNode(Index)->PartialComputationExpr);
    }

    auto ContextualizedIndexMap =
        IndexMap.replaceDimsAndSymbols(DimReplacements, SymReplacements, 0,
                                       IndexComputations->getNumSymbols());

    IndexExprs.assign(ContextualizedIndexMap.getResults().begin(),
                      ContextualizedIndexMap.getResults().end());
  }

  AffineExpr AccessExpr = getAffineConstantExpr(0, Context);

  const auto &[Strides, Offset] =
      LoadOrStoreOp.getMemRefType().getStridesAndOffset();
  for (auto [IndexExpr, Stride] : llvm::zip_first(IndexExprs, Strides)) {
    AccessExpr = AccessExpr + Stride * IndexExpr;
  }

  return AccessExpr;
}

template <typename LoadOrStoreOpTy>
void CacheAnalysisGraph::findAndAddEdges(
    LoadOrStoreOpTy LoadOrStoreOp, IndexComputationGraph *IndexComputations) {
  auto *OwnNode = getOrAddNode(LoadOrStoreOp, IndexComputations);

  auto MemRef = LoadOrStoreOp.getMemRef();
  if (auto Itr = ReferenceGroups.find(MemRef); Itr != ReferenceGroups.end()) {
    auto *ReferenceNode = Itr->second;
    if (ReferenceNode != OwnNode) {
      ReferenceNode->Edges.push_back(AnalysisGraphEdge{
          OwnNode, OwnNode->AccessExpr - ReferenceNode->AccessExpr});
    }
  } else {
    ReferenceGroups[MemRef] = OwnNode;
  }

  findRecursiveSelfRelation(LoadOrStoreOp, IndexComputations);
}

template <typename LoadOrStoreOpTy>
void CacheAnalysisGraph::findRecursiveSelfRelation(
    LoadOrStoreOpTy LoadOrStoreOp, IndexComputationGraph *IndexComputations) {
  if (LoadOrStoreOp->template getParentOfType<scf::ForOp>() ||
      LoadOrStoreOp->template getParentOfType<scf::WhileOp>() ||
      LoadOrStoreOp->template getParentOfType<affine::AffineForOp>()) {
    // Access belongs to a loop nest, skip.
    return;
  }

  auto *Context = LoadOrStoreOp->getContext();
  auto Func = LoadOrStoreOp->template getParentOfType<func::FuncOp>();

  // TODO: Support indirect recursion.
  llvm::SmallVector<func::CallOp> RecursiveCalls;
  Func->walk([&](func::CallOp Call) {
    if (Call.getCallee() == Func.getSymName()) {
      RecursiveCalls.push_back(Call);
    }
  });
  if (RecursiveCalls.empty()) {
    // Non-recursive function.
    return;
  }

  llvm::SmallVector<AffineExpr> IndexExprs;
  for (auto Index : LoadOrStoreOp.getIndices()) {
    IndexExprs.push_back(
        IndexComputations->getOrAddNode(Index)->PartialComputationExpr);
  }
  for (auto RecursiveCall : RecursiveCalls) {
    llvm::DenseMap<AffineExpr, AffineExpr> Replacements;
    for (auto [Arg, ArgValue] :
         llvm::zip_equal(Func.getArguments(), RecursiveCall->getOperands())) {
      auto ArgSymbol = getAffineSymbolExpr(
          IndexComputations->getSymbolPosition(Arg), Context);
      Replacements[ArgSymbol] =
          IndexComputations->getOrAddNode(ArgValue)->PartialComputationExpr;
    }
    AffineMap IndexDeltaMap =
        AffineMap::get(0, IndexComputations->getNumSymbols(), Context);
    for (auto IndexExpr : IndexExprs) {
      auto NextIndexExpr = IndexExpr.replace(Replacements);
      auto IndexDelta = NextIndexExpr - IndexExpr;
      IndexDeltaMap =
          IndexDeltaMap.insertResult(IndexDelta, IndexDeltaMap.getNumResults());
    }
    llvm::outs() << IndexDeltaMap << '\n';
  }
}

} // namespace mlir::foo

#endif // LIB_ANALYSIS_CACHEANALYSISGRAPH_H_
