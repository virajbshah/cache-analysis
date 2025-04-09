#include "IndexComputationGraph.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>

namespace mlir::foo {

IndexGraphNode *IndexComputationGraph::getOrAddNode(Value Val) {
  if (auto Itr = NodeIndices.find(Val); Itr != NodeIndices.end()) {
    return Nodes[Itr->second];
  }

  auto *Context = Val.getContext();

  auto *Node = new IndexGraphNode{Val};
  if (auto *DefiningOp = Val.getDefiningOp()) {
    if (auto ConstantOp = dyn_cast<arith::ConstantOp>(DefiningOp)) {
      auto Constant = ConstantOp.getValue();
      assert(Constant.getType().isIntOrIndex() &&
             "Floats should not be part of index computations.");
      Node->Kind = IndexNodeKind::Constant;
      Node->PartialComputationExpr = getAffineConstantExpr(
          dyn_cast<IntegerAttr>(Constant).getInt(), Context);
    } else if (auto IndexCastOp = dyn_cast<arith::IndexCastOp>(DefiningOp)) {
      Node->Kind = IndexNodeKind::IndexCast;
      Node->PartialComputationExpr =
          getOrAddNode(IndexCastOp.getIn())->PartialComputationExpr;
    } else if (auto AddOp = dyn_cast<arith::AddIOp>(DefiningOp)) {
      Node->Kind = IndexNodeKind::Add;
      Node->PartialComputationExpr =
          getOrAddNode(AddOp.getLhs())->PartialComputationExpr +
          getOrAddNode(AddOp.getRhs())->PartialComputationExpr;
    } else {
      Node->Kind = IndexNodeKind::UnknownOp;
      Node->PartialComputationExpr = getAffineSymbolExpr(Nodes.size(), Context);
    }
  } else {
    Node->Kind = IndexNodeKind::BlockArgument;
    Node->PartialComputationExpr = getAffineSymbolExpr(Nodes.size(), Context);
  }

  NodeIndices[Val] = Nodes.size();
  Nodes.push_back(Node);

  return Node;
}

std::size_t IndexComputationGraph::getSymbolPosition(Value Val) const {
  if (auto Itr = NodeIndices.find(Val); Itr != NodeIndices.end()) {
    return Itr->second;
  }

  return -1;
}

void IndexComputationGraph::printIndexComputations() const {
  for (const auto *Node : Nodes) {
    llvm::outs() << Node->InnerValue << " => " << Node->PartialComputationExpr
                 << '\n';
  }
}

void IndexComputationGraph::printSymbolLegend() const {
  for (const auto *Node : Nodes) {
    llvm::outs() << 's' << getSymbolPosition(Node->InnerValue) << " => "
                 << Node->InnerValue << ", " << Node->InnerValue.getLoc()
                 << '\n';
  }
}

} // namespace mlir::foo
