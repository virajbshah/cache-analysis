#ifndef LIB_ANALYSIS_INDEXCOMPUTATIONGRAPH_H_
#define LIB_ANALYSIS_INDEXCOMPUTATIONGRAPH_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>

namespace mlir::foo {

/// Enumerates all node types that `IndexComputationGraph`s can use to build
/// partial index computations. This includes common operations that can be
/// represented within `AffineExpr`s, along with types representing analysis
/// boundaries.
enum class IndexNodeKind : std::uint8_t {
  Constant,
  IndexCast,
  Add,
  UnknownOp,
  BlockArgument,
};

/// A node in an `IndexComputationGraph`. Represents a `Value` within the
/// analysis scope and an expression representing the partial computation of
/// access indices depending on it.
struct IndexGraphNode {
  /// The SSA value that this node represents.
  Value InnerValue;
  /// A tag for what kind of operation produced this node.
  IndexNodeKind Kind;

  /// The affine expression computed for this node.
  AffineExpr PartialComputationExpr;
  /// The source nodes of incoming edges into this node.
  /// TODO: Check if we really need to store this information.
  llvm::SmallVector<IndexGraphNode *> Parents;
};

/// Represents partial computations of indices into memory. Each index is
/// represented as an `AffineExpr` comprising of `Value`s and
/// affine-representable operations between them.
class IndexComputationGraph {
public:
  IndexComputationGraph() = default;

  /// Get a node for a given `Value`, adding one if not present. Traverses up
  /// the IR to build and cache computation expressions for newly added nodes.
  IndexGraphNode *getOrAddNode(Value Val);

  /// Get the position of the symbol corresponding to a `Value`.
  std::size_t getSymbolPosition(Value Val) const;

  /// Get the number of symbols currently tracked.
  std::size_t getNumSymbols() const { return Nodes.size(); }

  void printIndexComputations() const;

  /// Prints out the mapping between SSA values and symbols.
  void printSymbolLegend() const;

  ~IndexComputationGraph() {
    for (auto &Node : Nodes) {
      delete Node;
    }
  }

private:
  // List of nodes belonging to this graph.
  llvm::SmallVector<IndexGraphNode *> Nodes;
  // One-to-one mapping between `Value`s and indices into `Nodes`.
  llvm::DenseMap<Value, std::size_t> NodeIndices;
};

} // namespace mlir::foo

#endif // LIB_ANALYSIS_INDEXCOMPUTATIONGRAPH_H_
