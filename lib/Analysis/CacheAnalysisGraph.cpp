#include "CacheAnalysisGraph.h"

#include "lib/Analysis/IndexComputationGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>

namespace mlir::foo {

void CacheAnalysisGraph::printGraph() const {
  // Begin the dot graph.
  llvm::outs() << "digraph CacheAnalysisGraph {" << '\n';

  // Output each node with a unique name.
  for (std::size_t i = 0; i < Nodes.size(); ++i) {
    const auto *Node = Nodes[i];
    std::string Label = Node->toString();
    llvm::outs() << "  node" << i << " [label=\"" << Label << "\"];" << '\n';
  }

  // Output the edges between nodes.
  for (const auto &[i, Node] : llvm::enumerate(Nodes)) {
    // Get successors from the node.
    for (const auto &Edge : Node->Edges) {
      const auto *Receiver = Edge.Receiver;
      // Find the index of the receiver.
      const auto *Itr = std::find(Nodes.begin(), Nodes.end(), Receiver);
      if (Itr != Nodes.end()) {
        std::size_t j = std::distance(Nodes.begin(), Itr);
        llvm::outs() << "  node" << i << " -> node" << j << " [label=\"";
        Edge.DistanceExpr.print(llvm::outs());
        llvm::outs() << "\"];" << '\n';
      }
    }
  }
  llvm::outs() << "}" << '\n';
}

} // namespace mlir::foo
