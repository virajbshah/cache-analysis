#ifndef LIB_ANALYSIS_UTILS_H_
#define LIB_ANALYSIS_UTILS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::foo {

template <typename LoadOrStoreOpTy>
using is_affine_access = llvm::is_one_of<LoadOrStoreOpTy, affine::AffineLoadOp,
                                         affine::AffineStoreOp>;

} // namespace mlir::foo

#endif // LIB_ANALYSIS_UTILS_H_