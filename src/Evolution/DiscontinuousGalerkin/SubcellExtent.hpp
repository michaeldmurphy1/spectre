// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace evolution::dg {
/*!
 * \brief The number of finite-difference subcell grid points in a dimension
 * that has `dg_extent` DG grid points.
 *
 * This is the single definition of the DG-to-subcell refinement ratio;
 * `evolution::dg::subcell::fd::mesh` builds the subcell `Mesh` from it, and
 * `evolution::dg::compute_weighting_extents_override` uses it to weight
 * subcell-capable elements for the element distribution. It lives here rather
 * than in the `DgSubcell` library so that code which must not depend on
 * `DgSubcell` can use it.
 */
constexpr size_t subcell_extent_from_dg_extent(const size_t dg_extent) {
  return 2 * dg_extent - 1;
}
}  // namespace evolution::dg
