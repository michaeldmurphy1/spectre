// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Block.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
/*!
 * \brief Returns whether the `Block`'s topology is compatible with subcell.
 *
 * Returns `true` only if every dimension of the `Block` has a topology for
 * which `domain::topology_supports_subcell` is `true`, i.e. hypercubes and
 * the cartoon topologies. Anything else, such as a spherical shell, returns
 * `false`.
 *
 * \note This check depends only on `domain::Topology`, so it lives here
 * rather than in the `DgSubcell` library. That lets code which must not
 * depend on `DgSubcell` (such as the element distribution) use it, and avoids
 * forcing executables that do not use subcell to link against `DgSubcell`.
 */
template <size_t Dim>
bool block_supports_subcell(const Block<Dim>& block) {
  for (size_t d = 0; d < Dim; ++d) {
    if (not domain::topology_supports_subcell(gsl::at(block.topologies(), d))) {
      return false;
    }
  }
  return true;
}
}  // namespace evolution::dg
