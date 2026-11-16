// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/PatchTable.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/SubcellAllowed.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
evolution::patches::PatchTable<Dim> patch_table(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Spectral::Quadrature i1_quadrature,
    const std::vector<size_t>& only_dg_block_ids,
    const bool subcell_enabled_at_external_boundary,
    const std::optional<size_t>& max_elements_per_patch,
    const bool dedicated_procs) {
  return evolution::patches::PatchTable<Dim>{
      blocks, initial_refinement_levels,
      [&](const Element<Dim>& element) -> std::optional<Mesh<Dim>> {
        // Elements of DG-only blocks can't use subcell, so don't build their
        // meshes
        if (alg::found(only_dg_block_ids, element.id().block_id())) {
          return std::nullopt;
        }
        Mesh<Dim> dg_mesh = domain::create_initial_mesh(
            initial_extents, element, Spectral::Basis::Legendre, i1_quadrature);
        if (not starts_on_subcell_grid(element, dg_mesh, only_dg_block_ids,
                                       subcell_enabled_at_external_boundary)) {
          return std::nullopt;
        }
        return dg_mesh;
      },
      max_elements_per_patch, dedicated_procs};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template evolution::patches::PatchTable<DIM(data)> patch_table(        \
      const std::vector<Block<DIM(data)>>& blocks,                       \
      const std::vector<std::array<size_t, DIM(data)>>&                  \
          initial_refinement_levels,                                     \
      const std::vector<std::array<size_t, DIM(data)>>& initial_extents, \
      Spectral::Quadrature i1_quadrature,                                \
      const std::vector<size_t>& only_dg_block_ids,                      \
      bool subcell_enabled_at_external_boundary,                         \
      const std::optional<size_t>& max_elements_per_patch,               \
      bool dedicated_procs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd_patch
