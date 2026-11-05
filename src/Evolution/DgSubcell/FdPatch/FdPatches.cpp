// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/DgSubcell/SubcellAllowed.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
void FdPatches<Dim>::Location::pup(PUP::er& p) {
  p | patch;
  p | slot;
}

template <size_t Dim>
FdPatches<Dim>::FdPatches(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Spectral::Quadrature i1_quadrature,
    const std::vector<size_t>& only_dg_block_ids,
    const bool subcell_enabled_at_external_boundary) {
  for (const auto& block : blocks) {
    // Elements of a DG-only block can't use subcell, so skip building them.
    if (alg::found(only_dg_block_ids, block.id())) {
      number_of_dg_elements_ +=
          initial_element_ids(block.id(), initial_refinement_levels[block.id()])
              .size();
      continue;
    }
    std::vector<ElementId<Dim>> fd_element_ids{};
    std::optional<Mesh<Dim>> patch_dg_mesh{};
    for (const auto& element_id : initial_element_ids(
             block.id(), initial_refinement_levels[block.id()])) {
      const Element<Dim> element = domain::create_initial_element(
          element_id, blocks, initial_refinement_levels);
      const Mesh<Dim> dg_mesh = domain::create_initial_mesh(
          initial_extents, element, Spectral::Basis::Legendre, i1_quadrature);
      if (not starts_on_subcell_grid(element, dg_mesh, only_dg_block_ids,
                                     subcell_enabled_at_external_boundary)) {
        ++number_of_dg_elements_;
        continue;
      }
      if (not patch_dg_mesh.has_value()) {
        patch_dg_mesh = dg_mesh;
      } else if (dg_mesh != *patch_dg_mesh) {
        ERROR(
            "All FD elements of a patch must have the same DG mesh, but "
            "element "
            << element_id << " has " << dg_mesh << " while element "
            << fd_element_ids.front() << " has " << *patch_dg_mesh);
      }
      fd_element_ids.push_back(element_id);
    }
    if (fd_element_ids.empty()) {
      continue;
    }
    const size_t patch = block_ids_.size();
    for (size_t slot = 0; slot < fd_element_ids.size(); ++slot) {
      locations_.emplace(fd_element_ids[slot], Location{patch, slot});
    }
    block_ids_.push_back(block.id());
    element_ids_.push_back(std::move(fd_element_ids));
    dg_meshes_.push_back(std::move(*patch_dg_mesh));
  }
}

template <size_t Dim>
size_t FdPatches<Dim>::block_id(const size_t patch) const {
  ASSERT(patch < number_of_patches(),
         "Patch " << patch << " is out of range; there are "
                  << number_of_patches() << " patches.");
  return block_ids_[patch];
}

template <size_t Dim>
const std::vector<ElementId<Dim>>& FdPatches<Dim>::element_ids(
    const size_t patch) const {
  ASSERT(patch < number_of_patches(),
         "Patch " << patch << " is out of range; there are "
                  << number_of_patches() << " patches.");
  return element_ids_[patch];
}

template <size_t Dim>
const Mesh<Dim>& FdPatches<Dim>::dg_mesh(const size_t patch) const {
  ASSERT(patch < number_of_patches(),
         "Patch " << patch << " is out of range; there are "
                  << number_of_patches() << " patches.");
  return dg_meshes_[patch];
}

template <size_t Dim>
auto FdPatches<Dim>::location(const ElementId<Dim>& element_id) const
    -> std::optional<Location> {
  const auto it = locations_.find(element_id);
  if (it == locations_.end()) {
    return std::nullopt;
  }
  return it->second;
}

template <size_t Dim>
void FdPatches<Dim>::pup(PUP::er& p) {
  p | block_ids_;
  p | element_ids_;
  p | dg_meshes_;
  p | locations_;
  p | number_of_dg_elements_;
}

template <size_t Dim>
bool operator==(const FdPatches<Dim>& lhs, const FdPatches<Dim>& rhs) {
  return lhs.block_ids_ == rhs.block_ids_ and
         lhs.element_ids_ == rhs.element_ids_ and
         lhs.dg_meshes_ == rhs.dg_meshes_ and
         lhs.locations_ == rhs.locations_ and
         lhs.number_of_dg_elements_ == rhs.number_of_dg_elements_;
}

template <size_t Dim>
bool operator!=(const FdPatches<Dim>& lhs, const FdPatches<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                               \
  template class FdPatches<DIM(data)>;                       \
  template bool operator==(const FdPatches<DIM(data)>& lhs,  \
                           const FdPatches<DIM(data)>& rhs); \
  template bool operator!=(const FdPatches<DIM(data)>& lhs,  \
                           const FdPatches<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd_patch
