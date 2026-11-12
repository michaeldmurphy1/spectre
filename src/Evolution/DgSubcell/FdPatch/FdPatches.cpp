// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
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
#include "Domain/Structure/ZCurve.hpp"
#include "Evolution/DgSubcell/SubcellAllowed.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Literals.hpp"
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
    const bool subcell_enabled_at_external_boundary,
    const std::optional<size_t>& max_elements_per_patch) {
  if (max_elements_per_patch.has_value() and *max_elements_per_patch == 0) {
    ERROR("The maximum number of elements per FD patch must be positive.");
  }
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
    const size_t number_of_fd_elements = fd_element_ids.size();
    const size_t number_of_block_patches =
        max_elements_per_patch.has_value()
            ? (number_of_fd_elements + *max_elements_per_patch - 1) /
                  *max_elements_per_patch
            : 1;
    // Cut the Z-curve of the block's FD elements into consecutive pieces
    std::vector<size_t> z_curve_ranks(number_of_fd_elements);
    {
      std::vector<size_t> z_curve_order(number_of_fd_elements);
      std::iota(z_curve_order.begin(), z_curve_order.end(), 0_st);
      std::vector<size_t> z_curve_indices(number_of_fd_elements);
      for (size_t i = 0; i < number_of_fd_elements; ++i) {
        z_curve_indices[i] = domain::z_curve_index(fd_element_ids[i]);
      }
      std::sort(z_curve_order.begin(), z_curve_order.end(),
                [&z_curve_indices](const size_t lhs, const size_t rhs) {
                  return z_curve_indices[lhs] < z_curve_indices[rhs];
                });
      for (size_t rank = 0; rank < number_of_fd_elements; ++rank) {
        z_curve_ranks[z_curve_order[rank]] = rank;
      }
    }
    const size_t first_patch = block_ids_.size();
    block_ids_.resize(first_patch + number_of_block_patches, block.id());
    element_ids_.resize(first_patch + number_of_block_patches);
    dg_meshes_.resize(first_patch + number_of_block_patches, *patch_dg_mesh);
    // Piece `p` of the `k` pieces of the `n` elements holds the ranks
    // [floor(p * n / k), floor((p + 1) * n / k)), so rank `r` is in piece
    // ceil((r + 1) * k / n) - 1. Within each piece the elements keep the order
    // of `initial_element_ids()`.
    for (size_t i = 0; i < number_of_fd_elements; ++i) {
      const size_t patch = first_patch +
                           ((z_curve_ranks[i] + 1) * number_of_block_patches +
                            number_of_fd_elements - 1) /
                               number_of_fd_elements -
                           1;
      locations_.emplace(fd_element_ids[i],
                         Location{patch, element_ids_[patch].size()});
      element_ids_[patch].push_back(fd_element_ids[i]);
    }
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

std::vector<size_t> patch_procs(
    const size_t number_of_patches,
    const std::vector<std::vector<size_t>>& procs_on_nodes) {
  std::vector<const std::vector<size_t>*> nodes{};
  for (const auto& procs : procs_on_nodes) {
    if (not procs.empty()) {
      nodes.push_back(&procs);
    }
  }
  if (nodes.empty() and number_of_patches > 0) {
    ERROR("No processor may hold the " << number_of_patches << " FD patches.");
  }
  std::vector<size_t> result(number_of_patches);
  for (size_t patch = 0; patch < number_of_patches; ++patch) {
    const auto& procs = *nodes[patch % nodes.size()];
    result[patch] = procs[(patch / nodes.size()) % procs.size()];
  }
  return result;
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
