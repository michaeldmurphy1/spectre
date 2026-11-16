// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/PatchTable.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/ZCurve.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace evolution::patches {
template <size_t Dim>
void PatchTable<Dim>::Location::pup(PUP::er& p) {
  p | patch;
  p | slot;
}

template <size_t Dim>
PatchTable<Dim>::PatchTable(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const PatchMesh& patch_mesh,
    const std::optional<size_t>& max_elements_per_patch,
    const bool dedicated_procs)
    : dedicated_procs_(dedicated_procs) {
  if (max_elements_per_patch.has_value() and *max_elements_per_patch == 0) {
    ERROR("The maximum number of elements per patch must be positive.");
  }
  // The elements in patches with each DG mesh (in order of first
  // appearance), in the order of the blocks and along the Z-curve of each
  // block, with their positions in the order of the blocks and of
  // `initial_element_ids()`
  std::vector<Mesh<Dim>> meshes{};
  std::vector<std::vector<std::pair<size_t, ElementId<Dim>>>> sequences{};
  size_t position = 0;
  for (const auto& block : blocks) {
    const auto block_element_ids =
        initial_element_ids(block.id(), initial_refinement_levels[block.id()]);
    // (Z-curve index, mesh, position, element ID) of the elements in patches
    std::vector<std::tuple<size_t, size_t, size_t, ElementId<Dim>>>
        block_patch_elements{};
    for (const auto& element_id : block_element_ids) {
      const Element<Dim> element = domain::create_initial_element(
          element_id, blocks, initial_refinement_levels);
      const std::optional<Mesh<Dim>> element_mesh = patch_mesh(element);
      if (not element_mesh.has_value()) {
        ++number_of_elements_outside_patches_;
        ++position;
        continue;
      }
      const Mesh<Dim>& dg_mesh = *element_mesh;
      const auto mesh_it = alg::find(meshes, dg_mesh);
      const auto mesh =
          static_cast<size_t>(std::distance(meshes.begin(), mesh_it));
      if (mesh_it == meshes.end()) {
        meshes.push_back(dg_mesh);
        sequences.emplace_back();
      }
      block_patch_elements.emplace_back(domain::z_curve_index(element_id), mesh,
                                        position, element_id);
      ++position;
    }
    std::sort(block_patch_elements.begin(), block_patch_elements.end(),
              [](const auto& lhs, const auto& rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
              });
    for (const auto& [z_curve_index, mesh, element_position, element_id] :
         block_patch_elements) {
      sequences[mesh].emplace_back(element_position, element_id);
    }
  }

  // Cut each sequence into consecutive pieces
  for (size_t mesh = 0; mesh < meshes.size(); ++mesh) {
    const auto& sequence = sequences[mesh];
    const size_t number_of_patch_elements = sequence.size();
    const size_t number_of_pieces =
        max_elements_per_patch.has_value()
            ? (number_of_patch_elements + *max_elements_per_patch - 1) /
                  *max_elements_per_patch
            : 1;
    // Piece `p` of the `k` pieces of the `n` elements holds the ranks
    // [floor(p * n / k), floor((p + 1) * n / k)).
    for (size_t piece = 0; piece < number_of_pieces; ++piece) {
      std::vector<std::pair<size_t, ElementId<Dim>>> piece_elements(
          std::next(sequence.begin(),
                    static_cast<std::ptrdiff_t>(
                        piece * number_of_patch_elements / number_of_pieces)),
          std::next(
              sequence.begin(),
              static_cast<std::ptrdiff_t>(
                  (piece + 1) * number_of_patch_elements / number_of_pieces)));
      // Within a piece the elements keep the order of the blocks and of
      // `initial_element_ids()`
      std::sort(piece_elements.begin(), piece_elements.end(),
                [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
                });
      const size_t patch = element_ids_.size();
      auto& patch_element_ids = element_ids_.emplace_back();
      dg_meshes_.push_back(meshes[mesh]);
      for (const auto& [element_position, element_id] : piece_elements) {
        locations_.emplace(element_id,
                           Location{patch, patch_element_ids.size()});
        patch_element_ids.push_back(element_id);
      }
    }
  }
}

template <size_t Dim>
const std::vector<ElementId<Dim>>& PatchTable<Dim>::element_ids(
    const size_t patch) const {
  ASSERT(patch < number_of_patches(),
         "Patch " << patch << " is out of range; there are "
                  << number_of_patches() << " patches.");
  return element_ids_[patch];
}

template <size_t Dim>
const Mesh<Dim>& PatchTable<Dim>::dg_mesh(const size_t patch) const {
  ASSERT(patch < number_of_patches(),
         "Patch " << patch << " is out of range; there are "
                  << number_of_patches() << " patches.");
  return dg_meshes_[patch];
}

template <size_t Dim>
auto PatchTable<Dim>::location(const ElementId<Dim>& element_id) const
    -> std::optional<Location> {
  const auto it = locations_.find(element_id);
  if (it == locations_.end()) {
    return std::nullopt;
  }
  return it->second;
}

template <size_t Dim>
void PatchTable<Dim>::pup(PUP::er& p) {
  p | element_ids_;
  p | dg_meshes_;
  p | locations_;
  p | number_of_elements_outside_patches_;
  p | dedicated_procs_;
}

template <size_t Dim>
bool operator==(const PatchTable<Dim>& lhs, const PatchTable<Dim>& rhs) {
  return lhs.element_ids_ == rhs.element_ids_ and
         lhs.dg_meshes_ == rhs.dg_meshes_ and
         lhs.locations_ == rhs.locations_ and
         lhs.number_of_elements_outside_patches_ ==
             rhs.number_of_elements_outside_patches_ and
         lhs.dedicated_procs_ == rhs.dedicated_procs_;
}

template <size_t Dim>
bool operator!=(const PatchTable<Dim>& lhs, const PatchTable<Dim>& rhs) {
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
    ERROR("No processor may hold the " << number_of_patches << " patches.");
  }
  std::vector<size_t> result(number_of_patches);
  for (size_t patch = 0; patch < number_of_patches; ++patch) {
    const auto& procs = *nodes[patch % nodes.size()];
    result[patch] = procs[(patch / nodes.size()) % procs.size()];
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                \
  template class PatchTable<DIM(data)>;                       \
  template bool operator==(const PatchTable<DIM(data)>& lhs,  \
                           const PatchTable<DIM(data)>& rhs); \
  template bool operator!=(const PatchTable<DIM(data)>& lhs,  \
                           const PatchTable<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::patches
