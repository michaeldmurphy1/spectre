// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/PatchTable.hpp"

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
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace evolution::patches {
template <size_t Dim>
size_t PatchTable<Dim>::block_id(const size_t patch) const {
  ASSERT(patch < number_of_patches(),
         "Patch " << patch << " is out of range; there are "
                  << number_of_patches() << " patches.");
  return block_ids_[patch];
}

template <size_t Dim>
void PatchTable<Dim>::Location::pup(PUP::er& p) {
  p | patch;
  p | slot;
}

template <size_t Dim>
PatchTable<Dim>::PatchTable(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const PatchMesh& patch_mesh) {
  for (const auto& block : blocks) {
    std::vector<ElementId<Dim>> patch_element_ids{};
    std::optional<Mesh<Dim>> patch_dg_mesh{};
    for (const auto& element_id : initial_element_ids(
             block.id(), initial_refinement_levels[block.id()])) {
      const Element<Dim> element = domain::create_initial_element(
          element_id, blocks, initial_refinement_levels);
      const std::optional<Mesh<Dim>> dg_mesh = patch_mesh(element);
      if (not dg_mesh.has_value()) {
        ++number_of_elements_outside_patches_;
        continue;
      }
      if (not patch_dg_mesh.has_value()) {
        patch_dg_mesh = *dg_mesh;
      } else if (*dg_mesh != *patch_dg_mesh) {
        ERROR("All elements of a patch must have the same DG mesh, but element "
              << element_id << " has " << *dg_mesh << " while element "
              << patch_element_ids.front() << " has " << *patch_dg_mesh);
      }
      patch_element_ids.push_back(element_id);
    }
    if (patch_element_ids.empty()) {
      continue;
    }
    const size_t patch = block_ids_.size();
    for (size_t slot = 0; slot < patch_element_ids.size(); ++slot) {
      locations_.emplace(patch_element_ids[slot], Location{patch, slot});
    }
    block_ids_.push_back(block.id());
    element_ids_.push_back(std::move(patch_element_ids));
    dg_meshes_.push_back(std::move(*patch_dg_mesh));
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
  p | block_ids_;
  p | element_ids_;
  p | dg_meshes_;
  p | locations_;
  p | number_of_elements_outside_patches_;
}

template <size_t Dim>
bool operator==(const PatchTable<Dim>& lhs, const PatchTable<Dim>& rhs) {
  return lhs.block_ids_ == rhs.block_ids_ and
         lhs.element_ids_ == rhs.element_ids_ and
         lhs.dg_meshes_ == rhs.dg_meshes_ and
         lhs.locations_ == rhs.locations_ and
         lhs.number_of_elements_outside_patches_ ==
             rhs.number_of_elements_outside_patches_;
}

template <size_t Dim>
bool operator!=(const PatchTable<Dim>& lhs, const PatchTable<Dim>& rhs) {
  return not(lhs == rhs);
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
