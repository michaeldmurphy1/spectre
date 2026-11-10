// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \cond
template <size_t VolumeDim>
class Block;
template <size_t VolumeDim>
class Element;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Running elements as patches: one parallel component member stepping many
/// elements of the same mesh together on the device.
namespace evolution::patches {
/*!
 * \brief Which elements belong to which patch.
 *
 * \details A method decides which elements go into patches (e.g. FD
 * elements, see `evolution::dg::subcell::fd_patch::patch_table()`); the
 * other elements stay in the `DgElementArray`. A patch holds the elements of
 * one block that are in patches, and blocks without such elements have no
 * patch. Patches are numbered in order of increasing block ID, and the
 * elements of a patch (its slots) are in the order of `initial_element_ids()`.
 *
 * All elements of a patch must have the same DG mesh, so that the patch can
 * step them with batched kernels.
 *
 * The table is computed from the domain and options alone, so every process
 * computes the same table. It doesn't change during the evolution because
 * patches don't support AMR.
 */
template <size_t Dim>
class PatchTable {
 public:
  /// The patch of an element and its index within the patch.
  struct Location {
    size_t patch;
    size_t slot;

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);

    bool operator==(const Location& rhs) const = default;
  };

  PatchTable() = default;

  /// The DG mesh of an element if it goes into a patch, or `std::nullopt` if
  /// it stays in the `DgElementArray`
  using PatchMesh =
      std::function<std::optional<Mesh<Dim>>(const Element<Dim>& element)>;

  /// The patches of the elements of the domain with `blocks` and initial
  /// refinement levels for which `patch_mesh` gives a mesh
  PatchTable(
      const std::vector<Block<Dim>>& blocks,
      const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
      const PatchMesh& patch_mesh);

  size_t number_of_patches() const { return block_ids_.size(); }

  /// The block whose elements make up the `patch`.
  size_t block_id(size_t patch) const;

  /// The elements of the `patch`, in slot order.
  const std::vector<ElementId<Dim>>& element_ids(size_t patch) const;

  /// The DG mesh shared by all elements of the `patch`.
  const Mesh<Dim>& dg_mesh(size_t patch) const;

  /// The patch and slot of `element_id`, or `std::nullopt` if the element is
  /// not in any patch (i.e. it is in the `DgElementArray`).
  std::optional<Location> location(const ElementId<Dim>& element_id) const;

  /// Total number of elements in all patches.
  size_t number_of_elements() const { return locations_.size(); }

  /// Total number of elements that aren't in a patch, i.e. that are in the
  /// `DgElementArray`
  size_t number_of_elements_outside_patches() const {
    return number_of_elements_outside_patches_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalDim>
  friend bool operator==(const PatchTable<LocalDim>& lhs,
                         const PatchTable<LocalDim>& rhs);

  std::vector<size_t> block_ids_{};
  std::vector<std::vector<ElementId<Dim>>> element_ids_{};
  std::vector<Mesh<Dim>> dg_meshes_{};
  std::unordered_map<ElementId<Dim>, Location> locations_{};
  size_t number_of_elements_outside_patches_ = 0;
};

template <size_t Dim>
bool operator==(const PatchTable<Dim>& lhs, const PatchTable<Dim>& rhs);

template <size_t Dim>
bool operator!=(const PatchTable<Dim>& lhs, const PatchTable<Dim>& rhs);
}  // namespace evolution::patches
