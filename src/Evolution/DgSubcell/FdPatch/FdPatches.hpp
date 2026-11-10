// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \cond
template <size_t VolumeDim>
class Block;
namespace PUP {
class er;
}  // namespace PUP
namespace Spectral {
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

/// Running FD elements as patches: one parallel component member stepping many
/// FD elements together on the device.
namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief Which FD elements belong to which patch.
 *
 * \details A patch holds the elements of one block that use the subcell (FD)
 * grid for the whole evolution, i.e. the elements for which
 * `evolution::dg::subcell::starts_on_subcell_grid()` is true (this assumes
 * `AlwaysUseSubcells`). Blocks without such elements have no patch. Patches are
 * numbered in order of increasing block ID, and the elements of a patch (its
 * slots) are in the order of `initial_element_ids()`.
 *
 * All elements of a patch must have the same DG mesh, so that the patch can
 * step them with batched kernels.
 *
 * The table is computed from the domain and options alone, so every process
 * computes the same table. It doesn't change during the evolution because FD
 * patches don't support AMR.
 */
template <size_t Dim>
class FdPatches {
 public:
  /// The patch of an FD element and its index within the patch.
  struct Location {
    size_t patch;
    size_t slot;

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);

    bool operator==(const Location& rhs) const = default;
  };

  FdPatches() = default;

  FdPatches(
      const std::vector<Block<Dim>>& blocks,
      const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      Spectral::Quadrature i1_quadrature,
      const std::vector<size_t>& only_dg_block_ids,
      bool subcell_enabled_at_external_boundary);

  size_t number_of_patches() const { return block_ids_.size(); }

  /// The block whose FD elements make up the `patch`.
  size_t block_id(size_t patch) const;

  /// The FD elements of the `patch`, in slot order.
  const std::vector<ElementId<Dim>>& element_ids(size_t patch) const;

  /// The DG mesh shared by all elements of the `patch`.
  const Mesh<Dim>& dg_mesh(size_t patch) const;

  /// The patch and slot of `element_id`, or `std::nullopt` if the element is
  /// not in any patch (i.e. it is a DG element).
  std::optional<Location> location(const ElementId<Dim>& element_id) const;

  /// Total number of elements in all patches.
  size_t number_of_elements() const { return locations_.size(); }

  /// Total number of elements that aren't in a patch, i.e. the DG elements.
  size_t number_of_dg_elements() const { return number_of_dg_elements_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalDim>
  friend bool operator==(const FdPatches<LocalDim>& lhs,
                         const FdPatches<LocalDim>& rhs);

  std::vector<size_t> block_ids_{};
  std::vector<std::vector<ElementId<Dim>>> element_ids_{};
  std::vector<Mesh<Dim>> dg_meshes_{};
  std::unordered_map<ElementId<Dim>, Location> locations_{};
  size_t number_of_dg_elements_ = 0;
};

template <size_t Dim>
bool operator==(const FdPatches<Dim>& lhs, const FdPatches<Dim>& rhs);

template <size_t Dim>
bool operator!=(const FdPatches<Dim>& lhs, const FdPatches<Dim>& rhs);
}  // namespace evolution::dg::subcell::fd_patch
