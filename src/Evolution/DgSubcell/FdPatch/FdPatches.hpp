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
 * \details A patch holds elements of one block that use the subcell (FD) grid
 * for the whole evolution, i.e. elements for which
 * `evolution::dg::subcell::starts_on_subcell_grid()` is true (this assumes
 * `AlwaysUseSubcells`). Blocks without such elements have no patch. By default
 * a block's FD elements form one patch. With `max_elements_per_patch`, they
 * are split into the fewest patches that hold at most that many elements,
 * with sizes that differ by at most one: the elements are ordered along the
 * Z-curve of the block (see `domain::z_curve_index()`) and cut into
 * consecutive pieces, so that each patch is spatially compact and has few
 * faces with other patches. This is how a large block is spread over several
 * GPUs.
 *
 * Patches are numbered in order of increasing block ID, and then along the
 * Z-curve. The elements of a patch (its slots) are in the order of
 * `initial_element_ids()`.
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
      bool subcell_enabled_at_external_boundary,
      const std::optional<size_t>& max_elements_per_patch = std::nullopt);

  size_t number_of_patches() const { return block_ids_.size(); }

  /// The block of the FD elements of the `patch`.
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

/*!
 * \brief The processor (PE) of each of the `number_of_patches` patches.
 *
 * \details `procs_on_nodes[n]` are the processors of node (process) `n` that
 * may hold patches. The patches are distributed round-robin over the nodes
 * that have such processors, and the patches of a node round-robin over its
 * processors. With one process per GPU this spreads the patches evenly over
 * the GPUs, and patches on the same GPU get different processors, i.e.
 * different device execution space instances (see
 * `Parallel::device_execution_space()`) as long as there are enough.
 */
std::vector<size_t> patch_procs(
    size_t number_of_patches,
    const std::vector<std::vector<size_t>>& procs_on_nodes);

template <size_t Dim>
bool operator!=(const FdPatches<Dim>& lhs, const FdPatches<Dim>& rhs);
}  // namespace evolution::dg::subcell::fd_patch
