// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Info.hpp"

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
 * \details A patch holds elements that use the subcell (FD) grid for the
 * whole evolution, i.e. elements for which
 * `evolution::dg::subcell::starts_on_subcell_grid()` is true (this assumes
 * `AlwaysUseSubcells`). The elements of a patch may be in several blocks.
 *
 * All elements of a patch have the same DG mesh, so that the patch can step
 * them with batched kernels. The FD elements with the same DG mesh are ordered
 * by block and along the Z-curve of each block (see `domain::z_curve_index()`).
 * By default they form one patch. With `max_elements_per_patch`, they are
 * split into the fewest patches that hold at most that many elements, with
 * sizes that differ by at most one, by cutting this order into consecutive
 * pieces, so that each patch is spatially compact and has few faces with other
 * patches. This is how the FD elements are spread over several GPUs.
 *
 * Patches are numbered by DG mesh (in order of the first element with the
 * mesh), and then in this order. The elements of a patch (its slots) are in
 * the order of the blocks and of `initial_element_ids()`.
 *
 * The table is computed from the domain and options alone, so every process
 * computes the same table. It doesn't change during the evolution because FD
 * patches don't support AMR.
 *
 * The table also decides where the patches run (`procs()`), and whether the
 * DG elements avoid those processors (`dedicated_procs()`).
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
      const std::optional<size_t>& max_elements_per_patch = std::nullopt,
      bool dedicated_procs = false);

  size_t number_of_patches() const { return element_ids_.size(); }

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

  /// Whether the DG elements avoid the processors of the patches, so that a
  /// patch's actions never wait behind those of DG elements on its processor
  bool dedicated_procs() const { return dedicated_procs_; }

  /// The processor (PE) of each patch, avoiding the `procs_to_ignore` (see
  /// `patch_procs()`)
  template <typename DistribObject>
  std::vector<size_t> procs(
      const DistribObject& distributed_object,
      const std::unordered_set<size_t>& procs_to_ignore) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalDim>
  friend bool operator==(const FdPatches<LocalDim>& lhs,
                         const FdPatches<LocalDim>& rhs);

  std::vector<std::vector<ElementId<Dim>>> element_ids_{};
  std::vector<Mesh<Dim>> dg_meshes_{};
  std::unordered_map<ElementId<Dim>, Location> locations_{};
  size_t number_of_dg_elements_ = 0;
  bool dedicated_procs_ = false;
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
template <typename DistribObject>
std::vector<size_t> FdPatches<Dim>::procs(
    const DistribObject& distributed_object,
    const std::unordered_set<size_t>& procs_to_ignore) const {
  const auto number_of_nodes =
      Parallel::number_of_nodes<size_t>(distributed_object);
  std::vector<std::vector<size_t>> procs_on_nodes(number_of_nodes);
  for (size_t node = 0; node < number_of_nodes; ++node) {
    const auto first_proc =
        Parallel::first_proc_on_node<size_t>(node, distributed_object);
    const auto number_of_procs =
        Parallel::procs_on_node<size_t>(node, distributed_object);
    for (size_t proc = first_proc; proc < first_proc + number_of_procs;
         ++proc) {
      if (not procs_to_ignore.contains(proc)) {
        procs_on_nodes[node].push_back(proc);
      }
    }
  }
  return patch_procs(number_of_patches(), procs_on_nodes);
}

template <size_t Dim>
bool operator!=(const FdPatches<Dim>& lhs, const FdPatches<Dim>& rhs);
}  // namespace evolution::dg::subcell::fd_patch
