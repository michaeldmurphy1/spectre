// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t VolumeDim>
class Element;
namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
class FdPatches;
}  // namespace evolution::dg::subcell::fd_patch
/// \endcond

namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief Fills the ghost zones of the padded volume data of a patch.
 *
 * \details The ghost variables of all elements of a patch are computed into an
 * unpadded batch (a `BatchedVariables`, so element `e` starts at
 * `e * number_of_variables * number_of_cells` and variable `v` of the element
 * at `v * number_of_cells` past that) and are then copied into a padded
 * `fd::device::VolumeView` for the reconstruction. `fill_from_patch()` does, in
 * one kernel:
 * - copy the cells of each element into the interior of its padded element,
 * - for each face between two elements of the patch, including periodic faces,
 *   copy the `ghost_width` cells of the neighbor next to the face into the
 *   ghost zone.
 *
 * All elements of a patch are in one block, so all neighbors in the patch are
 * aligned and conforming. The ghost zones of the faces whose neighbor isn't in
 * the patch, i.e. a DG element or an element of another patch, are listed in
 * `external_faces()` and are filled from the data received from the neighbor.
 * External boundaries aren't supported (only periodic boundaries, which are
 * faces between elements).
 *
 * \note The interior copy could be avoided by computing the ghost variables
 * directly into the interior of the padded view, which costs a strided batch
 * of the ghost variables instead of the `BatchedVariables`.
 *
 * \note The ghost zones from inside the patch don't wait for any messages.
 * Another possible optimization is to overlap communication with computation:
 * the reconstruction and the time derivative of all cells that don't need
 * ghost data from external faces could be computed while the messages from DG
 * elements and other patches are in flight, and only the cells next to
 * external faces afterwards. This requires splitting the kernels by sets of
 * cells or faces.
 */
template <size_t Dim>
class GhostZones {
 public:
  /// A face whose ghost zone is filled with data from outside the patch.
  struct ExternalFace {
    size_t slot;
    Direction<Dim> direction;
    ElementId<Dim> neighbor;
  };

  GhostZones() = default;

  /// `elements` are the elements of the `patch` in slot order. The transfers
  /// are copied to the device on `exec`.
  GhostZones(const Kokkos::DefaultExecutionSpace& exec,
             const std::vector<Element<Dim>>& elements,
             const FdPatches<Dim>& fd_patches, size_t patch, size_t ghost_width,
             size_t number_of_variables);

  /// The number of cells of each element in each dimension
  const Index<Dim>& subcell_extents() const { return subcell_extents_; }

  size_t ghost_width() const { return ghost_width_; }

  size_t number_of_variables() const { return number_of_variables_; }

  /// Copy the unpadded `ghost_variables` of all elements of the patch into
  /// `padded_vars`, and fill the ghost zones of the faces between elements of
  /// the patch. Enqueued on `exec`.
  ///
  /// `padded_vars` must have been allocated with `fd::device::make_volume_view`
  /// for the elements of the patch, `number_of_variables()` variables,
  /// `subcell_extents()` and `ghost_width()`.
  void fill_from_patch(
      const Kokkos::DefaultExecutionSpace& exec,
      const ::fd::device::VolumeView& padded_vars,
      const Kokkos::View<const double*>& ghost_variables) const;

  /// The faces whose neighbor isn't in the patch, ordered by slot and then
  /// direction.
  const std::vector<ExternalFace>& external_faces() const {
    return external_faces_;
  }

 private:
  size_t number_of_elements_ = 0;
  Index<Dim> subcell_extents_{};
  size_t ghost_width_ = 0;
  size_t number_of_variables_ = 0;
  ::fd::device::Transfers from_patch_{};
  std::vector<ExternalFace> external_faces_{};
};
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
