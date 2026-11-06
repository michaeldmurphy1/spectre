// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
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
 * \brief Replaces the packaged data on the faces of a patch that border DG
 * elements with the data the DG elements send, as
 * `evolution::dg::subcell::correct_package_data` does on the host.
 *
 * \details A DG neighbor sends its packaged data on the shared face (its
 * mortar data, already oriented for the receiver). `apply()` projects it onto
 * the subcell faces with `evolution::dg::subcell::fd::project` and overwrites
 * the packaged data of the side of the face that belongs to the neighbor: the
 * upper side on the last face of an element in an upper direction, and the
 * lower side on the first face in a lower direction.
 *
 * The packaged data of each side is a batch of `3 * number_of_elements` (in
 * 3d) face sets, ordered by direction and then element, as computed by
 * `grmhd::ValenciaDivClean::BoundaryCorrections::package_data_on_device`.
 *
 * The received data of all faces is in one buffer, laid out as described by
 * `Face`. The DG neighbor may send more components than the subcells use (e.g.
 * the GH and GRMHD corrections of a combined system, see
 * `correct_package_data`): `variables_to_offset_in_dg_grid` components are
 * skipped and the next `number_of_packaged_components` are used.
 *
 * Patch elements always use the subcells, so the internal mortar data that
 * `correct_package_data` also handles (after a rollback from DG) never exists.
 * Only DG neighbors in the same block, with the same extents, are supported.
 */
template <size_t Dim>
class PackagedDataFromDg {
 public:
  /// A face whose packaged data comes from a DG neighbor
  struct Face {
    size_t slot;
    Direction<Dim> direction;
    ElementId<Dim> neighbor;
    /// Where the mortar data received from the neighbor starts in the buffer
    /// passed to `apply()`. It takes `data_size` values, component by
    /// component.
    size_t data_offset;
    size_t data_size;
  };

  PackagedDataFromDg() = default;

  /// `elements` are the elements of the `patch` in slot order. The DG
  /// neighbors send `number_of_dg_components` components. The transfers are
  /// copied to the device on `exec`.
  PackagedDataFromDg(const Kokkos::DefaultExecutionSpace& exec,
                     const std::vector<Element<Dim>>& elements,
                     const FdPatches<Dim>& fd_patches, size_t patch,
                     size_t number_of_packaged_components,
                     size_t number_of_dg_components,
                     size_t variables_to_offset_in_dg_grid);

  /// The faces that border DG elements, ordered by slot and then direction
  const std::vector<Face>& faces() const { return faces_; }

  /// The size of the buffer that holds the mortar data of all `faces()`
  size_t received_data_size() const { return received_data_size_; }

  /// Overwrite the packaged data of the `faces()` in the storage of the
  /// `lower_packaged_data` and `upper_packaged_data` batches with the projected
  /// `received_data`. Enqueued on `exec`. Does nothing if there are no
  /// `faces()`, also for a default-constructed object.
  void apply(const Kokkos::DefaultExecutionSpace& exec,
             const Kokkos::View<double*>& lower_packaged_data,
             const Kokkos::View<double*>& upper_packaged_data,
             const Kokkos::View<const double*>& received_data) const;

 private:
  size_t number_of_packaged_components_ = 0;
  size_t packaged_data_size_ = 0;
  std::vector<Face> faces_{};
  size_t received_data_size_ = 0;
  ::fd::device::Transfers to_lower_{};
  ::fd::device::Transfers to_upper_{};
};
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
