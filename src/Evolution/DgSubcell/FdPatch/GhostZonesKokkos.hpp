// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class Element;
template <size_t Dim>
class Mesh;
template <size_t VolumeDim>
class OrientationMap;
namespace evolution::patches {
template <size_t Dim>
class PatchTable;
}  // namespace evolution::patches
/// \endcond

namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief Where `GhostZones` reads a range of the ghost variables of the
 * elements from.
 *
 * \details The ghost variables are split into consecutive parts, each of which
 * is read from its own unpadded batch (with the layout of a
 * `BatchedVariables`): the `number_of_variables` variables of a part are the
 * variables `first_variable, ...` of a batch with `variables_per_element`
 * variables, so variable `v` of the part of element `e` starts at
 * `(e * variables_per_element + first_variable + v) * number_of_cells`. E.g.
 * ghost variables that are also evolved variables can be read from the
 * evolved variables without copying them into a batch of the ghost variables
 * first. In the padded view and in the ghost data sent to and received from
 * the neighbors, the variables of the parts are in order.
 */
struct GhostVariablesPart {
  size_t number_of_variables;
  size_t variables_per_element;
  size_t first_variable = 0;
};

/*!
 * \brief Fills the ghost zones of the padded volume data of a patch.
 *
 * \details The ghost variables of all elements of a patch are computed into
 * unpadded batches (see `GhostVariablesPart`) and are then copied into a padded
 * `fd::device::VolumeView` for the reconstruction. `fill_from_patch()` does, in
 * one kernel for each part:
 * - copy the cells of each element into the interior of its padded element,
 * - for each face between two elements of the patch, including periodic faces,
 *   copy the `ghost_width` cells of the neighbor next to the face into the
 *   ghost zone.
 *
 * All elements of a patch have the same mesh, and the cells of neighbors in
 * other blocks line up with the ghost cells (see
 * `check_cells_coincide_across_blocks`), so the ghost zones from neighbors in
 * the patch are copies, reoriented for neighbors in blocks with another
 * orientation (see `cells_for_neighbor_ghost_zone`). The ghost zones of the
 * faces whose neighbor isn't in the patch, i.e. a DG element or an element of
 * another patch, are listed in `external_faces()`. They are filled by
 * `fill_from_external()` from the data received from the neighbors, with the
 * same treatment as on the host (see
 * `evolution::dg::subcell::insert_or_update_neighbor_volume_data`):
 * - An FD neighbor sends the ghost data already sliced, reoriented and
 *   interpolated for this element, so it is copied.
 * - A DG neighbor sends its whole volume data, which is projected onto the
 *   ghost zone with `evolution::dg::subcell::fd::projection_matrix` in each
 *   dimension. The projection is only supported for DG neighbors in the same
 *   block, and not for `Spectral::Basis::ZernikeB1`.
 *
 * `slice_for_external()` computes the ghost data to send to the neighbors of
 * the external faces, as `evolution::dg::subcell::slice_data` does on the host:
 * the cells next to the face are copied (and reoriented for the neighbor's
 * frame), or, where there is an interpolant in
 * `fd_to_neighbor_fd_interpolants` (see
 * `evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd`),
 * interpolated to the ghost zone of the neighbor. Directions whose interpolant
 * is `std::nullopt` are left unfilled, as on the host.
 *
 * External boundaries aren't supported (only periodic boundaries, which are
 * faces between elements).
 *
 * \note The interior copy could be avoided by computing the ghost variables
 * directly into the interior of the padded view, which costs a strided batch
 * of the ghost variables instead of the `BatchedVariables`. Reading parts of
 * the ghost variables from where they already are (see `GhostVariablesPart`)
 * avoids an extra copy of them.
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
    /// The direction from the neighbor to the element in the neighbor's frame
    Direction<Dim> direction_from_neighbor;
    /// Whether the neighbor is a DG element, so it sends its volume data,
    /// or an FD element of another patch, so it sends the ghost zone.
    bool neighbor_is_dg;
    /// Where the ghost data received from the neighbor (without the RDMP
    /// data) starts in the buffer passed to `fill_from_external()`. The
    /// data takes `data_size` values, variable by variable.
    size_t data_offset;
    size_t data_size;
    /// Where the ghost data to send to the neighbor starts in the buffer
    /// filled by `slice_for_external()`. The data takes `send_size` values,
    /// variable by variable, in the layout of `slice_data`.
    size_t send_offset;
    size_t send_size;
  };

  GhostZones() = default;

  /// @{
  /// `elements` are the elements of the `patch` and
  /// `fd_to_neighbor_fd_interpolants` their interpolants, in slot order. The
  /// ghost variables are read in `parts` (see `GhostVariablesPart`), or all
  /// `number_of_variables` from one batch. The transfers are copied to the
  /// device on `exec`.
  GhostZones(const Kokkos::DefaultExecutionSpace& exec,
             const std::vector<Element<Dim>>& elements,
             const std::vector<
                 DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>>&
                 fd_to_neighbor_fd_interpolants,
             const evolution::patches::PatchTable<Dim>& patch_table,
             size_t patch, size_t ghost_width,
             std::vector<GhostVariablesPart> parts);

  GhostZones(const Kokkos::DefaultExecutionSpace& exec,
             const std::vector<Element<Dim>>& elements,
             const std::vector<
                 DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>>&
                 fd_to_neighbor_fd_interpolants,
             const evolution::patches::PatchTable<Dim>& patch_table,
             size_t patch, size_t ghost_width, size_t number_of_variables);
  /// @}

  /// The number of cells of each element in each dimension
  const Index<Dim>& subcell_extents() const { return subcell_extents_; }

  size_t ghost_width() const { return ghost_width_; }

  size_t number_of_variables() const { return number_of_variables_; }

  const std::vector<GhostVariablesPart>& parts() const { return parts_; }

  /// @{
  /// Copy the unpadded `ghost_variables` of all elements of the patch into
  /// `padded_vars`, and fill the ghost zones of the faces between elements of
  /// the patch. Enqueued on `exec`.
  ///
  /// There is one batch of `ghost_variables` for each of the `parts()` (see
  /// `GhostVariablesPart`). `padded_vars` must have been allocated with
  /// `fd::device::make_volume_view` for the elements of the patch,
  /// `number_of_variables()` variables, `subcell_extents()` and
  /// `ghost_width()`.
  void fill_from_patch(
      const Kokkos::DefaultExecutionSpace& exec,
      const ::fd::device::VolumeView& padded_vars,
      const std::vector<Kokkos::View<const double*>>& ghost_variables) const;

  void fill_from_patch(
      const Kokkos::DefaultExecutionSpace& exec,
      const ::fd::device::VolumeView& padded_vars,
      const Kokkos::View<const double*>& ghost_variables) const;
  /// @}

  /// The faces whose neighbor isn't in the patch, ordered by slot and then
  /// direction.
  const std::vector<ExternalFace>& external_faces() const {
    return external_faces_;
  }

  /// The size of the buffer that holds the ghost data received for all
  /// `external_faces()`.
  size_t external_data_size() const { return external_data_size_; }

  /// Fill the ghost zones of the `external_faces()` in `padded_vars` from the
  /// received ghost data in `external_data`, laid out as described by
  /// `ExternalFace`. Enqueued on `exec`.
  void fill_from_external(
      const Kokkos::DefaultExecutionSpace& exec,
      const ::fd::device::VolumeView& padded_vars,
      const Kokkos::View<const double*>& external_data) const;

  /// The size of the buffer that holds the ghost data to send for all
  /// `external_faces()`.
  size_t send_data_size() const { return send_data_size_; }

  /// @{
  /// Compute the ghost data to send for all `external_faces()` from the
  /// unpadded `ghost_variables` of the patch (one batch for each of the
  /// `parts()`) into `send_data`, laid out as described by `ExternalFace`.
  /// Enqueued on `exec`.
  void slice_for_external(
      const Kokkos::DefaultExecutionSpace& exec,
      const Kokkos::View<double*>& send_data,
      const std::vector<Kokkos::View<const double*>>& ghost_variables) const;

  void slice_for_external(
      const Kokkos::DefaultExecutionSpace& exec,
      const Kokkos::View<double*>& send_data,
      const Kokkos::View<const double*>& ghost_variables) const;
  /// @}

 private:
  void add_external_face(
      const Element<Dim>& element, size_t slot, size_t number_of_padded_cells,
      const Direction<Dim>& direction, const ElementId<Dim>& neighbor_id,
      bool neighbor_is_dg, const std::vector<size_t>& ghost_zone_points,
      const Mesh<Dim>& dg_mesh,
      const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
          fd_to_neighbor_fd_interpolants);

  // Where the variables of `part` of the element in `slot` start in the batch
  // of the part
  size_t source_offset(size_t part, size_t slot) const;

  // Checks the sizes of the batches of the parts
  void check_ghost_variables(
      const std::vector<Kokkos::View<const double*>>& ghost_variables) const;

  size_t number_of_elements_ = 0;
  Index<Dim> subcell_extents_{};
  size_t ghost_width_ = 0;
  std::vector<GhostVariablesPart> parts_{};
  // The first variable of each part in the padded view and the ghost data
  std::vector<size_t> first_variables_{};
  size_t number_of_variables_ = 0;
  // For each part
  std::vector<::fd::device::Transfers> from_patch_{};
  std::vector<ExternalFace> external_faces_{};
  size_t external_data_size_ = 0;
  ::fd::device::Transfers from_external_{};
  // The matrix ids in `from_external_` of the projections from DG neighbors
  DirectionMap<Dim, size_t> dg_projections_{};
  size_t send_data_size_ = 0;
  // For each part
  std::vector<::fd::device::Transfers> to_external_{};
};

/*!
 * \brief The flat indices of the cells of an element next to its face in
 * `direction`, in the order of the ghost zone of the neighbor across that face,
 * whose frame is related to the element's by `orientation_of_neighbor`.
 *
 * \details The cells are the `ghost_width` layers next to the face, in the
 * order of `fd::device::slab_points` reoriented to the neighbor's frame with
 * `orient_variables`, i.e. as the host would orient the sliced data. Both
 * elements must have the subcell `extents` (in their own frames).
 */
template <size_t Dim>
std::vector<size_t> cells_for_neighbor_ghost_zone(
    const Index<Dim>& extents, size_t ghost_width,
    const Direction<Dim>& direction,
    const OrientationMap<Dim>& orientation_of_neighbor);

/*!
 * \brief Check that the `elements` of the `patch` can copy the ghost data of
 * their FD neighbors in other blocks, i.e., that the cells of the neighbor that
 * fill the ghost zones of an element are exactly where its ghost cells are.
 *
 * \details `GhostZones` copies the ghost data between blocks, reoriented if
 * the blocks have different orientations, so the blocks must have the same
 * subcell extents, and their cells must line up across the block boundary, up
 * to a periodic shift (see
 * `SetInterpolators_detail::periodic_offset`). Otherwise the host interpolates
 * between the blocks (see `SetInterpolators`), which FD patches don't support
 * yet, so this is an error. The check is in the grid frame, whose maps don't
 * depend on time.
 */
template <size_t Dim>
void check_cells_coincide_across_blocks(
    const Domain<Dim>& domain, const std::vector<Element<Dim>>& elements,
    const evolution::patches::PatchTable<Dim>& patch_table, size_t patch,
    size_t ghost_width);
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
