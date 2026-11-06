// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Mesh;
class TimeStepId;
namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
class GhostZones;
template <size_t Dim>
class PackagedDataFromDg;
}  // namespace evolution::dg::subcell::fd_patch
/// \endcond

namespace evolution::dg::subcell::fd_patch {
/// Boundary data that an element of a patch sends to a neighbor outside the
/// patch
template <size_t Dim>
struct OutgoingBoundaryData {
  ElementId<Dim> receiver;
  /// Whether the receiver is a DG element or an element of another patch
  bool receiver_is_dg;
  /// The key of the data in the receiver's inbox: the direction from the
  /// receiver to the sender in the receiver's frame, and the sender
  DirectionalId<Dim> sender;
  evolution::dg::BoundaryData<Dim> data;
};

/*!
 * \brief Whether the `messages` for a time step hold the boundary data of all
 * faces of the patch whose neighbor is outside the patch (see
 * `GhostZones::external_faces()`).
 *
 * `element_ids` are the elements of the patch in slot order.
 */
template <size_t Dim>
bool received_all_boundary_data(
    const GhostZones<Dim>& ghost_zones,
    const std::vector<ElementId<Dim>>& element_ids,
    const typename Tags::BoundaryDataInbox<Dim>::messages_type& messages);

/*!
 * \brief Copy the received `messages` of a time step into the buffers of
 * `GhostZones::fill_from_external()` and `PackagedDataFromDg::apply()`.
 *
 * \details The ghost data sent by each neighbor ends with
 * `number_of_rdmp_values` values for the RDMP TCI, which FD patches don't use
 * since their elements never switch to DG. DG neighbors also send their
 * packaged data on the mortar. Errors if data is missing or has an unexpected
 * size, e.g. because a DG neighbor has a different mesh.
 */
template <size_t Dim>
void unpack_boundary_data(
    gsl::span<double> ghost_data, gsl::span<double> mortar_data,
    const GhostZones<Dim>& ghost_zones,
    const PackagedDataFromDg<Dim>& packaged_data_from_dg,
    const std::vector<ElementId<Dim>>& element_ids,
    const typename Tags::BoundaryDataInbox<Dim>::messages_type& messages,
    size_t number_of_rdmp_values);

/*!
 * \brief The boundary data that the elements of a patch send to their
 * neighbors outside the patch, from the `send_data` computed by
 * `GhostZones::slice_for_external()`.
 *
 * \details The data has the layout that an FD element sends on the host (see
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`): the sliced
 * ghost data followed by the maximum and then the minimum RDMP values of the
 * sending element, with the DG mesh `dg_mesh` and the subcell mesh of the
 * sender.
 */
template <size_t Dim>
std::vector<OutgoingBoundaryData<Dim>> pack_boundary_data(
    const GhostZones<Dim>& ghost_zones, gsl::span<const double> send_data,
    const std::vector<ElementId<Dim>>& element_ids, const Mesh<Dim>& dg_mesh,
    const std::vector<RdmpTciData>& rdmp_tci_data,
    const TimeStepId& next_time_step_id, int tci_status,
    size_t integration_order);
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
