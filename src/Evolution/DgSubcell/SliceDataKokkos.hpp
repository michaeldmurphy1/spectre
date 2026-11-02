// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Index;
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Device version of `slice_data`, with the same arguments and result.
 *
 * All directions are sliced, or interpolated with the
 * `fd_to_neighbor_fd_interpolants`, in one kernel on `exec` with
 * `fd::device::Transfers`. As for `slice_data`, directions whose interpolant
 * is `std::nullopt` are not filled, because the neighbor interpolates the DG
 * solution instead. The volume data is copied to the device (unless it is
 * already there) and the sliced data back to the host, since the data sent to
 * neighbors is still stored on the host.
 */
/// @{
template <size_t Dim>
DirectionMap<Dim, DataVector> slice_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const DataVector& volume_subcell_vars, const Index<Dim>& subcell_extents,
    size_t number_of_ghost_points,
    const std::unordered_set<Direction<Dim>>& directions_to_slice,
    size_t additional_buffer,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        fd_to_neighbor_fd_interpolants);

/// With the volume data already on the device, e.g., computed there by the
/// `GhostDataMutator` of `SendDataForReconstruction`.
template <size_t Dim>
DirectionMap<Dim, DataVector> slice_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<const double*>& volume_subcell_vars,
    const Index<Dim>& subcell_extents, size_t number_of_ghost_points,
    const std::unordered_set<Direction<Dim>>& directions_to_slice,
    size_t additional_buffer,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        fd_to_neighbor_fd_interpolants);
/// @}
}  // namespace evolution::dg::subcell
#endif  // SPECTRE_KOKKOS
