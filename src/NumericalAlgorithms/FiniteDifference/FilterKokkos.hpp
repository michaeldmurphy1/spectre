// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Index;
/// \endcond

namespace fd::device {
/*!
 * \brief Apply Kreiss-Oliger dissipation to a batch of elements in a Kokkos
 * kernel on `exec`, as `fd::kreiss_oliger_filter`.
 *
 * \details Every cell is handled by its own thread, which adds the dissipation
 * in each logical direction, in the same order as `fd::kreiss_oliger_filter`
 * so the results agree, and the value of the cell.
 *
 * \param exec the execution space instance that runs the kernel.
 * \param filtered_vars the filtered variables, with the same extents as the
 * unpadded volume data. Must not overlap `padded_volume_vars`.
 * \param padded_volume_vars the volume data padded with ghost zones of width
 * at least `fd_order / 2`, e.g. a subview of some of the variables of a
 * padded `VolumeView`.
 * \param volume_extents the number of cells in each element. Must be
 * isotropic.
 * \param fd_order the order of the dissipation. Can be 2, 4, 6, 8 or 10.
 * \param epsilon the strength of the dissipation.
 */
template <size_t Dim>
void kreiss_oliger_filter(const Kokkos::DefaultExecutionSpace& exec,
                          const StridedVolumeView& filtered_vars,
                          const ConstStridedVolumeView& padded_volume_vars,
                          const Index<Dim>& volume_extents, size_t fd_order,
                          double epsilon);
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
