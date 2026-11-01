// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
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
 * \brief Monotonised central-difference reconstruction of a batch of elements
 * in Kokkos kernels on `exec`.
 *
 * This computes the same face values as
 * `fd::reconstruction::monotonised_central`, see its documentation for
 * details. Instead of reconstructing one stripe at a time and transposing the
 * data for the \f$\eta\f$ and \f$\zeta\f$ directions, every cell is handled
 * by its own thread, reading its neighbors from `padded_volume_vars` with the
 * stride of the view in each direction.
 *
 * \param exec the execution space instance that runs the kernels.
 * \param upper_side_of_face_vars the reconstructed values on the upper side of
 * each face, i.e. reconstructed from the cell above the face. In direction
 * `d` the views have `volume_extents[d] + 1` faces in dimension `d`. The views
 * can have any strides (see `StridedVolumeView`), so they can point into
 * storage that holds more variables, e.g., a batch of face variables.
 * \param lower_side_of_face_vars same as `upper_side_of_face_vars`, but for
 * the lower side of the faces.
 * \param padded_volume_vars the volume data padded with ghost zones of width
 * at least 2, see `fd::device::pack_with_ghost_zones`.
 * \param volume_extents the number of cells in each element.
 */
template <size_t Dim>
void monotonised_central(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<StridedVolumeView, Dim>& upper_side_of_face_vars,
    const std::array<StridedVolumeView, Dim>& lower_side_of_face_vars,
    const ConstVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents);
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
