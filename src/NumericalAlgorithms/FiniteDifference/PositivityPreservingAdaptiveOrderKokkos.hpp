// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Index;
/// \endcond

namespace fd::device {
/*!
 * \brief Positivity-preserving adaptive-order reconstruction of a batch of
 * elements in Kokkos kernels on `exec`.
 *
 * This computes the same face values as
 * `fd::reconstruction::positivity_preserving_adaptive_order`, see its
 * documentation for details, with each cell handled by its own thread (see
 * `fd::device::monotonised_central`). The 7th and 9th orders are used if
 * `six_to_the_alpha_7` and `eight_to_the_alpha_9` hold values.
 *
 * \param exec the execution space instance that runs the kernels.
 * \param upper_side_of_face_vars the reconstructed values on the upper side of
 * each face (see `fd::device::monotonised_central`).
 * \param lower_side_of_face_vars same as `upper_side_of_face_vars`, but for
 * the lower side of the faces.
 * \param padded_volume_vars the volume data padded with ghost zones of width
 * at least 3 (4 with the 7th order, 5 with the 9th order), e.g. a subview of
 * some of the variables of a `ConstVolumeView`.
 * \param volume_extents the number of cells in each element.
 * \param positivity_preserving whether to fall back to lower orders if the
 * reconstructed values aren't positive.
 * \param low_order_reconstructor the reconstruction used if the 5th order
 * isn't, `Minmod` or `MonotonisedCentral`.
 * \param four_to_the_alpha_5 \f$4^{\alpha_5}\f$
 * \param six_to_the_alpha_7 \f$6^{\alpha_7}\f$
 * \param eight_to_the_alpha_9 \f$8^{\alpha_9}\f$
 */
template <size_t Dim>
void positivity_preserving_adaptive_order(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<StridedVolumeView, Dim>& upper_side_of_face_vars,
    const std::array<StridedVolumeView, Dim>& lower_side_of_face_vars,
    const ConstStridedVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents, bool positivity_preserving,
    ::fd::reconstruction::FallbackReconstructorType low_order_reconstructor,
    double four_to_the_alpha_5, const std::optional<double>& six_to_the_alpha_7,
    const std::optional<double>& eight_to_the_alpha_9);
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
