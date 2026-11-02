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
 * \brief Add the Cartesian flux divergence of a batch of elements to `dt_vars`
 * in a Kokkos kernel on `exec`.
 *
 * Computes the same as `evolution::dg::subcell::add_cartesian_flux_divergence`
 * for all variables and directions:
 *
 * \f{align*}{
 *   \partial_t u_{i} \mathrel{+}= \sum_d \frac{1}{\Delta\xi^d}
 *   \det\left(\frac{\partial\xi}{\partial x}\right)_{i}
 *   \left(\hat{F}^d_{i+1/2} - \hat{F}^d_{i-1/2}\right),
 * \f}
 *
 * adding the directions in order, so the result is the same as calling the
 * CPU function once per direction.
 *
 * \param dt_vars the time derivatives, with the extents of the volume data.
 * \param boundary_corrections the boundary corrections \f$\hat{F}^d\f$ on the
 * faces in each direction `d`, with `volume_extents[d] + 1` faces in
 * dimension `d` and the same variables as `dt_vars`.
 * \param det_inverse_jacobian the determinant of the inverse Jacobian at the
 * cell centers, with one variable.
 * \param one_over_delta the inverse logical grid spacing in each direction.
 * \param volume_extents the number of cells in each element.
 */
template <size_t Dim>
void add_cartesian_flux_divergence(
    const Kokkos::DefaultExecutionSpace& exec, const VolumeView& dt_vars,
    const std::array<ConstVolumeView, Dim>& boundary_corrections,
    const ConstVolumeView& det_inverse_jacobian,
    const std::array<double, Dim>& one_over_delta,
    const Index<Dim>& volume_extents);
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
