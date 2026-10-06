// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/FiniteDifference/Device/VolumeView.hpp"

/// \cond
template <size_t Dim>
class Index;
/// \endcond

namespace fd::device {
/*!
 * \brief Compute the logical partial derivatives of a batch of elements with
 * cell-centered finite differences in a Kokkos kernel.
 *
 * This computes the same derivatives as `fd::logical_partial_derivatives`, see
 * its documentation for details. Instead of differentiating one stripe at a
 * time and transposing the data for the \f$\eta\f$ and \f$\zeta\f$
 * directions, every cell is handled by its own thread, which computes the
 * derivatives in all directions with the strides of the view.
 *
 * \param logical_derivatives the derivatives in each logical direction, with
 * the same extents as the unpadded volume data.
 * \param padded_volume_vars the volume data padded with ghost zones of width
 * at least `fd_order / 2`, see `fd::device::pack_with_ghost_zones`.
 * \param volume_extents the number of cells in each element. Must be
 * isotropic.
 * \param fd_order the order of the finite difference stencil. Can be 2, 4, 6,
 * 8 or 10.
 */
template <size_t Dim>
void logical_partial_derivatives(
    const std::array<VolumeView, Dim>& logical_derivatives,
    const ConstVolumeView& padded_volume_vars, const Index<Dim>& volume_extents,
    size_t fd_order);
}  // namespace fd::device
