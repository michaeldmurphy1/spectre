// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
class DataVector;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::device {
/*!
 * \brief Compute the logical partial derivatives of a batch of elements with
 * cell-centered finite differences in a Kokkos kernel on `exec`.
 *
 * This computes the same derivatives as `fd::logical_partial_derivatives`, see
 * its documentation for details. Instead of differentiating one stripe at a
 * time and transposing the data for the \f$\eta\f$ and \f$\zeta\f$
 * directions, every cell is handled by its own thread, which computes the
 * derivatives in all directions with the strides of the view.
 *
 * \param exec the execution space instance that runs the kernel.
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
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<VolumeView, Dim>& logical_derivatives,
    const ConstVolumeView& padded_volume_vars, const Index<Dim>& volume_extents,
    size_t fd_order);

/// The inverse Jacobian from the element-logical to the inertial frame of a
/// batch of elements, with each component indexed by grid point and element
/// like a `Tags::BatchView`
template <size_t Dim>
using BatchInverseJacobian = TensorMetafunctions::swap_type<
    Kokkos::View<double**, Kokkos::LayoutStride>,
    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>>;

/*!
 * \brief Compute the inertial partial derivatives of a batch of elements with
 * cell-centered finite differences in a Kokkos kernel on `exec`.
 *
 * \details This computes the same derivatives as `fd::partial_derivatives`
 * (without the Cartoon basis). Each thread computes the logical derivatives of
 * one variable at one cell as `fd::device::logical_partial_derivatives` does
 * and transforms them with the inverse Jacobian at the cell, so the logical
 * derivatives are never stored. The results are summed in the same order as
 * on the host.
 *
 * \param exec the execution space instance that runs the kernel.
 * \param partial_derivatives the derivatives, with the same extents as the
 * unpadded volume data and `Dim` times as many variables. As in the
 * `Variables` of `Tags::deriv` tags, the derivatives of each variable follow
 * each other, so variable `Dim * v + i` is the derivative of variable `v` in
 * inertial direction `i`.
 * \param padded_volume_vars the volume data padded with ghost zones of width
 * at least `fd_order / 2`, see `fd::device::pack_with_ghost_zones`. Can be a
 * subview of some of the variables.
 * \param inverse_jacobian the inverse Jacobian at the cell centers.
 * \param volume_extents the number of cells in each element. Must be
 * isotropic.
 * \param fd_order the order of the finite difference stencil. Can be 2, 4, 6,
 * 8 or 10.
 */
template <size_t Dim>
void partial_derivatives(const Kokkos::DefaultExecutionSpace& exec,
                         const StridedVolumeView& partial_derivatives,
                         const ConstStridedVolumeView& padded_volume_vars,
                         const BatchInverseJacobian<Dim>& inverse_jacobian,
                         const Index<Dim>& volume_extents, size_t fd_order);
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
