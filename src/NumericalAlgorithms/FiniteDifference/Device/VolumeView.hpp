// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

/// Finite difference algorithms that run in Kokkos kernels
namespace fd::device {
/*!
 * \brief Volume data of a batch of elements with indices
 * `(element, variable, z, y, x)`.
 *
 * The `x` index varies fastest, so the data of each element has the same
 * layout as a `Variables` on the element. Unused dimensions have extent 1,
 * e.g. a 2d view has extents `(elements, variables, 1, ny, nx)`.
 *
 * Views padded with ghost zones (see `fd::device::pack_with_ghost_zones`) have
 * extent `n + 2 * ghost_width` in each used dimension, where `n` is the number
 * of cells in the element. Only the ghost zones adjacent to the faces of the
 * element are filled, not the edges and corners, because the stencils are
 * one-dimensional.
 *
 * \note Unlike the CPU implementations, which keep the ghost data in separate
 * buffers and copy it into a small stencil buffer near the element boundary,
 * the kernels use padded views. This costs one copy and the unused edges and
 * corners, but every thread then applies the same stencil with no branches and
 * coalesced memory access, for all directions and all kernels that use the
 * padded data.
 */
using VolumeView = Kokkos::View<double*****, Kokkos::LayoutRight>;
/// \copydoc VolumeView
using ConstVolumeView = Kokkos::View<const double*****, Kokkos::LayoutRight>;

/*!
 * \brief Allocate a `VolumeView` for `number_of_elements` elements with
 * `volume_extents` cells each, padded by `ghost_width` cells in each dimension.
 */
template <size_t Dim>
VolumeView make_volume_view(const std::string& label, size_t number_of_elements,
                            size_t number_of_variables,
                            const Index<Dim>& volume_extents,
                            size_t ghost_width = 0);

/*!
 * \brief Copy the volume data and ghost data of one element into the host
 * mirror of a padded `VolumeView`.
 *
 * The data is in the layout used by `fd::reconstruction` and
 * `fd::logical_partial_derivatives`: `ghost_cell_vars` holds the data of the
 * neighbor in each direction, with at least as many cells perpendicular to the
 * face as the ghost width of `padded_vars`. If there are more ghost cells, the
 * ones closest to the element are used.
 *
 * \note This is done on the host for now, so the padded data has to be copied
 * to the device afterwards. Eventually the volume and ghost data will already
 * be on the device and this will be a kernel that packs the whole batch at
 * once, taking views instead of spans for each element. Ideally the data would
 * be written directly into the padded view instead, e.g. the volume data into
 * a subview of its interior and received ghost data into its ghost zones, so no
 * copy is needed at all. In either case only the packing changes: the kernels
 * that take the padded view keep their interface.
 */
template <size_t Dim>
void pack_with_ghost_zones(
    const VolumeView::host_mirror_type& padded_vars, size_t element,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents);
}  // namespace fd::device
