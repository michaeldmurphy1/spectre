// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Index.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim, typename T>
class DirectionMap;
/// \endcond

/// Finite difference algorithms that run in Kokkos kernels
namespace fd::device {
/*!
 * \brief Volume data of a batch of elements with indices
 * `(element, variable, z, y, x)`.
 *
 * The `x` index varies fastest, so the data of each element has the same
 * layout as a `Variables` on the element, and an unpadded view has the same
 * layout as the storage of a `BatchedVariables`. Unused dimensions have
 * extent 1, e.g. a 2d view has extents `(elements, variables, 1, ny, nx)`.
 *
 * Views padded with ghost zones (see `fd::device::pack_with_ghost_zones`) have
 * extent `n + 2 * ghost_width` in each used dimension, where `n` is the number
 * of cells in the element. Only the ghost zones adjacent to the faces of the
 * element are filled, not the edges and corners, because the stencils are
 * one-dimensional. Kernels accept any padding that is at least as wide as
 * their stencils need, so one padded view can be shared by several kernels.
 *
 * \note Unlike the CPU implementations, which keep the ghost data in separate
 * buffers and copy it into a small stencil buffer near the element boundary,
 * the kernels use padded views. This costs one copy and the unused edges and
 * corners, but every thread then applies the same stencil with no branches and
 * coalesced memory access, for all directions and all kernels that use the
 * padded data. The kernels are likely limited by memory bandwidth, so future
 * optimizations should reduce global memory traffic rather than arithmetic,
 * guided by profiling: e.g. fusing the reconstruction with the computation of
 * the fluxes and the boundary correction at each face, so face values aren't
 * written to and read back from global memory, and staging pencils or slabs of
 * the padded data in shared memory. A whole padded element doesn't fit for
 * larger elements, e.g. 64 KB per variable for 16 cells and ghost zones of
 * width 2. These only change the kernels, not their interface.
 */
using VolumeView = Kokkos::View<double*****, Kokkos::LayoutRight>;
/// \copydoc VolumeView
using ConstVolumeView = Kokkos::View<const double*****, Kokkos::LayoutRight>;

/*!
 * \brief Data of a batch of elements indexed like a `VolumeView`, but with any
 * strides, e.g., a subview of the variables of a `VolumeView` or a view into
 * the storage of a `BatchedVariables` that holds more variables.
 *
 * A `VolumeView` converts to a `StridedVolumeView`.
 */
using StridedVolumeView = Kokkos::View<double*****, Kokkos::LayoutStride>;
/// \copydoc StridedVolumeView
using ConstStridedVolumeView =
    Kokkos::View<const double*****, Kokkos::LayoutStride>;

/*!
 * \brief Allocate a `VolumeView` for `number_of_elements` elements with
 * `volume_extents` cells each, padded by `ghost_width` cells in each dimension.
 *
 * The allocation is enqueued on the execution space instance `exec`. The data
 * is only initialized with `SPECTRE_NAN_INIT`, in which case it is set to
 * signaling NaNs like a `Variables`.
 */
template <size_t Dim>
VolumeView make_volume_view(const Kokkos::DefaultExecutionSpace& exec,
                            const std::string& label, size_t number_of_elements,
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

namespace detail {
/*!
 * \brief The width of the ghost zones of `padded_vars`, which holds elements
 * with `volume_extents` cells.
 *
 * Asserts that the padding is the same in all used dimensions.
 */
template <typename View, size_t Dim>
size_t ghost_width(const View& padded_vars, const Index<Dim>& volume_extents) {
  ASSERT(padded_vars.extent(4) >= volume_extents[0] and
             (padded_vars.extent(4) - volume_extents[0]) % 2 == 0,
         "The padded view has extent "
             << padded_vars.extent(4) << " in dimension 0, which isn't "
             << volume_extents[0] << " cells plus two ghost zones");
  const size_t result = (padded_vars.extent(4) - volume_extents[0]) / 2;
#ifdef SPECTRE_DEBUG
  for (size_t d = 0; d < 3; ++d) {
    const size_t expected_extent = d < Dim ? volume_extents[d] + 2 * result : 1;
    ASSERT(padded_vars.extent(4 - d) == expected_extent,
           "The padded view has extent "
               << padded_vars.extent(4 - d) << " in dimension " << d
               << " but expected " << expected_extent
               << " for ghost zones of width " << result);
  }
#endif  // SPECTRE_DEBUG
  return result;
}

/*!
 * \brief The cells `(element, variable, z, y, x)` of a batch, enumerated by a
 * flat index with `x` varying fastest.
 *
 * Kernels iterate over the cells with a `Kokkos::RangePolicy` of `size()` and
 * get the cell of each index with `operator()`. Unlike an `MDRangePolicy`,
 * consecutive threads always handle consecutive `x`, so memory access is
 * coalesced on all backends, and no threads are idle in tiles that extend past
 * the small extents of finite difference elements. Nothing needs to be tuned.
 */
class BatchCells {
 public:
  struct Cell {
    int element;
    int variable;
    /// The cell index in `(x, y, z)` order
    Kokkos::Array<int, 3> index;
  };

  /// `extents` are in `(x, y, z)` order. Errors if the total number of cells
  /// doesn't fit in an `int`.
  BatchCells(size_t number_of_elements, size_t number_of_variables,
             const Kokkos::Array<int, 3>& extents);

  int size() const { return size_; }

  KOKKOS_INLINE_FUNCTION Cell operator()(int index) const {
    Cell result{};
    for (size_t d = 0; d < 3; ++d) {
      result.index[d] = index % extents_[d];
      index /= extents_[d];
    }
    result.variable = index % number_of_variables_;
    result.element = index / number_of_variables_;
    return result;
  }

 private:
  int number_of_variables_;
  Kokkos::Array<int, 3> extents_;
  int size_;
};
}  // namespace detail
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
