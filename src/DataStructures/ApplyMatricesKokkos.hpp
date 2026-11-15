// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
class Matrix;
/// \endcond

/*!
 * \brief Apply a 1D matrix in each dimension to many components of a batch of
 * elements in device memory, as `apply_matrices` does on the host.
 *
 * \details The data is indexed by grid point, component and element, as the
 * storage of a `BatchedVariables`, with the first dimension varying fastest
 * within the grid points of a component. An empty matrix stands for the
 * identity in that dimension, which is skipped. The dimensions are applied one
 * at a time (cost \f$\sim N^{d+1}\f$ per component rather than \f$N^{2d}\f$),
 * each in one kernel over all components and elements, so applying all three
 * matrices enqueues three kernels.
 *
 * The matrices are copied to the device once on construction. The
 * intermediate results of the dimensions are kept in buffers that are
 * allocated in the first `apply` and reused while the number of components
 * and elements doesn't grow, so repeated calls don't allocate.
 */
class DeviceApplyMatrices {
 public:
  using View = Kokkos::View<double***, Kokkos::LayoutStride>;
  using ConstView = Kokkos::View<const double***, Kokkos::LayoutStride>;

  DeviceApplyMatrices() = default;
  /// `matrices[d]` maps the `source_extents[d]` points in dimension `d` to
  /// `matrices[d].rows()` points. If `source_components` isn't empty, `apply`
  /// reads only these components of the source, and writes them to the
  /// components `0, 1, ...` of the destination, e.g. to project the
  /// components of a row of a tensor.
  DeviceApplyMatrices(const std::array<Matrix, 3>& matrices,
                      const Index<3>& source_extents,
                      const std::vector<size_t>& source_components = {});

  /// The extents of the result
  const Index<3>& destination_extents() const { return destination_extents_; }

  /// Apply the matrices to `source` and write the result to `destination`,
  /// enqueued on `exec`. Both have the same number of elements and (unless
  /// `source_components` were given) components, and `source` has the points
  /// of `source_extents` and `destination` those of `destination_extents()`.
  /// `destination` must not overlap `source`.
  void apply(const Kokkos::DefaultExecutionSpace& exec, const View& destination,
             const ConstView& source);

 private:
  std::array<Kokkos::View<double**, Kokkos::LayoutRight>, 3> matrices_{};
  std::array<bool, 3> is_identity_{};
  Index<3> source_extents_{};
  Index<3> destination_extents_{};
  // Empty to read all components
  Kokkos::View<size_t*> source_components_{};
  std::array<Kokkos::View<double***, Kokkos::LayoutLeft>, 2> buffers_{};
};
#endif  // SPECTRE_KOKKOS
