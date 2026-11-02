// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Index;
class Matrix;
/// \endcond

namespace fd::device {
namespace Transfers_detail {
// The description of one transfer. This isn't a private member of
// `Transfers`, because nvcc doesn't allow capturing private types in device
// lambdas.
struct Transfer {
  size_t source_offset;
  size_t source_variable_stride;
  size_t destination_offset;
  size_t destination_variable_stride;
  // Number of matrix columns, or 0 for a copy
  size_t matrix_columns;
  // Offset of the column-major matrix in the matrix values
  size_t matrix_offset;
  size_t matrix_rows;
};
}  // namespace Transfers_detail

/*!
 * \brief Copies and interpolations of several variables between flat buffers
 * in device memory, e.g., to slice ghost data for neighbors or to fill the
 * ghost zones of a padded view.
 *
 * Each transfer reads `number_of_variables` variables from a source buffer and
 * writes the same number of variables to a destination buffer. Variable `v`
 * of a transfer starts at `offset + v * variable_stride` in each buffer, so a
 * transfer can read from or write to the storage of a `Variables`, one
 * element of a `VolumeView`, or a part of a buffer that is sent to a
 * neighbor. A transfer either
 * - copies the source points `source_points[i]` to the destination points
 *   `destination_points[i]`, which also handles reorienting data, or
 * - applies a dense interpolation matrix (e.g., from an `intrp::Irregular`) to
 *   the source points `0, ..., columns - 1`, writing row `i` to the
 *   destination point `destination_points[i]`.
 *
 * The transfers are described on the host with `add_copy` and `add_matrix`,
 * copied to the device once with `copy_to_device`, and can then be applied
 * any number of times. They only depend on the layout of the data, so they can
 * be built once, e.g., for an element and its neighbors. All transfers are
 * applied in a single kernel, so one object can describe all the ghost data of
 * an element or of a batch of elements, independently of how the elements
 * are distributed and where their data is stored.
 *
 * Destination points must not overlap, since the transfers run concurrently.
 */
class Transfers {
 public:
  /// Copy `source_points[i]` to `destination_points[i]`
  void add_copy(size_t source_offset, size_t source_variable_stride,
                size_t destination_offset, size_t destination_variable_stride,
                const std::vector<size_t>& source_points,
                const std::vector<size_t>& destination_points);

  /// Write row `i` of `matrix` applied to the source points
  /// `0, ..., matrix.columns() - 1` to `destination_points[i]`
  void add_matrix(size_t source_offset, size_t source_variable_stride,
                  size_t destination_offset, size_t destination_variable_stride,
                  const Matrix& matrix,
                  const std::vector<size_t>& destination_points);

  /// Store `matrix` once for several transfers, e.g. the same projection for
  /// the faces of many elements. Returns the id to pass to `add_matrix`.
  size_t store_matrix(const Matrix& matrix);

  /// Like `add_matrix` above, with a matrix stored with `store_matrix`
  void add_matrix(size_t source_offset, size_t source_variable_stride,
                  size_t destination_offset, size_t destination_variable_stride,
                  size_t matrix_id,
                  const std::vector<size_t>& destination_points);

  /// The number of destination points of all transfers
  size_t number_of_points() const { return point_transfer_.size(); }

  /// Copy the transfers to the device. Must be called after adding the
  /// transfers and before applying them.
  void copy_to_device(const Kokkos::DefaultExecutionSpace& exec);

  /// Apply the transfers to `number_of_variables` variables, enqueued on
  /// `exec`.
  void apply(const Kokkos::DefaultExecutionSpace& exec,
             const Kokkos::View<double*>& destination,
             const Kokkos::View<const double*>& source,
             size_t number_of_variables) const;

 private:
  using Transfer = Transfers_detail::Transfer;

  void add_transfer(const Transfer& transfer,
                    const std::vector<size_t>& source_points,
                    const std::vector<size_t>& destination_points);

  // Host descriptions
  std::vector<Transfer> transfers_{};
  // For each destination point: its transfer, the source point (for a copy)
  // or matrix row, and the destination point
  std::vector<int> point_transfer_{};
  std::vector<size_t> point_source_{};
  std::vector<size_t> point_destination_{};
  std::vector<double> matrix_values_{};
  struct StoredMatrix {
    size_t offset;
    size_t rows;
    size_t columns;
  };
  std::vector<StoredMatrix> matrices_{};
  // Device copies
  Kokkos::View<Transfer*> device_transfers_{};
  Kokkos::View<int*> device_point_transfer_{};
  Kokkos::View<size_t*> device_point_source_{};
  Kokkos::View<size_t*> device_point_destination_{};
  Kokkos::View<double*> device_matrix_values_{};
  bool on_device_ = false;
};

/// The flat indices, with \f$x\f$ varying fastest, of the cells of a volume
/// with `extents` whose index in `dimension` is in `[first, last)`, in the
/// order of `evolution::dg::subcell::slice_data`.
template <size_t Dim>
std::vector<size_t> slab_points(const Index<Dim>& extents, size_t dimension,
                                size_t first, size_t last);

/// The flat indices in a padded `VolumeView` element (see
/// `fd::device::make_volume_view`) of the `ghost_width` ghost cells on the
/// `upper_side` (or lower side) of the element in `dimension`, in the order of
/// `slab_points` of the ghost zone.
template <size_t Dim>
std::vector<size_t> padded_ghost_zone_points(const Index<Dim>& volume_extents,
                                             size_t ghost_width,
                                             size_t dimension, bool upper_side);

/// The flat indices in a padded `VolumeView` element of the cells of the
/// element itself, in the order of the unpadded data.
template <size_t Dim>
std::vector<size_t> padded_interior_points(const Index<Dim>& volume_extents,
                                           size_t ghost_width);
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
