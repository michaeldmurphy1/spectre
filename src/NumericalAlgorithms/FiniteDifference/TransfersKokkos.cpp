// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device {
namespace {
template <typename T>
Kokkos::View<T*> to_device(const Kokkos::DefaultExecutionSpace& exec,
                           const std::string& label,
                           const std::vector<T>& host_data) {
  Kokkos::View<T*> result(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label),
      host_data.size());
  Kokkos::deep_copy(
      exec, result,
      Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          host_data.data(), host_data.size()));
  return result;
}

// Extents in (x, y, z) order, with 1 for unused dimensions
template <size_t Dim>
std::array<size_t, 3> extents_3d(const Index<Dim>& extents) {
  std::array<size_t, 3> result{{1, 1, 1}};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = extents[d];
  }
  return result;
}
}  // namespace

void Transfers::add_transfer(const Transfer& transfer,
                             const std::vector<size_t>& source_points,
                             const std::vector<size_t>& destination_points) {
  ASSERT(not on_device_,
         "Cannot add transfers after copying them to the "
         "device");
  if (transfers_.size() >=
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    ERROR("Too many transfers: " << transfers_.size());
  }
  const int transfer_index = static_cast<int>(transfers_.size());
  transfers_.push_back(transfer);
  for (size_t i = 0; i < destination_points.size(); ++i) {
    point_transfer_.push_back(transfer_index);
    point_source_.push_back(source_points.empty() ? i : source_points[i]);
    point_destination_.push_back(destination_points[i]);
  }
}

void Transfers::add_copy(const size_t source_offset,
                         const size_t source_variable_stride,
                         const size_t destination_offset,
                         const size_t destination_variable_stride,
                         const std::vector<size_t>& source_points,
                         const std::vector<size_t>& destination_points) {
  ASSERT(source_points.size() == destination_points.size(),
         "Copying " << source_points.size() << " source points to "
                    << destination_points.size() << " destination points");
  add_transfer(
      Transfer{.source_offset = source_offset,
               .source_variable_stride = source_variable_stride,
               .destination_offset = destination_offset,
               .destination_variable_stride = destination_variable_stride,
               .matrix_columns = 0,
               .matrix_offset = 0,
               .matrix_rows = 0},
      source_points, destination_points);
}

void Transfers::add_matrix(const size_t source_offset,
                           const size_t source_variable_stride,
                           const size_t destination_offset,
                           const size_t destination_variable_stride,
                           const Matrix& matrix,
                           const std::vector<size_t>& destination_points) {
  ASSERT(matrix.rows() == destination_points.size(),
         "The matrix has " << matrix.rows() << " rows but there are "
                           << destination_points.size()
                           << " destination points");
  ASSERT(matrix.columns() > 0, "The matrix must have at least one column");
  const size_t matrix_offset = matrix_values_.size();
  // Column-major, like `Matrix`, so consecutive rows are read together
  for (size_t column = 0; column < matrix.columns(); ++column) {
    for (size_t row = 0; row < matrix.rows(); ++row) {
      matrix_values_.push_back(matrix(row, column));
    }
  }
  add_transfer(
      Transfer{.source_offset = source_offset,
               .source_variable_stride = source_variable_stride,
               .destination_offset = destination_offset,
               .destination_variable_stride = destination_variable_stride,
               .matrix_columns = matrix.columns(),
               .matrix_offset = matrix_offset,
               .matrix_rows = matrix.rows()},
      {}, destination_points);
}

void Transfers::copy_to_device(const Kokkos::DefaultExecutionSpace& exec) {
  device_transfers_ = to_device(exec, "transfers", transfers_);
  device_point_transfer_ = to_device(exec, "point transfers", point_transfer_);
  device_point_source_ = to_device(exec, "point sources", point_source_);
  device_point_destination_ =
      to_device(exec, "point destinations", point_destination_);
  device_matrix_values_ = to_device(exec, "matrix values", matrix_values_);
  // The host data must stay valid until the copies are done
  exec.fence();
  on_device_ = true;
}

void Transfers::apply(const Kokkos::DefaultExecutionSpace& exec,
                      const Kokkos::View<double*>& destination,
                      const Kokkos::View<const double*>& source,
                      const size_t number_of_variables) const {
  ASSERT(on_device_, "The transfers must be copied to the device first");
  const size_t number_of_points = point_transfer_.size();
  const auto transfers = device_transfers_;
  const auto point_transfer = device_point_transfer_;
  const auto point_source = device_point_source_;
  const auto point_destination = device_point_destination_;
  const auto matrix_values = device_matrix_values_;
  // Consecutive threads handle consecutive points of the same variable
  Kokkos::parallel_for(
      "fd::device::Transfers::apply",
      Kokkos::RangePolicy<>(exec, 0, number_of_points * number_of_variables),
      KOKKOS_LAMBDA(const size_t index) {
        const size_t point = index % number_of_points;
        const size_t var = index / number_of_points;
        const Transfers_detail::Transfer& transfer =
            transfers(point_transfer(point));
        const size_t source_start =
            transfer.source_offset + var * transfer.source_variable_stride;
        double value = 0.0;
        if (transfer.matrix_columns == 0) {
          value = source(source_start + point_source(point));
        } else {
          const size_t row = point_source(point);
          for (size_t column = 0; column < transfer.matrix_columns; ++column) {
            value += matrix_values(transfer.matrix_offset + row +
                                   column * transfer.matrix_rows) *
                     source(source_start + column);
          }
        }
        destination(transfer.destination_offset +
                    var * transfer.destination_variable_stride +
                    point_destination(point)) = value;
      });
}

template <size_t Dim>
std::vector<size_t> slab_points(const Index<Dim>& extents,
                                const size_t dimension, const size_t first,
                                const size_t last) {
  ASSERT(dimension < Dim and first <= last and last <= extents[dimension],
         "Invalid slab [" << first << ", " << last << ") in dimension "
                          << dimension << " of extents " << extents);
  auto lower_bounds = std::array<size_t, 3>{{0, 0, 0}};
  auto upper_bounds = extents_3d(extents);
  gsl::at(lower_bounds, dimension) = first;
  gsl::at(upper_bounds, dimension) = last;
  const auto all_extents = extents_3d(extents);
  std::vector<size_t> result{};
  for (size_t k = lower_bounds[2]; k < upper_bounds[2]; ++k) {
    for (size_t j = lower_bounds[1]; j < upper_bounds[1]; ++j) {
      for (size_t i = lower_bounds[0]; i < upper_bounds[0]; ++i) {
        result.push_back(i + all_extents[0] * (j + all_extents[1] * k));
      }
    }
  }
  return result;
}

template <size_t Dim>
std::vector<size_t> padded_ghost_zone_points(const Index<Dim>& volume_extents,
                                             const size_t ghost_width,
                                             const size_t dimension,
                                             const bool upper_side) {
  ASSERT(dimension < Dim, "Invalid dimension " << dimension);
  auto padded_extents = extents_3d(volume_extents);
  auto offsets = std::array<size_t, 3>{{0, 0, 0}};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(padded_extents, d) += 2 * ghost_width;
    gsl::at(offsets, d) = ghost_width;
  }
  // The ghost zone has `ghost_width` cells in `dimension` and the extents of
  // the element otherwise
  auto zone_extents = extents_3d(volume_extents);
  gsl::at(zone_extents, dimension) = ghost_width;
  gsl::at(offsets, dimension) =
      upper_side ? ghost_width + volume_extents[dimension] : 0;
  std::vector<size_t> result{};
  for (size_t k = 0; k < zone_extents[2]; ++k) {
    for (size_t j = 0; j < zone_extents[1]; ++j) {
      for (size_t i = 0; i < zone_extents[0]; ++i) {
        result.push_back(
            (i + offsets[0]) +
            padded_extents[0] *
                ((j + offsets[1]) + padded_extents[1] * (k + offsets[2])));
      }
    }
  }
  return result;
}

template <size_t Dim>
std::vector<size_t> padded_interior_points(const Index<Dim>& volume_extents,
                                           const size_t ghost_width) {
  auto padded_extents = extents_3d(volume_extents);
  auto offsets = std::array<size_t, 3>{{0, 0, 0}};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(padded_extents, d) += 2 * ghost_width;
    gsl::at(offsets, d) = ghost_width;
  }
  const auto extents = extents_3d(volume_extents);
  std::vector<size_t> result{};
  result.reserve(volume_extents.product());
  for (size_t k = 0; k < extents[2]; ++k) {
    for (size_t j = 0; j < extents[1]; ++j) {
      for (size_t i = 0; i < extents[0]; ++i) {
        result.push_back(
            (i + offsets[0]) +
            padded_extents[0] *
                ((j + offsets[1]) + padded_extents[1] * (k + offsets[2])));
      }
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template std::vector<size_t> slab_points(const Index<DIM(data)>& extents, \
                                           size_t dimension, size_t first,  \
                                           size_t last);                    \
  template std::vector<size_t> padded_ghost_zone_points(                    \
      const Index<DIM(data)>& volume_extents, size_t ghost_width,           \
      size_t dimension, bool upper_side);                                   \
  template std::vector<size_t> padded_interior_points(                      \
      const Index<DIM(data)>& volume_extents, size_t ghost_width);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
