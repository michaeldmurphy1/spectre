// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/ApplyMatricesKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// Apply `matrix` in `dimension` to data with `source_extents`, writing data
// whose extent in `dimension` is the number of rows of `matrix`. Reads the
// components `source_components` of the source, or all if it is empty.
template <typename Destination, typename Source>
void apply_in_dimension(
    const Kokkos::DefaultExecutionSpace& exec, const Destination& destination,
    const Source& source,
    const Kokkos::View<double**, Kokkos::LayoutRight>& matrix,
    const size_t dimension, const Index<3>& source_extents,
    const Kokkos::View<size_t*>& source_components) {
  Index<3> destination_extents = source_extents;
  destination_extents[dimension] = matrix.extent(0);
  const size_t number_of_points = destination_extents.product();
  const size_t number_of_components = destination.extent(1);
  const size_t number_of_elements = source.extent(2);
  const bool select_components = source_components.extent(0) != 0;
  const size_t columns = matrix.extent(1);
  // The stride of `dimension` in the source
  const size_t source_stride =
      dimension == 0 ? 1
                     : (dimension == 1 ? source_extents[0]
                                       : source_extents[0] * source_extents[1]);
  const size_t extent_0 = destination_extents[0];
  const size_t extent_1 = destination_extents[1];
  const size_t source_extent_0 = source_extents[0];
  const size_t source_extent_1 = source_extents[1];
  Kokkos::parallel_for(
      "DeviceApplyMatrices",
      Kokkos::RangePolicy<>(
          exec, 0,
          number_of_points * number_of_components * number_of_elements),
      KOKKOS_LAMBDA(const size_t k) {
        const size_t point = k % number_of_points;
        const size_t component = (k / number_of_points) % number_of_components;
        const size_t element = k / (number_of_points * number_of_components);
        const size_t i0 = point % extent_0;
        const size_t i1 = (point / extent_0) % extent_1;
        const size_t i2 = point / (extent_0 * extent_1);
        const size_t row = (dimension == 0 ? i0 : (dimension == 1 ? i1 : i2));
        // The source point with index 0 in `dimension`
        const size_t first_source_point =
            (dimension == 0 ? 0 : i0) +
            source_extent_0 * ((dimension == 1 ? 0 : i1) +
                               source_extent_1 * (dimension == 2 ? 0 : i2));
        const size_t source_component =
            select_components ? source_components(component) : component;
        double result = 0.0;
        for (size_t column = 0; column < columns; ++column) {
          result += matrix(row, column) *
                    source(first_source_point + column * source_stride,
                           source_component, element);
        }
        destination(point, component, element) = result;
      });
}
}  // namespace

DeviceApplyMatrices::DeviceApplyMatrices(
    const std::array<Matrix, 3>& matrices, const Index<3>& source_extents,
    const std::vector<size_t>& source_components)
    : source_extents_(source_extents), destination_extents_(source_extents) {
  if (not source_components.empty()) {
    source_components_ = Kokkos::View<size_t*>{
        "DeviceApplyMatrices source components", source_components.size()};
    const auto host_components = Kokkos::create_mirror_view(source_components_);
    for (size_t i = 0; i < source_components.size(); ++i) {
      host_components(i) = source_components[i];
    }
    Kokkos::deep_copy(source_components_, host_components);
  }
  for (size_t d = 0; d < 3; ++d) {
    const Matrix& matrix = gsl::at(matrices, d);
    gsl::at(is_identity_, d) = matrix.rows() == 0 and matrix.columns() == 0;
    if (gsl::at(is_identity_, d)) {
      continue;
    }
    if (matrix.columns() != source_extents[d]) {
      ERROR("The matrix in dimension " << d << " has " << matrix.columns()
                                       << " columns, but the source has "
                                       << source_extents[d]
                                       << " points in that dimension.");
    }
    destination_extents_[d] = matrix.rows();
    gsl::at(matrices_, d) = Kokkos::View<double**, Kokkos::LayoutRight>{
        "DeviceApplyMatrices matrix", matrix.rows(), matrix.columns()};
    const auto host_matrix = Kokkos::create_mirror_view(gsl::at(matrices_, d));
    for (size_t i = 0; i < matrix.rows(); ++i) {
      for (size_t j = 0; j < matrix.columns(); ++j) {
        host_matrix(i, j) = matrix(i, j);
      }
    }
    Kokkos::deep_copy(gsl::at(matrices_, d), host_matrix);
  }
}

void DeviceApplyMatrices::apply(const Kokkos::DefaultExecutionSpace& exec,
                                const View& destination,
                                const ConstView& source) {
  ASSERT(source.extent(0) == source_extents_.product() and
             destination.extent(0) == destination_extents_.product() and
             (source_components_.extent(0) == 0
                  ? source.extent(1) == destination.extent(1)
                  : source_components_.extent(0) == destination.extent(1)) and
             source.extent(2) == destination.extent(2),
         "The source (" << source.extent(0) << " points, " << source.extent(1)
                        << " components, " << source.extent(2)
                        << " elements) and destination ("
                        << destination.extent(0) << " points, "
                        << destination.extent(1) << " components, "
                        << destination.extent(2)
                        << " elements) don't match the extents "
                        << source_extents_ << " and " << destination_extents_);
  std::array<size_t, 3> dimensions{};
  size_t number_of_dimensions = 0;
  for (size_t d = 0; d < 3; ++d) {
    if (not gsl::at(is_identity_, d)) {
      gsl::at(dimensions, number_of_dimensions++) = d;
    }
  }
  if (number_of_dimensions == 0) {
    ERROR("At least one matrix must not be the identity.");
  }
  const size_t number_of_components = destination.extent(1);
  const size_t number_of_elements = source.extent(2);
  // The intermediate results, after all but the last dimension
  Index<3> extents = source_extents_;
  for (size_t i = 0; i + 1 < number_of_dimensions; ++i) {
    extents[gsl::at(dimensions, i)] =
        destination_extents_[gsl::at(dimensions, i)];
    auto& buffer = gsl::at(buffers_, i);
    if (buffer.extent(0) != extents.product() or
        buffer.extent(1) < number_of_components or
        buffer.extent(2) < number_of_elements) {
      buffer = Kokkos::View<double***, Kokkos::LayoutLeft>{
          Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                             "DeviceApplyMatrices buffer"),
          extents.product(), number_of_components, number_of_elements};
    }
  }
  extents = source_extents_;
  for (size_t i = 0; i < number_of_dimensions; ++i) {
    const size_t d = gsl::at(dimensions, i);
    const bool first = i == 0;
    const bool last = i + 1 == number_of_dimensions;
    const auto buffer = [this, number_of_components,
                         number_of_elements](const size_t index) {
      return Kokkos::subview(gsl::at(buffers_, index), Kokkos::ALL(),
                             Kokkos::make_pair(size_t{0}, number_of_components),
                             Kokkos::make_pair(size_t{0}, number_of_elements));
    };
    if (first and last) {
      apply_in_dimension(exec, destination, source, gsl::at(matrices_, d), d,
                         extents, source_components_);
    } else if (first) {
      apply_in_dimension(exec, buffer(0), source, gsl::at(matrices_, d), d,
                         extents, source_components_);
    } else if (last) {
      apply_in_dimension(exec, destination, buffer(i - 1),
                         gsl::at(matrices_, d), d, extents, {});
    } else {
      apply_in_dimension(exec, buffer(i), buffer(i - 1), gsl::at(matrices_, d),
                         d, extents, {});
    }
    extents[d] = destination_extents_[d];
  }
}
#endif  // SPECTRE_KOKKOS
