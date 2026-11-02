// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace helpers = TestHelpers::fd::device;

Kokkos::View<double*> to_device(const DataVector& host) {
  Kokkos::View<double*> result("data", host.size());
  Kokkos::deep_copy(
      result,
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          host.data(), host.size()));
  return result;
}

DataVector to_host(const Kokkos::View<const double*>& device) {
  DataVector result(device.extent(0));
  Kokkos::deep_copy(
      Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          result.data(), result.size()),
      device);
  return result;
}

// Copies and matrices between buffers with offsets and variable strides
void test_transfers(const gsl::not_null<std::mt19937*> generator) {
  constexpr size_t number_of_variables = 3;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  // Two "elements" of 7 points each in the source, with a gap between
  // variables
  const size_t source_stride = 8;
  const size_t source_size = 2 * number_of_variables * source_stride;
  const auto source = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), DataVector(source_size));
  const size_t destination_stride = 5;
  const size_t destination_size = 2 * number_of_variables * destination_stride;
  DataVector expected(destination_size, -10.0);

  fd::device::Transfers transfers{};
  // Copy and reverse three points of the first element into the first part
  const std::vector<size_t> source_points{4, 0, 2};
  const std::vector<size_t> destination_points{2, 1, 0};
  transfers.add_copy(0, source_stride, 0, destination_stride, source_points,
                     destination_points);
  for (size_t var = 0; var < number_of_variables; ++var) {
    for (size_t i = 0; i < source_points.size(); ++i) {
      expected[var * destination_stride + destination_points[i]] =
          source[var * source_stride + source_points[i]];
    }
  }
  // Interpolate the second element to two points of the second part
  Matrix matrix(2, 7);
  for (size_t row = 0; row < matrix.rows(); ++row) {
    for (size_t column = 0; column < matrix.columns(); ++column) {
      matrix(row, column) = dist(*generator);
    }
  }
  const size_t source_offset = number_of_variables * source_stride;
  const size_t destination_offset = number_of_variables * destination_stride;
  const std::vector<size_t> matrix_destination_points{3, 1};
  transfers.add_matrix(source_offset, source_stride, destination_offset,
                       destination_stride, matrix, matrix_destination_points);
  for (size_t var = 0; var < number_of_variables; ++var) {
    for (size_t row = 0; row < matrix.rows(); ++row) {
      double value = 0.0;
      for (size_t column = 0; column < matrix.columns(); ++column) {
        value += matrix(row, column) *
                 source[source_offset + var * source_stride + column];
      }
      expected[destination_offset + var * destination_stride +
               matrix_destination_points[row]] = value;
    }
  }
  CHECK(transfers.number_of_points() == 5);

  const Kokkos::DefaultExecutionSpace exec{};
  transfers.copy_to_device(exec);
  const auto destination = to_device(DataVector(destination_size, -10.0));
  transfers.apply(exec, destination, to_device(source), number_of_variables);
  CHECK_ITERABLE_APPROX(to_host(destination), expected);
}

// Slicing with `slab_points` and interpolation gives the same data as
// `evolution::dg::subcell::slice_data`
template <size_t Dim>
void test_slicing(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Dim);
  constexpr size_t number_of_variables = 2;
  constexpr size_t ghost_width = 2;
  const Index<Dim> extents(5);
  const size_t number_of_points = extents.product();
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto volume = make_with_random_values<DataVector>(
      generator, make_not_null(&dist),
      DataVector(number_of_variables * number_of_points));
  std::unordered_set<Direction<Dim>> directions{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    directions.insert(direction);
  }

  // Interpolate in the first direction, copy in the others
  const Direction<Dim> interpolated_direction = Direction<Dim>::lower_xi();
  const Mesh<Dim> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  const size_t slice_points = ghost_width * extents.slice_away(0).product();
  tnsr::I<DataVector, Dim, Frame::ElementLogical> target_points{slice_points};
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t i = 0; i < slice_points; ++i) {
      target_points.get(d)[i] = 0.9 * dist(*generator);
    }
  }
  const intrp::Irregular<Dim> interpolant{mesh, target_points};
  DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>> interpolants{};
  interpolants[DirectionalId<Dim>{interpolated_direction, ElementId<Dim>{1}}] =
      interpolant;
  const auto expected = evolution::dg::subcell::slice_data(
      volume, extents, ghost_width, directions, 0, interpolants);

  // All directions in one buffer
  fd::device::Transfers transfers{};
  size_t offset = 0;
  DirectionMap<Dim, size_t> offsets{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    offsets[direction] = offset;
    std::vector<size_t> destination_points(slice_points);
    std::iota(destination_points.begin(), destination_points.end(), 0_st);
    if (direction == interpolated_direction) {
      transfers.add_matrix(0, number_of_points, offset, slice_points,
                           interpolant.matrix(), destination_points);
    } else {
      const size_t dim = direction.dimension();
      transfers.add_copy(
          0, number_of_points, offset, slice_points,
          direction.side() == Side::Lower
              ? fd::device::slab_points(extents, dim, 0, ghost_width)
              : fd::device::slab_points(
                    extents, dim, extents[dim] - ghost_width, extents[dim]),
          destination_points);
    }
    offset += number_of_variables * slice_points;
  }
  const Kokkos::DefaultExecutionSpace exec{};
  transfers.copy_to_device(exec);
  const Kokkos::View<double*> sliced("sliced", offset);
  transfers.apply(exec, sliced, to_device(volume), number_of_variables);
  const DataVector host_sliced = to_host(sliced);
  for (const auto& direction : Direction<Dim>::all_directions()) {
    CAPTURE(direction);
    DataVector result(number_of_variables * slice_points);
    std::copy(
        std::next(host_sliced.begin(),
                  static_cast<std::ptrdiff_t>(offsets[direction])),
        std::next(host_sliced.begin(), static_cast<std::ptrdiff_t>(
                                           offsets[direction] + result.size())),
        result.begin());
    CHECK_ITERABLE_APPROX(result, expected.at(direction));
  }
}

// Filling a padded view with `padded_interior_points` and
// `padded_ghost_zone_points` gives the same data as
// `fd::device::pack_with_ghost_zones`
template <size_t Dim>
void test_padding(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Dim);
  constexpr size_t number_of_variables = 2;
  constexpr size_t ghost_width = 2;
  const Index<Dim> extents(4);
  // The neighbors send one more ghost cell than needed
  const auto data = helpers::make_random_element_data(
      generator, 1, number_of_variables, extents, ghost_width + 1);
  const Kokkos::DefaultExecutionSpace exec{};
  const auto expected = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{},
      helpers::pack(exec, data, number_of_variables, extents, ghost_width));

  // Source buffer: the volume data followed by the ghost data of each
  // direction
  DataVector source(data[0].volume_vars.size());
  std::copy(data[0].volume_vars.begin(), data[0].volume_vars.end(),
            source.begin());
  fd::device::Transfers transfers{};
  const auto padded = fd::device::make_volume_view(
      exec, "padded", 1, number_of_variables, extents, ghost_width);
  const size_t padded_points = padded.size() / number_of_variables;
  transfers.add_copy(0, extents.product(), 0, padded_points,
                     fd::device::slab_points(extents, 0, 0, extents[0]),
                     fd::device::padded_interior_points(extents, ghost_width));
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const size_t dim = direction.dimension();
    const auto& ghost_data = data[0].ghost_data.at(direction);
    Index<Dim> ghost_extents = extents;
    ghost_extents[dim] = ghost_width + 1;
    const size_t source_offset = source.size();
    DataVector new_source(source.size() + ghost_data.size());
    std::ranges::copy(source, new_source.begin());
    std::ranges::copy(ghost_data,
                      std::next(new_source.begin(),
                                static_cast<std::ptrdiff_t>(source_offset)));
    source = std::move(new_source);
    // The ghost cells closest to the element
    const bool upper_side = direction.side() == Side::Upper;
    transfers.add_copy(
        source_offset, ghost_extents.product(), 0, padded_points,
        upper_side
            ? fd::device::slab_points(ghost_extents, dim, 0, ghost_width)
            : fd::device::slab_points(ghost_extents, dim, 1, ghost_width + 1),
        fd::device::padded_ghost_zone_points(extents, ghost_width, dim,
                                             upper_side));
  }
  transfers.copy_to_device(exec);
  transfers.apply(exec, Kokkos::View<double*>(padded.data(), padded.size()),
                  to_device(source), number_of_variables);
  const auto result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, padded);
  // Compare everything but the unused edges and corners
  const auto interior =
      fd::device::padded_interior_points(extents, ghost_width);
  std::vector<size_t> compared = interior;
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto zone = fd::device::padded_ghost_zone_points(
        extents, ghost_width, direction.dimension(),
        direction.side() == Side::Upper);
    compared.insert(compared.end(), zone.begin(), zone.end());
  }
  using FlatView =
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  const FlatView flat_result(result.data(), result.size());
  const FlatView flat_expected(expected.data(), expected.size());
  for (size_t var = 0; var < number_of_variables; ++var) {
    for (const size_t point : compared) {
      CAPTURE(var);
      CAPTURE(point);
      CHECK(flat_result(var * padded_points + point) ==
            flat_expected(var * padded_points + point));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.Transfers.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test_transfers(make_not_null(&generator));
  test_slicing<1>(make_not_null(&generator));
  test_slicing<2>(make_not_null(&generator));
  test_slicing<3>(make_not_null(&generator));
  test_padding<1>(make_not_null(&generator));
  test_padding<2>(make_not_null(&generator));
  test_padding<3>(make_not_null(&generator));
}
