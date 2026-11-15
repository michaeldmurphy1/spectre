// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ApplyMatricesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
Matrix random_matrix(const gsl::not_null<std::mt19937*> generator,
                     const size_t rows, const size_t columns) {
  std::uniform_real_distribution<double> distribution{-1.0, 1.0};
  Matrix matrix(rows, columns);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < columns; ++j) {
      matrix(i, j) = distribution(*generator);
    }
  }
  return matrix;
}

// `rows[d] == 0` is the identity in dimension `d`. With `select` only the
// source components 2 and 0 are projected, in that order.
void check(const gsl::not_null<std::mt19937*> generator,
           const Index<3>& source_extents, const std::array<size_t, 3>& rows,
           const bool select = false) {
  CAPTURE(source_extents);
  CAPTURE(rows);
  std::array<Matrix, 3> matrices{};
  Index<3> destination_extents = source_extents;
  for (size_t d = 0; d < 3; ++d) {
    if (gsl::at(rows, d) != 0) {
      gsl::at(matrices, d) =
          random_matrix(generator, gsl::at(rows, d), source_extents[d]);
      destination_extents[d] = gsl::at(rows, d);
    }
  }
  constexpr size_t number_of_components = 3;
  constexpr size_t number_of_elements = 3;
  const std::vector<size_t> source_components =
      select ? std::vector<size_t>{2, 0} : std::vector<size_t>{};
  const size_t number_of_destination_components =
      select ? source_components.size() : number_of_components;
  const size_t source_points = source_extents.product();
  const size_t destination_points = destination_extents.product();
  std::uniform_real_distribution<double> distribution{-1.0, 1.0};
  const Kokkos::View<double***, Kokkos::LayoutLeft> source{
      "source", source_points, number_of_components, number_of_elements};
  const auto host_source = Kokkos::create_mirror_view(source);
  for (size_t e = 0; e < number_of_elements; ++e) {
    for (size_t c = 0; c < number_of_components; ++c) {
      for (size_t p = 0; p < source_points; ++p) {
        host_source(p, c, e) = distribution(*generator);
      }
    }
  }
  Kokkos::deep_copy(source, host_source);
  const Kokkos::View<double***, Kokkos::LayoutLeft> destination{
      "destination", destination_points, number_of_destination_components,
      number_of_elements};

  DeviceApplyMatrices apply{matrices, source_extents, source_components};
  CHECK(apply.destination_extents() == destination_extents);
  const Kokkos::DefaultExecutionSpace exec{};
  // Twice, to check that the buffers are reused
  for (size_t repeat = 0; repeat < 2; ++repeat) {
    apply.apply(exec, destination, source);
  }
  const auto host_destination =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, destination);

  for (size_t e = 0; e < number_of_elements; ++e) {
    for (size_t c = 0; c < number_of_destination_components; ++c) {
      const size_t source_component = select ? source_components[c] : c;
      DataVector u{source_points};
      for (size_t p = 0; p < source_points; ++p) {
        u[p] = host_source(p, source_component, e);
      }
      const DataVector expected = apply_matrices(matrices, u, source_extents);
      DataVector result{destination_points};
      for (size_t p = 0; p < destination_points; ++p) {
        result[p] = host_destination(p, c, e);
      }
      CHECK_ITERABLE_APPROX(result, expected);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.ApplyMatrices.Kokkos",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  check(make_not_null(&generator), Index<3>{3, 4, 5}, {{6, 7, 8}});
  check(make_not_null(&generator), Index<3>{3, 4, 5}, {{0, 7, 2}});
  check(make_not_null(&generator), Index<3>{3, 4, 5}, {{0, 0, 2}});
  check(make_not_null(&generator), Index<3>{3, 4, 5}, {{5, 0, 0}});
  check(make_not_null(&generator), Index<3>{9, 9, 9}, {{18, 17, 17}});
  check(make_not_null(&generator), Index<3>{9, 9, 9}, {{17, 18, 17}}, true);
  check(make_not_null(&generator), Index<3>{3, 4, 5}, {{0, 6, 0}}, true);
}
