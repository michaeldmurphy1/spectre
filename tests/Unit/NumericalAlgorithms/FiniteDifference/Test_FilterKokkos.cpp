// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/Filter.hpp"
#include "NumericalAlgorithms/FiniteDifference/FilterKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

namespace {
using ::fd::device::VolumeView;
namespace helpers = TestHelpers::fd::device;

constexpr size_t number_of_elements = 3;
constexpr size_t number_of_variables = 2;
// Large enough for all orders
constexpr size_t number_of_cells = 10;
constexpr double epsilon = 0.3;

// Filter the last `number_of_variables` variables of the data on the device,
// from and into subviews of views with an extra variable, so the views aren't
// contiguous. Returns the host mirror of the filtered variables.
template <size_t Dim>
VolumeView::host_mirror_type filter_on_device(
    const std::vector<helpers::ElementData<Dim>>& data,
    const Index<Dim>& volume_extents, const size_t ghost_width,
    const size_t fd_order) {
  const Kokkos::DefaultExecutionSpace exec{};
  const auto padded = helpers::pack(exec, data, number_of_variables + 1,
                                    volume_extents, ghost_width);
  const auto filtered =
      ::fd::device::make_volume_view(exec, "filtered", number_of_elements,
                                     number_of_variables + 1, volume_extents);
  const auto variables = Kokkos::make_pair(1_st, number_of_variables + 1);
  ::fd::device::kreiss_oliger_filter(
      exec,
      Kokkos::subview(filtered, Kokkos::ALL(), variables, Kokkos::ALL(),
                      Kokkos::ALL(), Kokkos::ALL()),
      Kokkos::subview(padded, Kokkos::ALL(), variables, Kokkos::ALL(),
                      Kokkos::ALL(), Kokkos::ALL()),
      volume_extents, fd_order, epsilon);
  const auto result = ::fd::device::make_volume_view(
      exec, "result", number_of_elements, number_of_variables, volume_extents);
  Kokkos::deep_copy(
      exec, result,
      Kokkos::subview(filtered, Kokkos::ALL(), variables, Kokkos::ALL(),
                      Kokkos::ALL(), Kokkos::ALL()));
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
}

// A copy of the last `number_of_variables` variables of the data
DataVector last_variables(const DataVector& data) {
  const size_t size =
      data.size() / (number_of_variables + 1) * number_of_variables;
  DataVector result(size);
  std::copy(data.end() - static_cast<std::ptrdiff_t>(size), data.end(),
            result.begin());
  return result;
}

// The volume and ghost data of the last `number_of_variables` variables
template <size_t Dim>
helpers::ElementData<Dim> last_variables(
    const helpers::ElementData<Dim>& data) {
  helpers::ElementData<Dim> result{};
  result.volume_vars = last_variables(data.volume_vars);
  for (const auto& [direction, ghost_data] : data.ghost_data) {
    result.ghost_data[direction] = last_variables(ghost_data);
  }
  return result;
}

// Compare to `fd::kreiss_oliger_filter` with random data
template <size_t Dim>
void test_compare_to_host(const gsl::not_null<std::mt19937*> generator,
                          const size_t fd_order) {
  const Index<Dim> volume_extents(number_of_cells);
  const size_t ghost_width = fd_order / 2;
  const auto data = helpers::make_random_element_data(
      generator, number_of_elements, number_of_variables + 1, volume_extents,
      ghost_width);
  const auto filtered =
      filter_on_device(data, volume_extents, ghost_width, fd_order);

  const Mesh<Dim> mesh(number_of_cells, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered);
  for (size_t element = 0; element < number_of_elements; ++element) {
    CAPTURE(element);
    const auto element_data = last_variables(data[element]);
    DataVector expected(element_data.volume_vars.size());
    auto expected_span = gsl::make_span(expected.data(), expected.size());
    ::fd::kreiss_oliger_filter(make_not_null(&expected_span),
                               gsl::make_span(element_data.volume_vars.data(),
                                              element_data.volume_vars.size()),
                               helpers::make_spans(element_data.ghost_data),
                               mesh, number_of_variables, fd_order, epsilon);
    CHECK_ITERABLE_APPROX(helpers::element_data(filtered, element), expected);
  }
}

// The dissipation of order `fd_order` vanishes for polynomials of lower
// degree, so the filter doesn't change them
template <size_t Dim>
void test_polynomial(const size_t fd_order) {
  const Index<Dim> volume_extents(number_of_cells);
  const size_t ghost_width = fd_order / 2;
  const double delta = 2.0 / static_cast<double>(number_of_cells);
  const auto coord = [delta](const int cell) {
    return 0.5 * (-1.0 + (cell + 0.5) * delta);
  };
  const auto value = [&coord, fd_order](const size_t element, const size_t var,
                                        const std::array<int, Dim>& cell) {
    auto result = static_cast<double>(var);
    for (size_t d = 0; d < Dim; ++d) {
      result += (1.0 + static_cast<double>(d + element)) *
                pow(coord(gsl::at(cell, d)), fd_order - 1);
    }
    return result;
  };
  const auto data =
      helpers::make_element_data(number_of_elements, number_of_variables + 1,
                                 volume_extents, ghost_width, value);
  const auto filtered =
      filter_on_device(data, volume_extents, ghost_width, fd_order);
  // The stencils cancel terms of size ~100
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t element = 0; element < number_of_elements; ++element) {
    CAPTURE(element);
    const DataVector expected = helpers::make_data(
        volume_extents, number_of_variables, std::array<int, Dim>{},
        [&](const size_t var, const std::array<int, Dim>& cell) {
          return value(element, var + 1, cell);
        });
    CHECK_ITERABLE_CUSTOM_APPROX(helpers::element_data(filtered, element),
                                 expected, custom_approx);
  }
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator) {
  for (const size_t fd_order : {2_st, 4_st, 6_st, 8_st, 10_st}) {
    CAPTURE(Dim);
    CAPTURE(fd_order);
    test_compare_to_host<Dim>(generator, fd_order);
    test_polynomial<Dim>(fd_order);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.Filter.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
