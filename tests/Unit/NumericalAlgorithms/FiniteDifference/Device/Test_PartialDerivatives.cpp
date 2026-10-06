// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/Device.hpp"
#include "NumericalAlgorithms/FiniteDifference/Device/PartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/Device/VolumeView.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
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
// Large enough for all derivative orders
constexpr size_t number_of_cells = 10;

// Differentiate the data on the device. Returns the host mirrors of the
// derivatives.
template <size_t Dim>
std::array<VolumeView::host_mirror_type, Dim> differentiate_on_device(
    const std::vector<helpers::ElementData<Dim>>& data,
    const Index<Dim>& volume_extents, const size_t ghost_width,
    const size_t fd_order) {
  const auto padded =
      helpers::pack(data, number_of_variables, volume_extents, ghost_width);
  std::array<VolumeView, Dim> derivatives{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(derivatives, d) = ::fd::device::make_volume_view(
        "derivative " + std::to_string(d), number_of_elements,
        number_of_variables, volume_extents);
  }
  ::fd::device::logical_partial_derivatives(derivatives, padded, volume_extents,
                                            fd_order);
  return helpers::to_host(derivatives);
}

// Compare to `fd::logical_partial_derivatives` with random data
template <size_t Dim>
void test_compare_to_host(const gsl::not_null<std::mt19937*> generator,
                          const size_t fd_order) {
  const Index<Dim> volume_extents(number_of_cells);
  const size_t ghost_width = fd_order / 2;
  const auto data = helpers::make_random_element_data(
      generator, number_of_elements, number_of_variables, volume_extents,
      ghost_width);
  const auto derivatives =
      differentiate_on_device(data, volume_extents, ghost_width, fd_order);

  const Mesh<Dim> mesh(number_of_cells, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered);
  for (size_t element = 0; element < number_of_elements; ++element) {
    const auto& volume_vars = data[element].volume_vars;
    std::array<DataVector, Dim> expected{};
    std::array<gsl::span<double>, Dim> expected_spans{};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(expected, d) = DataVector(volume_vars.size());
      gsl::at(expected_spans, d) =
          gsl::make_span(gsl::at(expected, d).data(), volume_vars.size());
    }
    ::fd::logical_partial_derivatives(
        make_not_null(&expected_spans),
        gsl::make_span(volume_vars.data(), volume_vars.size()),
        helpers::make_spans(data[element].ghost_data), mesh,
        number_of_variables, fd_order);
    for (size_t d = 0; d < Dim; ++d) {
      CAPTURE(element);
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(
          helpers::element_data(gsl::at(derivatives, d), element),
          gsl::at(expected, d));
    }
  }
}

// A stencil of order `fd_order` is exact for polynomials of degree
// `fd_order`. The neighbors send more ghost cells than needed, which checks
// that the ghost zones are filled with the cells adjacent to the element.
template <size_t Dim>
void test_polynomial(const size_t fd_order) {
  const Index<Dim> volume_extents(number_of_cells);
  const size_t ghost_width = fd_order / 2;
  const double delta = 2.0 / static_cast<double>(number_of_cells);
  // The logical coordinate of the cell, scaled to keep the polynomials of
  // order 1
  const auto scaled_coord = [delta](const int cell) {
    return 0.5 * (-1.0 + (cell + 0.5) * delta);
  };
  const auto coefficient = [](const size_t element, const size_t var,
                              const size_t d) {
    return 1.0 + static_cast<double>(d + element) -
           0.5 * static_cast<double>(var);
  };
  const auto data = helpers::make_element_data(
      number_of_elements, number_of_variables, volume_extents, ghost_width + 1,
      [&](const size_t element, const size_t var,
          const std::array<int, Dim>& cell) {
        auto result = static_cast<double>(var);
        for (size_t d = 0; d < Dim; ++d) {
          result += coefficient(element, var, d) *
                    pow(scaled_coord(gsl::at(cell, d)), fd_order);
        }
        return result;
      });
  const auto derivatives =
      differentiate_on_device(data, volume_extents, ghost_width, fd_order);

  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t element = 0; element < number_of_elements; ++element) {
      const DataVector expected = helpers::make_data(
          volume_extents, number_of_variables, std::array<int, Dim>{},
          [&](const size_t var, const std::array<int, Dim>& cell) {
            return coefficient(element, var, d) * 0.5 *
                   static_cast<double>(fd_order) *
                   pow(scaled_coord(gsl::at(cell, d)), fd_order - 1);
          });
      CAPTURE(element);
      CAPTURE(d);
      CHECK_ITERABLE_CUSTOM_APPROX(
          helpers::element_data(gsl::at(derivatives, d), element), expected,
          custom_approx);
    }
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

SPECTRE_TEST_CASE("Unit.FiniteDifference.Device.PartialDerivatives",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
