// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.tpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivativesKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

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
  const Kokkos::DefaultExecutionSpace exec{};
  const auto padded = helpers::pack(exec, data, number_of_variables,
                                    volume_extents, ghost_width);
  std::array<VolumeView, Dim> derivatives{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(derivatives, d) = ::fd::device::make_volume_view(
        exec, "derivative " + std::to_string(d), number_of_elements,
        number_of_variables, volume_extents);
  }
  ::fd::device::logical_partial_derivatives(exec, derivatives, padded,
                                            volume_extents, fd_order);
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

// Compare to `fd::partial_derivatives` with random data and a random inverse
// Jacobian. The variables are a subview of a padded view with an extra
// variable and the derivatives a subview of a view with an extra variable, so
// neither is contiguous.
template <size_t Dim>
void test_inertial_compare_to_host(const gsl::not_null<std::mt19937*> generator,
                                   const size_t fd_order) {
  using derivative_tags =
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, Dim, Frame::Inertial>>;
  constexpr size_t number_of_inertial_variables = 1 + Dim;
  const Index<Dim> volume_extents(number_of_cells);
  const size_t number_of_points = volume_extents.product();
  const size_t ghost_width = fd_order / 2;
  const auto data = helpers::make_random_element_data(
      generator, number_of_elements, number_of_inertial_variables + 1,
      volume_extents, ghost_width);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  const Kokkos::DefaultExecutionSpace exec{};
  const auto padded =
      helpers::pack(exec, data, number_of_inertial_variables + 1,
                    volume_extents, ghost_width);
  // Indexed by grid point, component and element, like a `BatchedVariables`
  const Kokkos::View<double***, Kokkos::LayoutLeft> jacobian_storage{
      "inverse jacobian", number_of_points, Dim * Dim, number_of_elements};
  const auto host_jacobian_storage =
      Kokkos::create_mirror_view(Kokkos::HostSpace{}, jacobian_storage);
  ::fd::device::BatchInverseJacobian<Dim> inverse_jacobian{};
  for (size_t component = 0; component < Dim * Dim; ++component) {
    for (size_t element = 0; element < number_of_elements; ++element) {
      for (size_t point = 0; point < number_of_points; ++point) {
        host_jacobian_storage(point, component, element) = dist(*generator);
      }
    }
    inverse_jacobian[component] = Kokkos::subview(
        jacobian_storage, Kokkos::ALL(), component, Kokkos::ALL());
  }
  Kokkos::deep_copy(exec, jacobian_storage, host_jacobian_storage);
  const auto derivatives = ::fd::device::make_volume_view(
      exec, "derivatives", number_of_elements,
      Dim * number_of_inertial_variables + 1, volume_extents);
  const auto derivative_variables =
      Kokkos::make_pair(1_st, Dim * number_of_inertial_variables + 1);
  ::fd::device::partial_derivatives(
      exec,
      Kokkos::subview(derivatives, Kokkos::ALL(), derivative_variables,
                      Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
      Kokkos::subview(padded, Kokkos::ALL(),
                      Kokkos::make_pair(1_st, number_of_inertial_variables + 1),
                      Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
      inverse_jacobian, volume_extents, fd_order);
  const auto host_derivatives =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, derivatives);

  const Mesh<Dim> mesh(number_of_cells, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered);
  for (size_t element = 0; element < number_of_elements; ++element) {
    CAPTURE(element);
    const auto& volume_vars = data[element].volume_vars;
    // Without the first variable
    const auto variables_span =
        gsl::make_span(volume_vars.data(), volume_vars.size())
            .subspan(number_of_points,
                     number_of_inertial_variables * number_of_points);
    DirectionMap<Dim, gsl::span<const double>> ghost_spans{};
    for (const auto& [direction, ghost_data] : data[element].ghost_data) {
      const size_t ghost_size = ghost_data.size() /
                                (number_of_inertial_variables + 1) *
                                number_of_inertial_variables;
      ghost_spans[direction] =
          gsl::make_span(ghost_data.data(), ghost_data.size())
              .subspan(ghost_data.size() - ghost_size, ghost_size);
    }
    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
        host_inverse_jacobian{number_of_points};
    for (size_t component = 0; component < Dim * Dim; ++component) {
      for (size_t point = 0; point < number_of_points; ++point) {
        host_inverse_jacobian[component][point] =
            host_jacobian_storage(point, component, element);
      }
    }
    Variables<db::wrap_tags_in<::Tags::deriv, derivative_tags,
                               tmpl::size_t<Dim>, Frame::Inertial>>
        expected{number_of_points};
    ::fd::partial_derivatives<derivative_tags>(
        make_not_null(&expected), variables_span, ghost_spans, mesh,
        number_of_inertial_variables, fd_order, host_inverse_jacobian);
    const DataVector result{
        &host_derivatives(element, 1, 0, 0, 0),
        Dim * number_of_inertial_variables * number_of_points};
    CHECK_ITERABLE_APPROX(result,
                          (DataVector{expected.data(), expected.size()}));
  }
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator) {
  for (const size_t fd_order : {2_st, 4_st, 6_st, 8_st, 10_st}) {
    CAPTURE(Dim);
    CAPTURE(fd_order);
    test_compare_to_host<Dim>(generator, fd_order);
    test_polynomial<Dim>(fd_order);
    test_inertial_compare_to_host<Dim>(generator, fd_order);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.PartialDerivatives.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
