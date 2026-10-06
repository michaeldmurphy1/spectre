// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeStencils.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
constexpr size_t center = 15;
constexpr double grid_spacing = 0.1;

// The derivative of `f(x) = x^2 + 3 x` at the `center` of the `values`, where
// consecutive grid points are `stride` apart. The stencils are exact for this
// quadratic, so the result is `2 x + 3` for all orders.
template <size_t Order, bool UnitStride>
KOKKOS_FUNCTION double derivative(
    const std::array<double, 2 * center + 1>& values, const int stride) {
  using Stencil = fd::detail::DerivativeStencil<Order, UnitStride>;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return Stencil::pointwise(values.data() + center, stride,
                            Stencil::derivative_weights(1.0 / grid_spacing));
}

// Values of `f(x) = x^2 + 3 x` at `x = x_center + grid_spacing * k` at index
// `center + stride * k`. The other entries are set to a large value, so that
// using them would give a wrong derivative.
std::array<double, 2 * center + 1> make_values(const double x_center,
                                               const int stride) {
  std::array<double, 2 * center + 1> values{};
  values.fill(1.0e10);
  const int max_k = static_cast<int>(center) / stride;
  for (int k = -max_k; k <= max_k; ++k) {
    const double x = x_center + grid_spacing * k;
    gsl::at(values, static_cast<size_t>(static_cast<int>(center) +
                                        stride * k)) = x * x + 3.0 * x;
  }
  return values;
}

void test_derivative_stencils() {
  const double x_center = 0.4;
  const auto unit_stride_values = make_values(x_center, 1);
  const auto strided_values = make_values(x_center, 3);
  const auto derivatives = KOKKOS_LAMBDA() {
    return std::array{derivative<2, true>(unit_stride_values, 1),
                      derivative<4, true>(unit_stride_values, 1),
                      derivative<6, true>(unit_stride_values, 1),
                      derivative<8, true>(unit_stride_values, 1),
                      derivative<10, true>(unit_stride_values, 1),
                      derivative<2, false>(strided_values, 3),
                      derivative<4, false>(strided_values, 3),
                      derivative<6, false>(strided_values, 3),
                      derivative<8, false>(strided_values, 3),
                      derivative<10, false>(strided_values, 3)};
  };
  TestHelpers::check_on_device(derivatives);
  for (const double result : derivatives()) {
    CHECK(result == approx(2.0 * x_center + 3.0));
  }
}

// The monotonised central reconstruction in each of its branches: an
// extremum, a steep left and right side, and a smooth profile
void test_monotonised_central() {
  const std::array<std::array<double, 3>, 4> profiles{{{{1.0, 2.0, 1.0}},
                                                       {{0.0, 3.0, 3.5}},
                                                       {{0.0, 0.5, 3.5}},
                                                       {{1.0, 2.0, 3.5}}}};
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    std::array<double, 8> results{};
    for (size_t i = 0; i < profiles.size(); ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      const double* const center_value = gsl::at(profiles, i).data() + 1;
      const auto faces = fd::reconstruction::detail::
          MonotonisedCentralReconstructor::pointwise(center_value, 1);
      gsl::at(results, 2 * i) = faces[0];
      gsl::at(results, 2 * i + 1) = faces[1];
    }
    return results;
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.DerivativeStencils.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  test_derivative_stencils();
  test_monotonised_central();
}
