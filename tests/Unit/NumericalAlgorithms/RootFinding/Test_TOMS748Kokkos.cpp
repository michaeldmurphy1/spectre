// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
constexpr size_t number_of_roots = 20;
constexpr double abs_tol = 1.e-14;
constexpr double rel_tol = 1.e-14;

// The scalar version is also used in Kokkos kernels, where it must give the
// same roots as on the host
void test_on_device() {
  const auto find_roots = KOKKOS_LAMBDA() {
    std::array<double, number_of_roots> roots{};
    for (size_t i = 0; i < number_of_roots; ++i) {
      const double c = 0.1 + static_cast<double>(i);
      gsl::at(roots, i) =
          RootFinder::toms748([c](const double x) { return x * x * x - c; },
                              0.0, c + 1.0, abs_tol, rel_tol);
    }
    return roots;
  };
  const Approx custom_approx = Approx::custom().epsilon(1.e-13).scale(1.0);
  TestHelpers::check_on_device(find_roots, custom_approx);
  const auto roots = find_roots();
  for (size_t i = 0; i < number_of_roots; ++i) {
    const double c = 0.1 + static_cast<double>(i);
    CAPTURE(c);
    CHECK(gsl::at(roots, i) == custom_approx(cbrt(c)));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748.Kokkos",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  test_on_device();
}
