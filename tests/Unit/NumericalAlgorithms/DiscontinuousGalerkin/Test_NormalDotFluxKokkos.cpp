// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
void test_on_device() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto normal = make_with_random_values<tnsr::i<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto flux = make_with_random_values<tnsr::Ij<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto vector = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    tnsr::i<double, 3> normal_dot_flux_result{};
    normal_dot_flux(make_not_null(&normal_dot_flux_result), normal, flux);
    tnsr::iJ<double, 3> normal_times_flux_result{};
    normal_times_flux(make_not_null(&normal_times_flux_result), normal, vector);
    std::array<double, 12> result{};
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(result, j) = normal_dot_flux_result.get(j);
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(result, 3 + 3 * i + j) = normal_times_flux_result.get(i, j);
      }
    }
    return result;
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.NormalDotFlux.Kokkos",
                  "[Unit][Evolution]") {
  test_on_device();
}
