// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/EagerMath/OuterProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
void test_on_device() {
  tnsr::i<double, 2> lhs{};
  get<0>(lhs) = 1.;
  get<1>(lhs) = 2.;
  tnsr::Ij<double, 2> rhs{};
  get<0, 0>(rhs) = 1.;
  get<0, 1>(rhs) = 2.;
  get<1, 0>(rhs) = 3.;
  get<1, 1>(rhs) = 4.;
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    tnsr::iJk<double, 2> result{};
    outer_product(make_not_null(&result), lhs, rhs);
    const tnsr::iJk<double, 2> returned_result = outer_product(lhs, rhs);
    std::array<double, 16> values{};
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        for (size_t k = 0; k < 2; ++k) {
          gsl::at(values, 4 * i + 2 * j + k) = result.get(i, j, k);
          gsl::at(values, 8 + 4 * i + 2 * j + k) = returned_result.get(i, j, k);
        }
      }
    }
    return values;
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.OuterProduct.Kokkos",
                  "[DataStructures][Unit]") {
  test_on_device();
}
