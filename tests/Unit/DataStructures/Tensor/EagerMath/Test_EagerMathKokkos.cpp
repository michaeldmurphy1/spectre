// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>
#include <type_traits>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
void test_dot_product(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto vector_a = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto vector_b = make_with_random_values<tnsr::I<double, 3>>(
      generator, make_not_null(&distribution));
  const auto one_form = make_with_random_values<tnsr::i<double, 3>>(
      generator, make_not_null(&distribution));
  const auto metric = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return TestHelpers::flatten(
        dot_product(vector_a, vector_b), dot_product(vector_a, one_form),
        dot_product(vector_a, vector_b, metric),
        // Passing the same vector twice takes a different code path
        dot_product(vector_a, vector_a, metric));
  });
}

void test_raise_or_lower_index(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto one_form = make_with_random_values<tnsr::i<double, 3>>(
      generator, make_not_null(&distribution));
  const auto tensor = make_with_random_values<tnsr::ijj<double, 3>>(
      generator, make_not_null(&distribution));
  const auto inverse_metric = make_with_random_values<tnsr::II<double, 3>>(
      generator, make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return TestHelpers::flatten(
        raise_or_lower_index(one_form, inverse_metric),
        raise_or_lower_first_index(tensor, inverse_metric));
  });
}

void test_trace(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto rank_three = make_with_random_values<tnsr::ijj<double, 3>>(
      generator, make_not_null(&distribution));
  const auto rank_two = make_with_random_values<tnsr::ii<double, 3>>(
      generator, make_not_null(&distribution));
  const auto inverse_metric = make_with_random_values<tnsr::II<double, 3>>(
      generator, make_not_null(&distribution));
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return TestHelpers::flatten(trace_last_indices(rank_three, inverse_metric),
                                trace(rank_two, inverse_metric));
  });
}
template <typename TensorType>
void test_determinant_and_inverse(
    const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  auto tensor = make_with_random_values<TensorType>(
      generator, make_not_null(&distribution));
  // Diagonally dominant, so that it is far from singular
  for (size_t i = 0; i < TensorType::index_dim(0); ++i) {
    tensor.get(i, i) += 4.0;
  }
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    const auto [det, inv] = determinant_and_inverse(tensor);
    Scalar<double> det_not_null{};
    std::remove_cv_t<decltype(inv)> inv_not_null{};
    determinant_and_inverse(make_not_null(&det_not_null),
                            make_not_null(&inv_not_null), tensor);
    return TestHelpers::flatten(det, inv, det_not_null, inv_not_null);
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.Kokkos",
                  "[Unit][DataStructures]") {
  MAKE_GENERATOR(generator);
  test_dot_product(make_not_null(&generator));
  test_raise_or_lower_index(make_not_null(&generator));
  test_trace(make_not_null(&generator));
  test_determinant_and_inverse<tnsr::ij<double, 1>>(make_not_null(&generator));
  test_determinant_and_inverse<tnsr::ij<double, 2>>(make_not_null(&generator));
  test_determinant_and_inverse<tnsr::ii<double, 2>>(make_not_null(&generator));
  test_determinant_and_inverse<tnsr::ij<double, 3>>(make_not_null(&generator));
  test_determinant_and_inverse<tnsr::ii<double, 3>>(make_not_null(&generator));
  test_determinant_and_inverse<tnsr::ab<double, 3>>(make_not_null(&generator));
  test_determinant_and_inverse<tnsr::aa<double, 3>>(make_not_null(&generator));
}
