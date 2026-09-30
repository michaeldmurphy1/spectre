// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// Calls every function in `Christoffel.hpp` in a kernel
void test_on_device() {
  // Arbitrary values, with a symmetric inverse metric by construction
  tnsr::abb<double, 3, Frame::Inertial> d_metric{};
  for (size_t i = 0; i < d_metric.size(); ++i) {
    d_metric[i] = 0.1 * static_cast<double>(i) - 1.3;
  }
  tnsr::AA<double, 3, Frame::Inertial> inverse_metric{};
  for (size_t i = 0; i < inverse_metric.size(); ++i) {
    inverse_metric[i] = 0.2 * static_cast<double>(i) + 0.5;
  }
  tnsr::ijj<double, 3, Frame::Inertial> d_spatial_metric{};
  for (size_t i = 0; i < d_spatial_metric.size(); ++i) {
    d_spatial_metric[i] = 0.3 * static_cast<double>(i) - 2.0;
  }
  tnsr::II<double, 3, Frame::Inertial> inverse_spatial_metric{};
  for (size_t i = 0; i < inverse_spatial_metric.size(); ++i) {
    inverse_spatial_metric[i] = 0.4 * static_cast<double>(i) + 1.0;
  }
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    constexpr size_t number_of_components = 40;
    std::array<double, 4 * number_of_components + 3> values{};
    tnsr::abb<double, 3, Frame::Inertial> first_kind{};
    gr::christoffel_first_kind(make_not_null(&first_kind), d_metric);
    const auto returned_first_kind = gr::christoffel_first_kind(d_metric);
    tnsr::Abb<double, 3, Frame::Inertial> second_kind{};
    gr::christoffel_second_kind(make_not_null(&second_kind), d_metric,
                                inverse_metric);
    const auto returned_second_kind =
        gr::christoffel_second_kind(d_metric, inverse_metric);
    for (size_t i = 0; i < number_of_components; ++i) {
      gsl::at(values, i) = first_kind[i];
      gsl::at(values, number_of_components + i) = returned_first_kind[i];
      gsl::at(values, 2 * number_of_components + i) = second_kind[i];
      gsl::at(values, 3 * number_of_components + i) = returned_second_kind[i];
    }
    tnsr::i<double, 3, Frame::Inertial> trace{};
    gr::trace_spatial_christoffel_first_kind(
        make_not_null(&trace), d_spatial_metric, inverse_spatial_metric);
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(values, 4 * number_of_components + i) = trace.get(i);
    }
    return values;
  });
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.Christoffel.Kokkos",
    "[PointwiseFunctions][Unit]") {
  test_on_device();
}
