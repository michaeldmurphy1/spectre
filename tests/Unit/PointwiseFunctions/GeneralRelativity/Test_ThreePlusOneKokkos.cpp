// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ThreePlusOneKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// The device helpers give the values of the host functions, and the 3+1
// decomposition of a spacetime metric built from a lapse, shift and spatial
// metric gives them back
void test_against_host(const gsl::not_null<std::mt19937*> generator) {
  const double used_for_size = 0.0;
  const auto lapse = TestHelpers::gr::random_lapse(generator, used_for_size);
  const auto shift = TestHelpers::gr::random_shift<3>(generator, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size);
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);

  // The helpers have the operations of the host functions in the same order,
  // but compilers may contract multiplications and additions into fused
  // multiply-adds differently in the inlined helpers and in the host functions
  // (e.g. GCC with its default -ffp-contract=fast), so they agree to roundoff.
  const auto device_spatial_metric =
      gr::device::spatial_metric(spacetime_metric);
  CHECK(device_spatial_metric == gr::spatial_metric(spacetime_metric));
  CHECK_ITERABLE_APPROX(device_spatial_metric, spatial_metric);

  const auto [det, inverse] =
      gr::device::determinant_and_inverse(device_spatial_metric);
  const auto [host_det, host_inverse] =
      determinant_and_inverse(device_spatial_metric);
  CHECK(det == approx(get(host_det)));
  CHECK_ITERABLE_APPROX(inverse, host_inverse);

  const auto device_shift = gr::device::shift(spacetime_metric, inverse);
  CHECK_ITERABLE_APPROX(device_shift, gr::shift(spacetime_metric, inverse));
  // Recovered from the spacetime metric through the inverse spatial metric,
  // so with the roundoff of the inversion
  Approx inversion_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(device_shift, shift, inversion_approx);
  const double device_lapse = gr::device::lapse(spacetime_metric, device_shift);
  CHECK(device_lapse == approx(get(gr::lapse(device_shift, spacetime_metric))));
  CHECK(device_lapse == inversion_approx(get(lapse)));

  CHECK_ITERABLE_APPROX(
      gr::device::inverse_spacetime_metric(device_lapse, device_shift, inverse),
      gr::inverse_spacetime_metric(Scalar<double>{device_lapse}, device_shift,
                                   inverse));
  CHECK_ITERABLE_APPROX(
      gr::device::spacetime_normal_vector(device_lapse, device_shift),
      gr::spacetime_normal_vector(Scalar<double>{device_lapse}, device_shift));

  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    const auto local_spatial_metric =
        gr::device::spatial_metric(spacetime_metric);
    const auto [local_det, local_inverse] =
        gr::device::determinant_and_inverse(local_spatial_metric);
    const auto local_shift = gr::device::shift(spacetime_metric, local_inverse);
    const double local_lapse = gr::device::lapse(spacetime_metric, local_shift);
    return TestHelpers::flatten(
        local_spatial_metric, Scalar<double>{local_det}, local_inverse,
        local_shift, Scalar<double>{local_lapse},
        gr::device::inverse_spacetime_metric(local_lapse, local_shift,
                                             local_inverse),
        gr::device::spacetime_normal_vector(local_lapse, local_shift));
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.ThreePlusOne",
                  "[Unit][PointwiseFunctions]") {
  MAKE_GENERATOR(generator);
  for (size_t i = 0; i < 10; ++i) {
    test_against_host(make_not_null(&generator));
  }
}
