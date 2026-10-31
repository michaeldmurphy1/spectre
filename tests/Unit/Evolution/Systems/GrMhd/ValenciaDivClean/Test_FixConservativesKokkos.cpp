// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags = grmhd::ValenciaDivClean::Tags;
using conservative_tags = grmhd::ValenciaDivClean::FixConservativesOnDeviceTags;
using metric_tags = tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                               gr::Tags::InverseSpatialMetric<DataVector, 3>,
                               gr::Tags::SqrtDetSpatialMetric<DataVector>>;

// The variables of a batch of elements on the device
struct DeviceBatch {
  BatchedVariables<conservative_tags> conservatives;
  BatchedVariables<metric_tags> metric;
};

DeviceBatch to_device(const std::vector<Variables<conservative_tags>>& cons,
                      const std::vector<Variables<metric_tags>>& metric) {
  const size_t num_points = cons.front().number_of_grid_points();
  BatchedVariables<conservative_tags, Kokkos::HostSpace> host_cons{num_points,
                                                                   cons.size()};
  BatchedVariables<metric_tags, Kokkos::HostSpace> host_metric{num_points,
                                                               cons.size()};
  for (size_t e = 0; e < cons.size(); ++e) {
    Variables<conservative_tags> element_cons{};
    set_data_ref_to_element(make_not_null(&element_cons), host_cons, e);
    element_cons = cons[e];
    Variables<metric_tags> element_metric{};
    set_data_ref_to_element(make_not_null(&element_metric), host_metric, e);
    element_metric = metric[e];
  }
  DeviceBatch result{{num_points, cons.size()}, {num_points, cons.size()}};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&result.conservatives), host_cons);
  deep_copy(exec, make_not_null(&result.metric), host_metric);
  exec.fence();
  return result;
}

Variables<conservative_tags> element_to_host(
    const BatchedVariables<conservative_tags>& conservatives, const size_t e) {
  BatchedVariables<conservative_tags, Kokkos::HostSpace> host_cons{
      conservatives.number_of_grid_points(),
      conservatives.number_of_elements()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&host_cons), conservatives);
  exec.fence();
  Variables<conservative_tags> element_cons{};
  set_data_ref_to_element(make_not_null(&element_cons), host_cons, e);
  return element_cons;
}

PointwiseStatusSummary fix_on_device(
    const gsl::not_null<DeviceBatch*> batch,
    const grmhd::ValenciaDivClean::FixConservatives& variable_fixer) {
  return grmhd::ValenciaDivClean::fix_conservatives_on_device(
      Kokkos::DefaultExecutionSpace{}, make_not_null(&batch->conservatives),
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
          batch->metric),
      get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, 3>>>(
          batch->metric),
      get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
          batch->metric),
      variable_fixer);
}

// Random states, many of which need fixing, including the momentum density
Variables<conservative_tags> random_conservatives(
    const gsl::not_null<std::mt19937*> generator, const size_t num_points,
    const bool zero_magnetic_field) {
  std::uniform_real_distribution<> dist_d(1.e-12, 1.0);
  std::uniform_real_distribution<> dist_unit(0.0, 1.0);
  std::uniform_real_distribution<> dist_vector(-2.0, 2.0);
  Variables<conservative_tags> cons{num_points};
  get<Tags::TildeD>(cons) = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_d), DataVector(num_points));
  get<Tags::TildeYe>(cons) = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), DataVector(num_points));
  get(get<Tags::TildeYe>(cons)) *= 0.5 * get(get<Tags::TildeD>(cons));
  get<Tags::TildeTau>(cons) = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), DataVector(num_points));
  get(get<Tags::TildeTau>(cons)) = 2.0 * get(get<Tags::TildeTau>(cons)) - 0.2;
  get<Tags::TildeS<>>(cons) =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_vector), DataVector(num_points));
  get<Tags::TildeB<>>(cons) =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_vector), DataVector(num_points));
  if (zero_magnetic_field) {
    for (auto& component : get<Tags::TildeB<>>(cons)) {
      component = 0.0;
    }
  }
  get(get<Tags::TildePhi>(cons)) = 0.0;
  return cons;
}

Variables<metric_tags> random_metric(
    const gsl::not_null<std::mt19937*> generator, const size_t num_points) {
  std::uniform_real_distribution<> dist_metric(0.8, 1.2);
  Variables<metric_tags> metric{num_points, 0.0};
  get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(metric)) = 1.0;
  for (size_t d = 0; d < 3; ++d) {
    get<gr::Tags::SpatialMetric<DataVector, 3>>(metric).get(d, d) =
        make_with_random_values<DataVector>(
            generator, make_not_null(&dist_metric), DataVector(num_points));
    get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(metric).get(d, d) =
        1.0 / get<gr::Tags::SpatialMetric<DataVector, 3>>(metric).get(d, d);
    get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(metric)) *=
        sqrt(get<gr::Tags::SpatialMetric<DataVector, 3>>(metric).get(d, d));
  }
  return metric;
}

// The batched version fixes the variables of each element like the
// `DataVector` version. The magnetic field of the middle element is zero, so
// whether to assume a non-zero magnetic field must be decided for each element.
void test_random_batch(
    const gsl::not_null<std::mt19937*> generator,
    const hydro::MagneticFieldTreatment magnetic_field_treatment) {
  CAPTURE(magnetic_field_treatment);
  const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
      1.e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.e-4,
      1.e-3,  1.0e-1,  1.e-2,   true,   magnetic_field_treatment};
  const size_t num_points = 100;
  const size_t num_elements = 3;
  std::vector<Variables<conservative_tags>> cons{};
  std::vector<Variables<metric_tags>> metric{};
  for (size_t e = 0; e < num_elements; ++e) {
    cons.push_back(random_conservatives(generator, num_points, e == 1));
    metric.push_back(random_metric(generator, num_points));
  }
  auto batch = to_device(cons, metric);
  const PointwiseStatusSummary status =
      fix_on_device(make_not_null(&batch), variable_fixer);
  CHECK(status.number_of_failures == 0);

  bool host_needed_fixing = false;
  for (size_t e = 0; e < num_elements; ++e) {
    CAPTURE(e);
    auto expected = cons[e];
    host_needed_fixing |= variable_fixer(
        make_not_null(&get<Tags::TildeD>(expected)),
        make_not_null(&get<Tags::TildeYe>(expected)),
        make_not_null(&get<Tags::TildeTau>(expected)),
        make_not_null(&get<Tags::TildeS<>>(expected)),
        get<Tags::TildeB<>>(expected),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(metric[e]),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(metric[e]),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(metric[e]));
    const auto fixed = element_to_host(batch.conservatives, e);
    // The device code may be compiled differently (e.g. with fused
    // multiply-adds)
    const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeD>(fixed),
                                 get<Tags::TildeD>(expected), custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeYe>(fixed),
                                 get<Tags::TildeYe>(expected), custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeTau>(fixed),
                                 get<Tags::TildeTau>(expected), custom_approx);
    // The momentum density depends on the root of a polynomial that is only
    // found to a tolerance of 1e-14. The `DataVector` version finds roots for
    // SIMD batches of points, where some steps are chosen for the whole batch,
    // so the roots can differ within the tolerance. This difference is
    // amplified when the Lorentz factor is close to its lower bound.
    const Approx root_finder_approx =
        Approx::custom().epsilon(1.e-10).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeS<>>(fixed),
                                 get<Tags::TildeS<>>(expected),
                                 root_finder_approx);
    // The other variables are unchanged
    CHECK(get<Tags::TildeB<>>(fixed) == get<Tags::TildeB<>>(cons[e]));
    CHECK(get<Tags::TildePhi>(fixed) == get<Tags::TildePhi>(cons[e]));
  }
  CHECK(status.any_flag == host_needed_fixing);
}

// A spatial metric that isn't positive definite makes the magnetic field
// squared negative. Then the lower bound for the root find of the Lorentz
// factor is larger than the upper bound, so the root find fails. The failure is
// reported for the second element, whose variables stay unfixed, and the error
// shows the variables at its first grid point.
void test_root_find_failure() {
  const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
      1.e-12,  1.0e-11,
      1.0e-10, 1.0e-9,
      0.0,     0.0,
      1.e-12,  0.0,
      true,    hydro::MagneticFieldTreatment::AssumeNonZero};
  const size_t num_points = 3;
  Variables<conservative_tags> cons{num_points, 0.0};
  get(get<Tags::TildeD>(cons)) = 1.0;
  get(get<Tags::TildeYe>(cons)) = 0.1;
  get(get<Tags::TildeTau>(cons)) = 1.0;
  get<1>(get<Tags::TildeS<>>(cons)) = 10.0;
  get<0>(get<Tags::TildeB<>>(cons)) = 1.0;
  Variables<metric_tags> metric{num_points, 0.0};
  get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(metric)) = 1.0;
  for (size_t d = 0; d < 3; ++d) {
    get<gr::Tags::SpatialMetric<DataVector, 3>>(metric).get(d, d) = 1.0;
    get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(metric).get(d, d) = 1.0;
  }
  auto bad_metric = metric;
  get<0, 0>(get<gr::Tags::SpatialMetric<DataVector, 3>>(bad_metric)) = -1.0;

  auto batch = to_device({cons, cons}, {metric, bad_metric});
  // The magnetic field squared is negative, so the square root of the product
  // of the squares of the magnetic field and the momentum density is NaN
  const ScopedFpeState disable_fpes(false);
  const PointwiseStatusSummary status =
      fix_on_device(make_not_null(&batch), variable_fixer);
  CHECK(status.number_of_failures == num_points);
  CHECK(status.first_failed_element == 1);
  CHECK(status.first_failed_grid_point == 0);
  CHECK(element_to_host(batch.conservatives, 1) == cons);
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::error_fixing_conservatives_on_device(
          Kokkos::DefaultExecutionSpace{}, status, batch.conservatives,
          get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
              batch.metric),
          get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, 3>>>(
              batch.metric),
          get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
              batch.metric),
          "the element [B0,(L0I0)]"),
      Catch::Matchers::ContainsSubstring(
          "Failed to fix conserved variables on the device because the root "
          "finder failed to find the lorentz factor at 3 of 6 grid points. At "
          "the first one, grid point 0 of the element [B0,(L0I0)]:") and
          Catch::Matchers::ContainsSubstring("tilde_tau = T()=1.0"));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.GrMhd.ValenciaDivClean.FixConservatives.Kokkos",
    "[VariableFixing][Unit]") {
  MAKE_GENERATOR(generator);
  for (const auto magnetic_field_treatment :
       {hydro::MagneticFieldTreatment::AssumeZero,
        hydro::MagneticFieldTreatment::CheckIfZero,
        hydro::MagneticFieldTreatment::AssumeNonZero}) {
    test_random_batch(make_not_null(&generator), magnetic_field_treatment);
  }
  test_root_find_failure();
}
