// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/Serialize.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#endif  // SPECTRE_KOKKOS

namespace {
// The pointwise version that is used in Kokkos kernels must fix the variables
// like the `DataVector` version
void check_on_device(
    [[maybe_unused]] const grmhd::ValenciaDivClean::FixConservatives&
        variable_fixer,
    [[maybe_unused]] const Scalar<DataVector>& tilde_d,
    [[maybe_unused]] const Scalar<DataVector>& tilde_ye,
    [[maybe_unused]] const Scalar<DataVector>& tilde_tau,
    [[maybe_unused]] const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    [[maybe_unused]] const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    [[maybe_unused]] const tnsr::ii<DataVector, 3, Frame::Inertial>&
        spatial_metric,
    [[maybe_unused]] const tnsr::II<DataVector, 3, Frame::Inertial>&
        inv_spatial_metric,
    [[maybe_unused]] const Scalar<DataVector>& sqrt_det_spatial_metric) {
#ifdef SPECTRE_KOKKOS
  using AtGridPoint = grmhd::ValenciaDivClean::FixConservatives::AtGridPoint;
  namespace Tags = grmhd::ValenciaDivClean::Tags;
  auto host_tilde_d = tilde_d;
  auto host_tilde_ye = tilde_ye;
  auto host_tilde_tau = tilde_tau;
  auto host_tilde_s = tilde_s;
  const bool host_needed_fixing = variable_fixer(
      &host_tilde_d, &host_tilde_ye, &host_tilde_tau, &host_tilde_s, tilde_b,
      spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric);

  Variables<AtGridPoint::return_tags> device_results{get(tilde_d).size()};
  const bool device_needed_fixing =
      copy_and_apply_pointwise_on_device<AtGridPoint>(
          make_not_null(&device_results), AtGridPoint::return_tags{},
          AtGridPoint::argument_tags{}, tilde_d, tilde_ye, tilde_tau, tilde_s,
          tilde_b, spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
          variable_fixer,
          variable_fixer.assume_non_zero_magnetic_field(tilde_b));

  CHECK(device_needed_fixing == host_needed_fixing);
  // The device code may be compiled differently (e.g. with fused
  // multiply-adds)
  const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeD>(device_results), host_tilde_d,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeYe>(device_results),
                               host_tilde_ye, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeTau>(device_results),
                               host_tilde_tau, custom_approx);
  // The momentum density depends on the root of a polynomial that is only found
  // to a tolerance of 1e-14. The `DataVector` version finds roots for SIMD
  // batches of points, where some steps are chosen for the whole batch, so the
  // roots can differ within the tolerance. This difference is amplified when
  // the Lorentz factor is close to its lower bound.
  const Approx root_finder_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::TildeS<>>(device_results),
                               host_tilde_s, root_finder_approx);
#endif  // SPECTRE_KOKKOS
}

void test_variable_fixer(
    const grmhd::ValenciaDivClean::FixConservatives& variable_fixer,
    const bool enable) {
  // Call variable fixer at five points
  // [0]:  tilde_d is too small, should be raised to limit
  // [1]:  tilde_ye is too small, should be raised to limit
  // [2]:  tilde_tau is too small, raise to level of needed, which also
  //       causes tilde_s to be zeroed
  // [3]:  tilde_S is too big, so it is lowered
  // [4]:  all values are good, no changes

  Scalar<DataVector> tilde_d{DataVector{2.e-12, 1.0, 1.0, 1.0, 1.0}};
  // We assume that ye = 0.1
  Scalar<DataVector> tilde_ye{DataVector{2.e-13, 2.0e-10, 1.0, 1.0, 1.0}};
  Scalar<DataVector> tilde_tau{DataVector{4.5, 4.5, 1.5, 4.5, 4.5}};
  auto tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto tilde_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  tilde_s.get(0) = DataVector{3.0, 3.0, 0.0, 6.0, 5.0};
  tilde_b.get(1) = DataVector{2.0, 2.0, 2.0, 2.0, 2.0};

  auto expected_tilde_d = tilde_d;
  if (enable) {
    get(expected_tilde_d)[0] = 1.e-12;
  }

  auto expected_tilde_ye = tilde_ye;
  if (enable) {
    get(expected_tilde_ye)[0] = 1.e-13;  // since Y_e = 0.1
    get(expected_tilde_ye)[1] = 1.e-10;
  }

  auto expected_tilde_tau = tilde_tau;
  if (enable) {
    get(expected_tilde_tau)[2] = 2.0;
  }

  auto expected_tilde_s = tilde_s;
  if (enable) {
    expected_tilde_s.get(0)[3] = sqrt(27.0);
  }

  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(tilde_d, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
    inv_spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
  }

  check_on_device(variable_fixer, tilde_d, tilde_ye, tilde_tau, tilde_s,
                  tilde_b, spatial_metric, inv_spatial_metric,
                  sqrt_det_spatial_metric);
  CHECK(enable == variable_fixer(&tilde_d, &tilde_ye, &tilde_tau, &tilde_s,
                                 tilde_b, spatial_metric, inv_spatial_metric,
                                 sqrt_det_spatial_metric));

  CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d);
  CHECK_ITERABLE_APPROX(tilde_ye, expected_tilde_ye);
  CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
  CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s);
}

void test_variable_fixer_zero_b_field(
    const grmhd::ValenciaDivClean::FixConservatives& variable_fixer,
    const bool enable) {
  // Call variable fixer with AssumeZero for MagneticField
  // at two points.
  // [0] : negative tilde tau, should be raised to zero.
  // [1] : positive tilde tau, no changes.
  Scalar<DataVector> tilde_d{DataVector{1.0, 1.0}};
  // We assume that ye = 0.1
  Scalar<DataVector> tilde_ye{DataVector{1.0, 1.0}};
  Scalar<DataVector> tilde_tau{DataVector{-2.0, 4.5}};
  auto expected_tilde_tau = tilde_tau;
  if (enable) {
    get(expected_tilde_tau)[0] = 0.0;
  }
  auto tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto tilde_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(tilde_d, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
    inv_spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
  }

  check_on_device(variable_fixer, tilde_d, tilde_ye, tilde_tau, tilde_s,
                  tilde_b, spatial_metric, inv_spatial_metric,
                  sqrt_det_spatial_metric);
  CHECK(enable == variable_fixer(&tilde_d, &tilde_ye, &tilde_tau, &tilde_s,
                                 tilde_b, spatial_metric, inv_spatial_metric,
                                 sqrt_det_spatial_metric));
  CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
}

// Random states, many of which need fixing, including the momentum density
void test_random_on_device(
    const gsl::not_null<std::mt19937*> generator,
    const hydro::MagneticFieldTreatment magnetic_field_treatment) {
  CAPTURE(magnetic_field_treatment);
  const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
      1.e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.e-4,
      1.e-3,  1.0e-1,  1.e-2,   true,   magnetic_field_treatment};
  const size_t num_points = 100;
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_d(1.e-12, 1.0);
  std::uniform_real_distribution<> dist_unit(0.0, 1.0);
  std::uniform_real_distribution<> dist_vector(-2.0, 2.0);
  std::uniform_real_distribution<> dist_metric(0.8, 1.2);
  const auto tilde_d = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_d), used_for_size);
  auto tilde_ye = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), used_for_size);
  get(tilde_ye) *= 0.5 * get(tilde_d);
  auto tilde_tau = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), used_for_size);
  get(tilde_tau) = 2.0 * get(tilde_tau) - 0.2;
  const auto tilde_s =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_vector), used_for_size);
  const auto tilde_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_vector), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist_metric), used_for_size);
    inv_spatial_metric.get(d, d) = 1.0 / spatial_metric.get(d, d);
    get(sqrt_det_spatial_metric) *= sqrt(spatial_metric.get(d, d));
  }
  check_on_device(variable_fixer, tilde_d, tilde_ye, tilde_tau, tilde_s,
                  tilde_b, spatial_metric, inv_spatial_metric,
                  sqrt_det_spatial_metric);
}

void run_benchmark(const bool enable) {
  if (not enable) {
    return;
  }
  const size_t num_points = 512;
  Scalar<DataVector> tilde_d{DataVector(num_points)};
  Scalar<DataVector> tilde_ye{DataVector(num_points)};
  Scalar<DataVector> tilde_tau{DataVector(num_points)};
  auto tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto tilde_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  for (size_t i = 0; i < num_points; ++i) {
    if (i % 2 == 0) {
      // tilde_d is too small, should be raised to limit
      get(tilde_d)[i] = 2.e-12;
      get(tilde_ye)[i] = 2.e-13;
      get(tilde_tau)[i] = 4.5;
      tilde_s.get(0)[i] = 3.0;
      tilde_b.get(0)[i] = 2.0e-6;
    } else {
      // tilde_tau is too small, raise to level of needed, which also
      // causes tilde_s to be zeroed
      get(tilde_d)[i] = 1.0;
      get(tilde_ye)[i] = 1.0;
      get(tilde_tau)[i] = 1.5;
      tilde_s.get(0)[i] = 0.0;
      tilde_b.get(0)[i] = 2.0;
    }
  }
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(tilde_d, 0.0);
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(tilde_d, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
    inv_spatial_metric.get(d, d) = get(sqrt_det_spatial_metric);
  }

  const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
      1.e-12,  1.0e-11,
      1.0e-10, 1.0e-9,
      0.0,     0.0,
      1.0e-8,  0.0001,
      true,    hydro::MagneticFieldTreatment::AssumeNonZero};

  BENCHMARK("FixConservatives") {
    // Note: Benchmarking and perf indicates that software prefetching is likely
    // the next big gain for conservative variable fixing.
    variable_fixer(&tilde_d, &tilde_ye, &tilde_tau, &tilde_s, tilde_b,
                   spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric);
  };
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.GrMhd.ValenciaDivClean.FixConservatives",
                  "[VariableFixing][Unit]") {
  for (const bool enable : {true, false}) {
    const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
        1.e-12,  1.0e-11,
        1.0e-10, 1.0e-9,
        0.0,     0.0,
        1.e-12,  0.0,
        enable,  hydro::MagneticFieldTreatment::AssumeNonZero};

    const grmhd::ValenciaDivClean::FixConservatives variable_fixer_zero_b_field{
        1.e-12,  1.0e-11,
        1.0e-10, 1.0e-9,
        0.0,     0.0,
        1.e-12,  0.0,
        enable,  hydro::MagneticFieldTreatment::AssumeZero};

    test_variable_fixer(serialize_and_deserialize(variable_fixer), enable);
    test_variable_fixer_zero_b_field(
        serialize_and_deserialize(variable_fixer_zero_b_field), enable);
    test_serialization(variable_fixer);
    test_serialization(variable_fixer_zero_b_field);
  }

  const auto fixer_from_options =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::FixConservatives>(
          "MinimumValueOfD: 1.0e-12\n"
          "CutoffD: 1.0e-11\n"
          "MinimumValueOfYe: 1.0e-10\n"
          "CutoffYe: 1.0e-9\n"
          "SafetyFactorForB: 0.0\n"
          "SafetyFactorForS: 0.0\n"
          "SafetyFactorForSCutoffD: 1.0e-12\n"
          "SafetyFactorForSSlope: 0.0\n"
          "Enable: true\n"
          "MagneticField: AssumeNonZero\n");

  const auto fixer_from_options_zero_b_field =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::FixConservatives>(
          "MinimumValueOfD: 1.0e-12\n"
          "CutoffD: 1.0e-11\n"
          "MinimumValueOfYe: 1.0e-10\n"
          "CutoffYe: 1.0e-9\n"
          "SafetyFactorForB: 0.0\n"
          "SafetyFactorForS: 0.0\n"
          "SafetyFactorForSCutoffD: 1.0e-12\n"
          "SafetyFactorForSSlope: 0.0\n"
          "Enable: true\n"
          "MagneticField: AssumeZero\n");
  test_variable_fixer(fixer_from_options, true);
  test_variable_fixer_zero_b_field(fixer_from_options_zero_b_field, true);

  MAKE_GENERATOR(generator);
  for (const auto magnetic_field_treatment :
       {hydro::MagneticFieldTreatment::AssumeZero,
        hydro::MagneticFieldTreatment::CheckIfZero,
        hydro::MagneticFieldTreatment::AssumeNonZero}) {
    test_random_on_device(make_not_null(&generator), magnetic_field_treatment);
  }

  run_benchmark(false);
}
