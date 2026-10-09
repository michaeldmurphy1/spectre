// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// The pointwise version that is used in Kokkos kernels must fix the variables
// like the `DataVector` version
void check_on_device(
    const grmhd::ValenciaDivClean::FixConservatives& variable_fixer,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
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
}

// Random states, many of which need fixing, including the momentum density
void test_random_on_device(
    const gsl::not_null<std::mt19937*> generator,
    const hydro::MagneticFieldTreatment magnetic_field_treatment,
    const bool zero_magnetic_field) {
  CAPTURE(magnetic_field_treatment);
  CAPTURE(zero_magnetic_field);
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
  auto tilde_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_vector), used_for_size);
  if (zero_magnetic_field) {
    for (auto& component : tilde_b) {
      component = 0.0;
    }
  }
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
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.GrMhd.ValenciaDivClean.FixConservatives.Kokkos",
    "[VariableFixing][Unit]") {
  MAKE_GENERATOR(generator);
  for (const auto magnetic_field_treatment :
       {hydro::MagneticFieldTreatment::AssumeZero,
        hydro::MagneticFieldTreatment::CheckIfZero,
        hydro::MagneticFieldTreatment::AssumeNonZero}) {
    for (const bool zero_magnetic_field : {false, true}) {
      test_random_on_device(make_not_null(&generator), magnetic_field_treatment,
                            zero_magnetic_field);
    }
  }
}
