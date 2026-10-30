// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.tpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
using Fixer = VariableFixing::FixToAtmosphere<3>;
using Vlo = Fixer::VelocityLimitingOptions;
using Klo = Fixer::KappaLimitingOptions;

// Random states around the density cutoff, the velocity limiting densities and
// the kappa limiting densities of the fixers below, so all branches are taken
void test_random_on_device(
    const gsl::not_null<std::mt19937*> generator, const Fixer& fixer,
    const EquationsOfState::EquationOfState<true, 3>& eos) {
  const size_t num_points = 200;
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_log_density(-13.0, -8.0);
  std::uniform_real_distribution<> dist_velocity(-0.3, 0.3);
  std::uniform_real_distribution<> dist_temperature(0.0, 2.e-3);
  std::uniform_real_distribution<> dist_positive(0.1, 1.0);
  std::uniform_real_distribution<> dist_metric(0.9, 1.1);
  std::uniform_real_distribution<> dist_off_diagonal(-0.05, 0.05);

  auto rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_log_density), used_for_size);
  for (double& density : get(rest_mass_density)) {
    density = std::pow(10.0, density);
  }
  // A few points far above the atmosphere
  get(rest_mass_density)[0] = 1.e-3;
  get(rest_mass_density)[1] = 0.5;
  auto specific_internal_energy = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_positive), used_for_size);
  const auto spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_velocity), used_for_size);
  const auto lorentz_factor = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_positive), used_for_size);
  const auto pressure = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_positive), used_for_size);
  const auto temperature = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_temperature), used_for_size);
  auto electron_fraction = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_positive), used_for_size);
  get(electron_fraction) *= 0.5;
  auto spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_off_diagonal), used_for_size);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist_metric), used_for_size);
  }

  auto host_rest_mass_density = rest_mass_density;
  auto host_specific_internal_energy = specific_internal_energy;
  auto host_spatial_velocity = spatial_velocity;
  auto host_lorentz_factor = lorentz_factor;
  auto host_pressure = pressure;
  auto host_temperature = temperature;
  fixer(make_not_null(&host_rest_mass_density),
        make_not_null(&host_specific_internal_energy),
        make_not_null(&host_spatial_velocity),
        make_not_null(&host_lorentz_factor), make_not_null(&host_pressure),
        make_not_null(&host_temperature), electron_fraction, spatial_metric,
        eos);

  auto device_rest_mass_density = rest_mass_density;
  auto device_specific_internal_energy = specific_internal_energy;
  auto device_spatial_velocity = spatial_velocity;
  auto device_lorentz_factor = lorentz_factor;
  auto device_pressure = pressure;
  auto device_temperature = temperature;
  VariableFixing::fix_to_atmosphere_on_device(
      make_not_null(&device_rest_mass_density),
      make_not_null(&device_specific_internal_energy),
      make_not_null(&device_spatial_velocity),
      make_not_null(&device_lorentz_factor), make_not_null(&device_pressure),
      make_not_null(&device_temperature), electron_fraction, spatial_metric,
      eos, fixer);

  // The device code may be compiled differently (e.g. with fused
  // multiply-adds)
  const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(device_rest_mass_density, host_rest_mass_density,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_spatial_velocity, host_spatial_velocity,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_lorentz_factor, host_lorentz_factor,
                               custom_approx);
  // The temperature of the kappa limiting is found with a root finder to a
  // relative tolerance of 1e-13
  const Approx root_finder_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(device_temperature, host_temperature,
                               root_finder_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_specific_internal_energy,
                               host_specific_internal_energy,
                               root_finder_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_pressure, host_pressure,
                               root_finder_approx);

  // The helper used by the FD actions runs on the device in Kokkos builds
  auto helper_rest_mass_density = rest_mass_density;
  auto helper_specific_internal_energy = specific_internal_energy;
  auto helper_spatial_velocity = spatial_velocity;
  auto helper_lorentz_factor = lorentz_factor;
  auto helper_pressure = pressure;
  auto helper_temperature = temperature;
  VariableFixing::fix_to_atmosphere_on_host_or_device(
      make_not_null(&helper_rest_mass_density),
      make_not_null(&helper_specific_internal_energy),
      make_not_null(&helper_spatial_velocity),
      make_not_null(&helper_lorentz_factor), make_not_null(&helper_pressure),
      make_not_null(&helper_temperature), electron_fraction, spatial_metric,
      eos, fixer);
  CHECK(helper_rest_mass_density == device_rest_mass_density);
  CHECK(helper_specific_internal_energy == device_specific_internal_energy);
  CHECK(helper_spatial_velocity == device_spatial_velocity);
  CHECK(helper_lorentz_factor == device_lorentz_factor);
  CHECK(helper_pressure == device_pressure);
  CHECK(helper_temperature == device_temperature);
}
// The options don't allow a negative minimum temperature, but with one the
// pressure at the minimum temperature is negative. Then the maximum pressure of
// the kappa limiting, which is a factor larger than one times that, is smaller
// and doesn't bound the root, so the kappa limiting fails. In a Kokkos
// kernel this is reported through a flag.
void test_kappa_limiting_failure_on_device() {
  auto fixer_view = Fixer{
      1.e-12, 3.e-12, std::nullopt,
      Fixer::KappaLimitingOptions{
          3.e-12, 1.e-3, 3.e-11, 0.01, std::nullopt,
          false}}.device_view();
  fixer_view.kappa_limiting.min_temperature_from_eos = false;
  fixer_view.kappa_limiting.min_temperature = -1.0;
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>> eos{
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}};
  const auto eos_view = eos.device_view();
  const auto fix = KOKKOS_LAMBDA() {
    bool root_find_failed = false;
    double rest_mass_density = 1.e-11;
    double specific_internal_energy = 1.0;
    tnsr::I<double, 3, Frame::Inertial> spatial_velocity{0.0};
    double lorentz_factor = 1.0;
    double pressure = 1.0;
    double temperature = 1.0;
    tnsr::ii<double, 3, Frame::Inertial> spatial_metric{0.0};
    for (size_t d = 0; d < 3; ++d) {
      spatial_metric.get(d, d) = 1.0;
    }
    fixer_view.template fix_at_point<3>(
        make_not_null(&rest_mass_density),
        make_not_null(&specific_internal_energy),
        make_not_null(&spatial_velocity), make_not_null(&lorentz_factor),
        make_not_null(&pressure), make_not_null(&temperature), 0.1,
        spatial_metric, eos_view, &root_find_failed);
    return std::array{root_find_failed ? 1.0 : 0.0};
  };
  TestHelpers::check_on_device(fix);
  CHECK(fix()[0] == 1.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.FixToAtmosphere.Kokkos",
                  "[VariableFixing][Unit]") {
  MAKE_GENERATOR(generator);
  const Fixer no_limiting{1.e-12, 3.e-12, std::nullopt, std::nullopt};
  const Fixer limiting{1.e-12, 3.e-12, Vlo{0.0, 1.e-4, 3.e-12, 1.e-11},
                       Klo{3.e-12, 1.e-3, 3.e-11, 0.01, std::nullopt, false}};
  const Fixer limiting_everywhere{
      1.e-12, 3.e-12, Vlo{1.e-5, 1.e-4, 3.e-12, 1.e-11},
      Klo{3.e-12, 1.e-3, 3.e-11, 0.01, 1.e-3, true}};
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
      ideal_fluid{EquationsOfState::IdealFluid<true>{5.0 / 3.0}};
  const EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>
      polytrope{EquationsOfState::PolytropicFluid<true>{100.0, 2.0}};
  for (const auto* const fixer :
       {&no_limiting, &limiting, &limiting_everywhere}) {
    test_random_on_device(make_not_null(&generator), *fixer, ideal_fluid);
    test_random_on_device(make_not_null(&generator), *fixer, polytrope);
  }
  test_kappa_limiting_failure_on_device();
}
