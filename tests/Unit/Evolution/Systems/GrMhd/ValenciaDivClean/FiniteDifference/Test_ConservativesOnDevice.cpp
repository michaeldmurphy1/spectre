// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ConservativesOnDevice.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
namespace EoS = EquationsOfState;

// Random primitive variables and a diagonal spatial metric
struct Primitives {
  Scalar<DataVector> rest_mass_density;
  Scalar<DataVector> electron_fraction;
  Scalar<DataVector> temperature;
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity;
  Scalar<DataVector> lorentz_factor;
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field;
  Scalar<DataVector> sqrt_det_spatial_metric;
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric;
  Scalar<DataVector> divergence_cleaning_field;
};

Primitives make_primitives(const gsl::not_null<std::mt19937*> generator,
                           const size_t num_points) {
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_density(0.1, 2.0);
  std::uniform_real_distribution<> dist_electron_fraction(0.05, 0.5);
  std::uniform_real_distribution<> dist_velocity(-0.3, 0.3);
  std::uniform_real_distribution<> dist_field(-1.0, 1.0);
  std::uniform_real_distribution<> dist_metric(0.8, 1.2);
  Primitives prims{
      .rest_mass_density = make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_density), used_for_size),
      .electron_fraction = make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_electron_fraction), used_for_size),
      .temperature = make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_density), used_for_size),
      .spatial_velocity =
          make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
              generator, make_not_null(&dist_velocity), used_for_size),
      .lorentz_factor = make_with_value<Scalar<DataVector>>(used_for_size, 0.0),
      .magnetic_field =
          make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
              generator, make_not_null(&dist_field), used_for_size),
      .sqrt_det_spatial_metric =
          make_with_value<Scalar<DataVector>>(used_for_size, 1.0),
      .spatial_metric =
          make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
              used_for_size, 0.0),
      .divergence_cleaning_field = make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_field), used_for_size)};
  for (size_t d = 0; d < 3; ++d) {
    prims.spatial_metric.get(d, d) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist_metric), used_for_size);
    get(prims.sqrt_det_spatial_metric) *= sqrt(prims.spatial_metric.get(d, d));
  }
  // v^2 < 3 * 1.2 * 0.3^2 < 1
  get(prims.lorentz_factor) =
      1.0 /
      sqrt(1.0 - get(dot_product(prims.spatial_velocity, prims.spatial_velocity,
                                 prims.spatial_metric)));
  return prims;
}

// The fused kernel must give the same results as the EOS and
// `ConservativeFromPrimitive` on the host
void check_against_host(const gsl::not_null<std::mt19937*> generator,
                        const EoS::EquationOfState<true, 3>& eos) {
  const size_t num_points = 50;
  const Primitives prims = make_primitives(generator, num_points);

  const auto expected_specific_internal_energy =
      eos.specific_internal_energy_from_density_and_temperature(
          prims.rest_mass_density, prims.temperature, prims.electron_fraction);
  const auto expected_pressure = eos.pressure_from_density_and_temperature(
      prims.rest_mass_density, prims.temperature, prims.electron_fraction);
  Scalar<DataVector> expected_tilde_d{num_points};
  Scalar<DataVector> expected_tilde_ye{num_points};
  Scalar<DataVector> expected_tilde_tau{num_points};
  tnsr::i<DataVector, 3, Frame::Inertial> expected_tilde_s{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> expected_tilde_b{num_points};
  Scalar<DataVector> expected_tilde_phi{num_points};
  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&expected_tilde_d), make_not_null(&expected_tilde_ye),
      make_not_null(&expected_tilde_tau), make_not_null(&expected_tilde_s),
      make_not_null(&expected_tilde_b), make_not_null(&expected_tilde_phi),
      prims.rest_mass_density, prims.electron_fraction,
      expected_specific_internal_energy, expected_pressure,
      prims.spatial_velocity, prims.lorentz_factor, prims.magnetic_field,
      prims.sqrt_det_spatial_metric, prims.spatial_metric,
      prims.divergence_cleaning_field);

  Scalar<DataVector> specific_internal_energy{num_points};
  Scalar<DataVector> pressure{num_points};
  Scalar<DataVector> tilde_d{num_points};
  Scalar<DataVector> tilde_ye{num_points};
  Scalar<DataVector> tilde_tau{num_points};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{num_points};
  Scalar<DataVector> tilde_phi{num_points};
  grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
      Kokkos::DefaultExecutionSpace{}, make_not_null(&specific_internal_energy),
      make_not_null(&pressure), make_not_null(&tilde_d),
      make_not_null(&tilde_ye), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), make_not_null(&tilde_b),
      make_not_null(&tilde_phi), prims.rest_mass_density,
      prims.electron_fraction, prims.temperature, prims.spatial_velocity,
      prims.lorentz_factor, prims.magnetic_field, prims.sqrt_det_spatial_metric,
      prims.spatial_metric, prims.divergence_cleaning_field, eos);

  // The device code may be compiled differently (e.g. with fused
  // multiply-adds)
  const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(specific_internal_energy,
                               expected_specific_internal_energy,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(pressure, expected_pressure, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(tilde_d, expected_tilde_d, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(tilde_ye, expected_tilde_ye, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(tilde_tau, expected_tilde_tau, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(tilde_s, expected_tilde_s, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(tilde_b, expected_tilde_b, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(tilde_phi, expected_tilde_phi, custom_approx);
}

// EOSs without a `DeviceView` are an error rather than evaluated on the host
void check_error_without_device_view(
    const gsl::not_null<std::mt19937*> generator) {
  const size_t num_points = 5;
  const Primitives prims = make_primitives(generator, num_points);
  const EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>> eos{
      EoS::HybridEos<EoS::PolytropicFluid<true>>{
          EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5}};
  Scalar<DataVector> specific_internal_energy{num_points};
  Scalar<DataVector> pressure{num_points};
  Scalar<DataVector> tilde_d{num_points};
  Scalar<DataVector> tilde_ye{num_points};
  Scalar<DataVector> tilde_tau{num_points};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{num_points};
  Scalar<DataVector> tilde_phi{num_points};
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
          Kokkos::DefaultExecutionSpace{},
          make_not_null(&specific_internal_energy), make_not_null(&pressure),
          make_not_null(&tilde_d), make_not_null(&tilde_ye),
          make_not_null(&tilde_tau), make_not_null(&tilde_s),
          make_not_null(&tilde_b), make_not_null(&tilde_phi),
          prims.rest_mass_density, prims.electron_fraction, prims.temperature,
          prims.spatial_velocity, prims.lorentz_factor, prims.magnetic_field,
          prims.sqrt_det_spatial_metric, prims.spatial_metric,
          prims.divergence_cleaning_field, eos),
      Catch::Matchers::ContainsSubstring(
          "has no DeviceView, so it can't be evaluated on the device"));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Fd.ConservativesOnDevice",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  check_against_host(make_not_null(&generator),
                     EoS::Equilibrium3D<EoS::IdealFluid<true>>{
                         EoS::IdealFluid<true>{5.0 / 3.0}});
  check_against_host(make_not_null(&generator),
                     EoS::Barotropic3D<EoS::PolytropicFluid<true>>{
                         EoS::PolytropicFluid<true>{100.0, 2.0}});
  check_error_without_device_view(make_not_null(&generator));
}
