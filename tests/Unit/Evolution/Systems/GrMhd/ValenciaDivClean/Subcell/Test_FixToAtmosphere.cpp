// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.FixToAtmosphere",
    "[Unit][Evolution]") {
  using Fixer = VariableFixing::FixToAtmosphere<3>;
  using Mutator = grmhd::ValenciaDivClean::subcell::FixToAtmosphere;
  static_assert(std::is_same_v<Mutator::return_tags, Fixer::return_tags>);
  static_assert(
      std::is_same_v<
          Mutator::argument_tags,
          tmpl::push_back<Fixer::argument_tags, ::Tags::VariableFixer<Fixer>>>);

  MAKE_GENERATOR(generator);
  const size_t num_points = 50;
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_log_density(-13.0, -9.0);
  std::uniform_real_distribution<> dist_velocity(-0.3, 0.3);
  std::uniform_real_distribution<> dist_positive(0.1, 1.0);
  auto rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist_log_density),
      used_for_size);
  for (double& density : get(rest_mass_density)) {
    density = std::pow(10.0, density);
  }
  const auto spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist_velocity),
          used_for_size);
  const auto other = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist_positive), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = 1.0;
  }
  const Fixer fixer{1.e-12, 3.e-12,
                    Fixer::VelocityLimitingOptions{0.0, 1.e-4, 3.e-12, 1.e-11},
                    Fixer::KappaLimitingOptions{3.e-12, 1.e-3, 3.e-11, 0.01,
                                                std::nullopt, false}};
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>> eos{
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}};

  auto expected_rest_mass_density = rest_mass_density;
  auto expected_specific_internal_energy = other;
  auto expected_spatial_velocity = spatial_velocity;
  auto expected_lorentz_factor = other;
  auto expected_pressure = other;
  auto expected_temperature = Scalar<DataVector>{1.e-3 * get(other)};
  fixer(make_not_null(&expected_rest_mass_density),
        make_not_null(&expected_specific_internal_energy),
        make_not_null(&expected_spatial_velocity),
        make_not_null(&expected_lorentz_factor),
        make_not_null(&expected_pressure), make_not_null(&expected_temperature),
        other, spatial_metric, eos);

  auto box = db::create<tmpl::list<
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::LorentzFactor<DataVector>, hydro::Tags::Pressure<DataVector>,
      hydro::Tags::Temperature<DataVector>,
      hydro::Tags::ElectronFraction<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3>, hydro::Tags::GrmhdEquationOfState,
      ::Tags::VariableFixer<Fixer>>>(
      rest_mass_density, other, spatial_velocity, other, other,
      Scalar<DataVector>{1.e-3 * get(other)}, other, spatial_metric,
      std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>{
          std::make_unique<EquationsOfState::Equilibrium3D<
              EquationsOfState::IdealFluid<true>>>(eos)},
      fixer);
  db::mutate_apply<Mutator>(make_not_null(&box));

  // In Kokkos builds the variables are fixed on the device, which may be
  // compiled differently (e.g. with fused multiply-adds), and the temperature
  // is found with a root finder to a relative tolerance of 1e-13
  const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      db::get<hydro::Tags::RestMassDensity<DataVector>>(box),
      expected_rest_mass_density, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      db::get<hydro::Tags::SpecificInternalEnergy<DataVector>>(box),
      expected_specific_internal_energy, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      (db::get<hydro::Tags::SpatialVelocity<DataVector, 3>>(box)),
      expected_spatial_velocity, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      db::get<hydro::Tags::LorentzFactor<DataVector>>(box),
      expected_lorentz_factor, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(db::get<hydro::Tags::Pressure<DataVector>>(box),
                               expected_pressure, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      db::get<hydro::Tags::Temperature<DataVector>>(box), expected_temperature,
      custom_approx);
}
