// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDevice.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Schemes = grmhd::ValenciaDivClean::PrimitiveRecoverySchemes;
using KastaunThenNewmanThenPalenzuela =
    tmpl::list<Schemes::KastaunEtAl, Schemes::NewmanHamlin,
               Schemes::PalenzuelaEtAl>;
using NewmanHamlinThenPalenzuelaEtAl =
    tmpl::list<Schemes::NewmanHamlin, Schemes::PalenzuelaEtAl>;

struct Conservatives {
  Scalar<DataVector> tilde_d;
  Scalar<DataVector> tilde_ye;
  Scalar<DataVector> tilde_tau;
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s;
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b;
  Scalar<DataVector> tilde_phi;
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric;
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric;
  Scalar<DataVector> sqrt_det_spatial_metric;
};

// Random physical states. The magnetic field is zero at every other point, so
// both `KastaunEtAl` and `KastaunEtAlHydro` are used. The rest mass density is
// tiny at every fifth point, so these points are atmosphere.
template <typename Eos>
Conservatives random_conservatives(const gsl::not_null<std::mt19937*> generator,
                                   const Eos& eos, const size_t num_points) {
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_density(0.1, 1.0);
  std::uniform_real_distribution<> dist_unit(0.0, 1.0);
  std::uniform_real_distribution<> dist_velocity(-0.4, 0.4);
  std::uniform_real_distribution<> dist_field(-1.0, 1.0);
  std::uniform_real_distribution<> dist_metric(0.8, 1.2);

  auto rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_density), used_for_size);
  for (size_t i = 0; i < num_points; i += 5) {
    get(rest_mass_density)[i] = 1.e-14;
  }
  auto electron_fraction = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), used_for_size);
  get(electron_fraction) = 0.05 + 0.4 * get(electron_fraction);
  auto temperature = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), used_for_size);
  get(temperature) = 0.2 + get(temperature);
  const auto specific_internal_energy =
      eos.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  const auto pressure = eos.pressure_from_density_and_temperature(
      rest_mass_density, temperature, electron_fraction);
  auto magnetic_field =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_field), used_for_size);
  for (size_t i = 0; i < num_points; i += 2) {
    for (size_t d = 0; d < 3; ++d) {
      magnetic_field.get(d)[i] = 0.0;
    }
  }
  const auto divergence_cleaning_field =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_field), used_for_size);

  Conservatives result{};
  result.spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  result.inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  result.sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    result.spatial_metric.get(d, d) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist_metric), used_for_size);
    result.inv_spatial_metric.get(d, d) = 1.0 / result.spatial_metric.get(d, d);
    get(result.sqrt_det_spatial_metric) *=
        sqrt(result.spatial_metric.get(d, d));
  }
  // The velocity is small enough that v^i v_i < 1 for these metrics
  const auto spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_velocity), used_for_size);
  Scalar<DataVector> lorentz_factor{num_points, 0.0};
  for (size_t d = 0; d < 3; ++d) {
    get(lorentz_factor) +=
        result.spatial_metric.get(d, d) * square(spatial_velocity.get(d));
  }
  get(lorentz_factor) = 1.0 / sqrt(1.0 - get(lorentz_factor));

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&result.tilde_d), make_not_null(&result.tilde_ye),
      make_not_null(&result.tilde_tau), make_not_null(&result.tilde_s),
      make_not_null(&result.tilde_b), make_not_null(&result.tilde_phi),
      rest_mass_density, electron_fraction, specific_internal_energy, pressure,
      spatial_velocity, lorentz_factor, magnetic_field,
      result.sqrt_det_spatial_metric, result.spatial_metric,
      divergence_cleaning_field);
  return result;
}

// Recovers the primitives on the host and on the device and checks that they
// agree. Returns whether the recovery succeeded.
template <typename OrderedListOfSchemes, bool EnforcePhysicality>
bool check_on_device(
    const Conservatives& cons,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions& options) {
  CAPTURE(pretty_type::get_name<OrderedListOfSchemes>());
  CAPTURE(EnforcePhysicality);
  const size_t num_points = get(cons.tilde_d).size();
  const DataVector used_for_size(num_points);
  // The initial values, which are kept where the recovery doesn't change them
  auto host_rest_mass_density =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.5);
  auto host_electron_fraction =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.1);
  auto host_specific_internal_energy =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  auto host_spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                               0.0);
  auto host_magnetic_field =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                               0.0);
  auto host_divergence_cleaning_field =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto host_lorentz_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  auto host_pressure = make_with_value<Scalar<DataVector>>(used_for_size, 0.3);
  auto host_temperature =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);

  auto device_rest_mass_density = host_rest_mass_density;
  auto device_electron_fraction = host_electron_fraction;
  auto device_specific_internal_energy = host_specific_internal_energy;
  auto device_spatial_velocity = host_spatial_velocity;
  auto device_magnetic_field = host_magnetic_field;
  auto device_divergence_cleaning_field = host_divergence_cleaning_field;
  auto device_lorentz_factor = host_lorentz_factor;
  auto device_pressure = host_pressure;
  auto device_temperature = host_temperature;

  const bool host_succeeded =
      grmhd::ValenciaDivClean::PrimitiveFromConservative<OrderedListOfSchemes,
                                                         false>::
          template apply<EnforcePhysicality>(
              make_not_null(&host_rest_mass_density),
              make_not_null(&host_electron_fraction),
              make_not_null(&host_specific_internal_energy),
              make_not_null(&host_spatial_velocity),
              make_not_null(&host_magnetic_field),
              make_not_null(&host_divergence_cleaning_field),
              make_not_null(&host_lorentz_factor),
              make_not_null(&host_pressure), make_not_null(&host_temperature),
              cons.tilde_d, cons.tilde_ye, cons.tilde_tau, cons.tilde_s,
              cons.tilde_b, cons.tilde_phi, cons.spatial_metric,
              cons.inv_spatial_metric, cons.sqrt_det_spatial_metric, eos,
              options);
  const bool device_succeeded =
      grmhd::ValenciaDivClean::primitive_from_conservative_on_device<
          OrderedListOfSchemes, false, EnforcePhysicality>(
          Kokkos::DefaultExecutionSpace{},
          make_not_null(&device_rest_mass_density),
          make_not_null(&device_electron_fraction),
          make_not_null(&device_specific_internal_energy),
          make_not_null(&device_spatial_velocity),
          make_not_null(&device_magnetic_field),
          make_not_null(&device_divergence_cleaning_field),
          make_not_null(&device_lorentz_factor),
          make_not_null(&device_pressure), make_not_null(&device_temperature),
          cons.tilde_d, cons.tilde_ye, cons.tilde_tau, cons.tilde_s,
          cons.tilde_b, cons.tilde_phi, cons.spatial_metric,
          cons.inv_spatial_metric, cons.sqrt_det_spatial_metric, eos, options);
  CHECK(device_succeeded == host_succeeded);
  if (not host_succeeded) {
    // The host stops at the first grid point where the recovery fails, so the
    // results differ
    return false;
  }

  // The root finds converge to a tolerance of ~1e-15, and the device code may
  // be compiled differently (e.g. with fused multiply-adds)
  const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(device_rest_mass_density, host_rest_mass_density,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_electron_fraction, host_electron_fraction,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_specific_internal_energy,
                               host_specific_internal_energy, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_spatial_velocity, host_spatial_velocity,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_magnetic_field, host_magnetic_field,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_divergence_cleaning_field,
                               host_divergence_cleaning_field, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_lorentz_factor, host_lorentz_factor,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_pressure, host_pressure, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(device_temperature, host_temperature,
                               custom_approx);
  return true;
}

template <typename OrderedListOfSchemes, typename Eos>
void test_recovery(const gsl::not_null<std::mt19937*> generator,
                   const Eos& eos) {
  CAPTURE(pretty_type::get_runtime_type_name(eos));
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions options{
      1.e-12, 1.e-12, 1.e4};
  const auto cons = random_conservatives(generator, eos, 50);
  CHECK(check_on_device<OrderedListOfSchemes, true>(cons, eos, options));
  CHECK(check_on_device<OrderedListOfSchemes, false>(cons, eos, options));

  // The whole grid is below the cutoff, so it is all atmosphere
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      large_cutoff_options{10.0, 1.e-10, 1.e4};
  CHECK(check_on_device<OrderedListOfSchemes, true>(cons, eos,
                                                    large_cutoff_options));

  // Unphysical: all schemes fail if physicality isn't enforced
  auto unphysical_cons = cons;
  get(unphysical_cons.tilde_tau) = -10.0;
  CHECK_FALSE(check_on_device<OrderedListOfSchemes, false>(unphysical_cons, eos,
                                                           options));
  // Enforcing physicality, only some schemes recover some of these states, but
  // the host and device must agree (which is checked in `check_on_device`)
  check_on_device<OrderedListOfSchemes, true>(unphysical_cons, eos, options);
}

// Without `ErrorOnFailure` the recovery continues past failures, and the
// primitive variables are NaN only at the failed grid points
template <typename Eos>
void test_nan_at_failures(const gsl::not_null<std::mt19937*> generator,
                          const Eos& eos) {
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions options{
      1.e-12, 1.e-12, 1.e4};
  const size_t num_points = 10;
  auto cons = random_conservatives(generator, eos, num_points);
  const auto recover =
      [&cons, &eos,
       &options](const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>
                     prims) {
        return grmhd::ValenciaDivClean::primitive_from_conservative_on_device<
            tmpl::list<Schemes::NewmanHamlin>, false>(
            Kokkos::DefaultExecutionSpace{},
            make_not_null(
                &get<hydro::Tags::RestMassDensity<DataVector>>(*prims)),
            make_not_null(
                &get<hydro::Tags::ElectronFraction<DataVector>>(*prims)),
            make_not_null(
                &get<hydro::Tags::SpecificInternalEnergy<DataVector>>(*prims)),
            make_not_null(
                &get<hydro::Tags::SpatialVelocity<DataVector, 3>>(*prims)),
            make_not_null(
                &get<hydro::Tags::MagneticField<DataVector, 3>>(*prims)),
            make_not_null(
                &get<hydro::Tags::DivergenceCleaningField<DataVector>>(*prims)),
            make_not_null(&get<hydro::Tags::LorentzFactor<DataVector>>(*prims)),
            make_not_null(&get<hydro::Tags::Pressure<DataVector>>(*prims)),
            make_not_null(&get<hydro::Tags::Temperature<DataVector>>(*prims)),
            cons.tilde_d, cons.tilde_ye, cons.tilde_tau, cons.tilde_s,
            cons.tilde_b, cons.tilde_phi, cons.spatial_metric,
            cons.inv_spatial_metric, cons.sqrt_det_spatial_metric, eos,
            options);
      };
  Variables<hydro::grmhd_tags<DataVector>> expected{num_points, 0.5};
  CHECK(recover(make_not_null(&expected)));
  get(cons.tilde_tau)[3] = -10.0;
  get(cons.tilde_tau)[7] = -10.0;
  Variables<hydro::grmhd_tags<DataVector>> prims{num_points, 0.5};
  CHECK_FALSE(recover(make_not_null(&prims)));
  const auto& rest_mass_density =
      get(get<hydro::Tags::RestMassDensity<DataVector>>(prims));
  const auto& expected_rest_mass_density =
      get(get<hydro::Tags::RestMassDensity<DataVector>>(expected));
  for (size_t i = 0; i < num_points; ++i) {
    CAPTURE(i);
    if (i == 3 or i == 7) {
      CHECK(std::isnan(rest_mass_density[i]));
    } else {
      CHECK(rest_mass_density[i] == expected_rest_mass_density[i]);
    }
  }
}

// With `ErrorOnFailure`, a failure on the device is an error with the number of
// failed grid points and the variables at the first one
template <typename Eos>
void test_error_on_failure(const gsl::not_null<std::mt19937*> generator,
                           const Eos& eos) {
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions options{
      1.e-12, 1.e-12, 1.e4};
  auto cons = random_conservatives(generator, eos, 10);
  // Every fifth grid point is atmosphere, which doesn't fail
  get(cons.tilde_tau) = -10.0;
  Variables<hydro::grmhd_tags<DataVector>> prims{10, 0.5};
  CHECK_THROWS_WITH(
      (grmhd::ValenciaDivClean::primitive_from_conservative_on_device<
          tmpl::list<Schemes::NewmanHamlin>, true>(
          Kokkos::DefaultExecutionSpace{},
          make_not_null(&get<hydro::Tags::RestMassDensity<DataVector>>(prims)),
          make_not_null(&get<hydro::Tags::ElectronFraction<DataVector>>(prims)),
          make_not_null(
              &get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prims)),
          make_not_null(
              &get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prims)),
          make_not_null(&get<hydro::Tags::MagneticField<DataVector, 3>>(prims)),
          make_not_null(
              &get<hydro::Tags::DivergenceCleaningField<DataVector>>(prims)),
          make_not_null(&get<hydro::Tags::LorentzFactor<DataVector>>(prims)),
          make_not_null(&get<hydro::Tags::Pressure<DataVector>>(prims)),
          make_not_null(&get<hydro::Tags::Temperature<DataVector>>(prims)),
          cons.tilde_d, cons.tilde_ye, cons.tilde_tau, cons.tilde_s,
          cons.tilde_b, cons.tilde_phi, cons.spatial_metric,
          cons.inv_spatial_metric, cons.sqrt_det_spatial_metric, eos, options)),
      Catch::Matchers::ContainsSubstring(
          "All primitive inversion schemes failed on the device at 8 grid "
          "points. The first is s = 1.") and
          Catch::Matchers::ContainsSubstring("previous_lorentz_factor = 5.0"));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.GrMhd.ValenciaDivClean.PrimitiveFromConservative.Kokkos",
    "[Unit][GrMhd]") {
  MAKE_GENERATOR(generator);
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
      ideal_fluid{EquationsOfState::IdealFluid<true>{4.0 / 3.0, 0.0}};
  const EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>
      polytrope{EquationsOfState::PolytropicFluid<true>{10.0, 2.0}};
  // A minimum temperature gives a nonzero lower bound of the specific internal
  // energy
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
      ideal_fluid_with_minimum_temperature{
          EquationsOfState::IdealFluid<true>{4.0 / 3.0, 1.e-3}};
  for (const auto* eos :
       {static_cast<const EquationsOfState::EquationOfState<true, 3>*>(
            &ideal_fluid),
        static_cast<const EquationsOfState::EquationOfState<true, 3>*>(
            &ideal_fluid_with_minimum_temperature),
        static_cast<const EquationsOfState::EquationOfState<true, 3>*>(
            &polytrope)}) {
    // The schemes and lists of schemes used by the subcell actions
    test_recovery<tmpl::list<Schemes::KastaunEtAl>>(make_not_null(&generator),
                                                    *eos);
    test_recovery<tmpl::list<Schemes::NewmanHamlin>>(make_not_null(&generator),
                                                     *eos);
    test_recovery<tmpl::list<Schemes::PalenzuelaEtAl>>(
        make_not_null(&generator), *eos);
    test_recovery<NewmanHamlinThenPalenzuelaEtAl>(make_not_null(&generator),
                                                  *eos);
    test_recovery<KastaunThenNewmanThenPalenzuela>(make_not_null(&generator),
                                                   *eos);
  }
  test_nan_at_failures(make_not_null(&generator), ideal_fluid);
  test_error_on_failure(make_not_null(&generator), ideal_fluid);
}
