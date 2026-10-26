// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAlHydro.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

// The recovery schemes in `OrderedListOfPrimitiveRecoverySchemes` (and
// `KastaunEtAlHydro`) must be defined (by including their `.tpp` files) where
// these functions are instantiated.

namespace grmhd::ValenciaDivClean::PrimitiveFromConservative_detail {
// Whether the 3D EOS (on the host or its `DeviceView`) is barotropic
template <typename EosType>
constexpr bool eos_is_barotropic =
    tt::is_a_v<EquationsOfState::Barotropic3D, EosType> or
    tt::is_a_v<EquationsOfState::detail::Barotropic3DFunctions, EosType>;

// Tries the recovery `Scheme` unless `primitive_data` is already set
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <typename Scheme, bool EnforcePhysicality, typename EosType>
KOKKOS_FUNCTION void try_scheme(
    const gsl::not_null<
        std::optional<PrimitiveRecoverySchemes::PrimitiveRecoveryData>*>
        primitive_data,
    const double initial_pressure, const double tau,
    const double momentum_density_squared,
    const double momentum_density_dot_magnetic_field,
    const double magnetic_field_squared,
    const double rest_mass_density_times_lorentz_factor,
    const double electron_fraction, const EosType& equation_of_state,
    const PrimitiveFromConservativeOptions& options) {
  if (not primitive_data->has_value()) {
    *primitive_data = Scheme::template apply<EnforcePhysicality>(
        initial_pressure, tau, momentum_density_squared,
        momentum_density_dot_magnetic_field, magnetic_field_squared,
        rest_mass_density_times_lorentz_factor, electron_fraction,
        equation_of_state, options);
  }
}

// Tries the recovery `Schemes` in order until one succeeds
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <bool EnforcePhysicality, typename... Schemes, typename EosType>
KOKKOS_FUNCTION std::optional<PrimitiveRecoverySchemes::PrimitiveRecoveryData>
try_schemes(tmpl::list<Schemes...> /*meta*/, const double initial_pressure,
            const double tau, const double momentum_density_squared,
            const double momentum_density_dot_magnetic_field,
            const double magnetic_field_squared,
            const double rest_mass_density_times_lorentz_factor,
            const double electron_fraction, const EosType& equation_of_state,
            const PrimitiveFromConservativeOptions& options) {
  std::optional<PrimitiveRecoverySchemes::PrimitiveRecoveryData>
      primitive_data{};
  (try_scheme<Schemes, EnforcePhysicality>(
       make_not_null(&primitive_data), initial_pressure, tau,
       momentum_density_squared, momentum_density_dot_magnetic_field,
       magnetic_field_squared, rest_mass_density_times_lorentz_factor,
       electron_fraction, equation_of_state, options),
   ...);
  return primitive_data;
}

/*
 * Recovers the primitive variables at a grid point from the quantities derived
 * from the conservative variables in `PrimitiveFromConservative`. The
 * `pressure` is the initial guess and is overwritten by the result. Returns
 * `false` if all schemes failed, in which case only the `electron_fraction`
 * has been changed.
 *
 * The EOS is the EOS on the host or its `DeviceView`.
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool EnforcePhysicality, bool UseHydroOptimization, typename EosType>
KOKKOS_FUNCTION bool recover_at_point(
    const gsl::not_null<double*> rest_mass_density,
    const gsl::not_null<double*> electron_fraction,
    const gsl::not_null<double*> specific_internal_energy,
    const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> spatial_velocity,
    const gsl::not_null<double*> lorentz_factor,
    const gsl::not_null<double*> pressure, const double tilde_d,
    const double tilde_ye, const double tau,
    const double momentum_density_squared,
    const double momentum_density_dot_magnetic_field,
    const double magnetic_field_squared,
    const double rest_mass_density_times_lorentz_factor,
    const double sqrt_det_spatial_metric,
    const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
    const tnsr::I<double, 3, Frame::Inertial>& tilde_s_upper,
    const EosType& equation_of_state,
    const PrimitiveFromConservativeOptions& options) {
  using std::max;
  using std::min;
  constexpr bool is_barotropic = eos_is_barotropic<EosType>;
  const double cutoff_d = options.cutoff_d_for_inversion();
  const double floor_d = options.density_when_skipping_inversion();

  *electron_fraction =
      min(equation_of_state.electron_fraction_upper_bound(),
          max(tilde_ye / tilde_d,
              equation_of_state.electron_fraction_lower_bound()));

  std::optional<PrimitiveRecoverySchemes::PrimitiveRecoveryData>
      primitive_data{};
  // Quick exit from inversion in low-density regions where we will
  // apply atmosphere corrections anyways.
  if (rest_mass_density_times_lorentz_factor < cutoff_d) {
    double specific_energy_at_point =
        equation_of_state.specific_internal_energy_lower_bound(
            floor_d, *electron_fraction);
    if constexpr (is_barotropic) {
      specific_energy_at_point =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      Scalar<double>{floor_d}, Scalar<double>{0.0},
                      Scalar<double>{0.1}));
    }
    const double pressure_at_point =
        get(equation_of_state.pressure_from_density_and_energy(
            Scalar<double>{floor_d}, Scalar<double>{specific_energy_at_point},
            Scalar<double>{*electron_fraction}));
    const double enthalpy_density_at_point =
        floor_d + specific_energy_at_point * floor_d + pressure_at_point;
    primitive_data = PrimitiveRecoverySchemes::PrimitiveRecoveryData{
        floor_d,
        1.0,
        pressure_at_point,
        specific_energy_at_point,
        enthalpy_density_at_point,
        *electron_fraction};
  } else if (UseHydroOptimization and
             (magnetic_field_squared <
              100.0 * std::numeric_limits<double>::epsilon() * tau)) {
    // Not in atmosphere, and the magnetic field is negligible
    primitive_data = try_schemes<EnforcePhysicality>(
        tmpl::list<PrimitiveRecoverySchemes::KastaunEtAlHydro>{}, *pressure,
        tau, momentum_density_squared, momentum_density_dot_magnetic_field,
        magnetic_field_squared, rest_mass_density_times_lorentz_factor,
        *electron_fraction, equation_of_state, options);
  } else {
    primitive_data = try_schemes<EnforcePhysicality>(
        OrderedListOfPrimitiveRecoverySchemes{}, *pressure, tau,
        momentum_density_squared, momentum_density_dot_magnetic_field,
        magnetic_field_squared, rest_mass_density_times_lorentz_factor,
        *electron_fraction, equation_of_state, options);
  }

  if (not primitive_data.has_value()) {
    return false;
  }
  *rest_mass_density = primitive_data->rest_mass_density;
  // if we set the Lorentz factor to 1.0 in atmosphere, then
  // we also need to set the spatial_velocity accordingly.
  if (primitive_data->lorentz_factor == 1.0) {
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity->get(i) = 0.0;
    }
  } else {
    const double coefficient_of_b =
        momentum_density_dot_magnetic_field /
        (primitive_data->rho_h_w_squared *
         (primitive_data->rho_h_w_squared + magnetic_field_squared));
    const double coefficient_of_s =
        1.0 / (sqrt_det_spatial_metric *
               (primitive_data->rho_h_w_squared + magnetic_field_squared));
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity->get(i) = coefficient_of_b * magnetic_field.get(i) +
                                 coefficient_of_s * tilde_s_upper.get(i);
    }
  }
  *lorentz_factor = primitive_data->lorentz_factor;
  *pressure = primitive_data->pressure;
  if constexpr (not is_barotropic) {
    *specific_internal_energy = primitive_data->specific_internal_energy;
  }
  return true;
}
}  // namespace grmhd::ValenciaDivClean::PrimitiveFromConservative_detail
