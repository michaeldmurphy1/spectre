// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/VariableFixing/FixToAtmosphere.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace VariableFixing {
template <size_t Dim>
template <size_t ThermodynamicDim, typename EosType>
KOKKOS_FUNCTION void FixToAtmosphere<Dim>::DeviceView::fix_at_point(
    const gsl::not_null<double*> rest_mass_density,
    const gsl::not_null<double*> specific_internal_energy,
    const gsl::not_null<tnsr::I<double, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<double*> lorentz_factor,
    const gsl::not_null<double*> pressure,
    const gsl::not_null<double*> temperature, const double electron_fraction,
    const tnsr::ii<double, Dim, Frame::Inertial>& spatial_metric,
    const EosType& equation_of_state, bool* const root_find_failed) const {
  if (UNLIKELY(*rest_mass_density < density_cutoff)) {
    set_density_to_atmosphere<ThermodynamicDim>(
        rest_mass_density, specific_internal_energy, temperature, pressure,
        electron_fraction, equation_of_state);
  }
  if (limit_velocity) {
    apply_velocity_limit(spatial_velocity, lorentz_factor, *rest_mass_density,
                         spatial_metric);
  }

  // For 2D & 3D EoS, we also need to limit the temperature / energy
  if constexpr (ThermodynamicDim > 1) {
    bool changed_temperature = false;
    if (const double min_temperature =
            equation_of_state.temperature_lower_bound();
        *temperature < min_temperature) {
      *temperature = min_temperature;
      changed_temperature = true;
    }

    // We probably need a better maximum temperature as well, but this is not
    // as well defined. To be discussed once implementation needs improvement.
    if (const double max_temperature =
            equation_of_state.temperature_upper_bound();
        *temperature > max_temperature) {
      *temperature = max_temperature;
      changed_temperature = true;
    }

    if (limit_kappa) {
      changed_temperature |= apply_kappa_limit<ThermodynamicDim>(
          temperature, *rest_mass_density, electron_fraction, equation_of_state,
          root_find_failed);
      if (root_find_failed != nullptr and *root_find_failed) {
        return;
      }
    }

    if (changed_temperature) {
      if constexpr (ThermodynamicDim == 2) {
        *specific_internal_energy =
            get(equation_of_state
                    .specific_internal_energy_from_density_and_temperature(
                        Scalar<double>{*rest_mass_density},
                        Scalar<double>{*temperature}));
        *pressure = get(equation_of_state.pressure_from_density_and_energy(
            Scalar<double>{*rest_mass_density},
            Scalar<double>{*specific_internal_energy}));
      } else {
        *specific_internal_energy =
            get(equation_of_state
                    .specific_internal_energy_from_density_and_temperature(
                        Scalar<double>{*rest_mass_density},
                        Scalar<double>{*temperature},
                        Scalar<double>{electron_fraction}));
        *pressure = get(equation_of_state.pressure_from_density_and_temperature(
            Scalar<double>{*rest_mass_density}, Scalar<double>{*temperature},
            Scalar<double>{electron_fraction}));
      }
    }
  } else {
    (void)root_find_failed;
  }
}

template <size_t Dim>
template <size_t ThermodynamicDim, typename EosType>
KOKKOS_FUNCTION void
FixToAtmosphere<Dim>::DeviceView::set_density_to_atmosphere(
    const gsl::not_null<double*> rest_mass_density,
    const gsl::not_null<double*> specific_internal_energy,
    const gsl::not_null<double*> temperature,
    const gsl::not_null<double*> pressure, const double electron_fraction,
    const EosType& equation_of_state) const {
  const Scalar<double> atmosphere_density{density_of_atmosphere};
  *rest_mass_density = get(atmosphere_density);
  *temperature = equation_of_state.temperature_lower_bound();

  if constexpr (ThermodynamicDim == 1) {
    (void)electron_fraction;
    *pressure =
        get(equation_of_state.pressure_from_density(atmosphere_density));
    *specific_internal_energy =
        get(equation_of_state.specific_internal_energy_from_density(
            atmosphere_density));
  } else {
    const Scalar<double> atmosphere_temperature{*temperature};
    if constexpr (ThermodynamicDim == 2) {
      (void)electron_fraction;
      *specific_internal_energy =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      atmosphere_density, atmosphere_temperature));
      *pressure = get(equation_of_state.pressure_from_density_and_energy(
          atmosphere_density, Scalar<double>{*specific_internal_energy}));
    } else {
      *specific_internal_energy =
          get(equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
                      Scalar<double>{*rest_mass_density},
                      Scalar<double>{*temperature},
                      Scalar<double>{electron_fraction}));
      *pressure = get(equation_of_state.pressure_from_density_and_temperature(
          Scalar<double>{*rest_mass_density}, Scalar<double>{*temperature},
          Scalar<double>{electron_fraction}));
    }
  }
}

template <size_t Dim>
KOKKOS_FUNCTION void FixToAtmosphere<Dim>::DeviceView::apply_velocity_limit(
    const gsl::not_null<tnsr::I<double, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<double*> lorentz_factor, const double rest_mass_density,
    const tnsr::ii<double, Dim, Frame::Inertial>& spatial_metric) const {
  if (rest_mass_density > velocity_limiting.transition_density_bound) {
    return;
  }

  if (rest_mass_density < velocity_limiting.atmosphere_density_cutoff) {
    for (size_t i = 0; i < Dim; ++i) {
      spatial_velocity->get(i) = velocity_limiting.atmosphere_max_velocity;
    }
    if (LIKELY(velocity_limiting.atmosphere_max_velocity == 0.0)) {
      *lorentz_factor = 1.0;
      return;
    }
  }

  double magnitude_of_velocity = 0.0;
  for (size_t j = 0; j < Dim; ++j) {
    magnitude_of_velocity += spatial_velocity->get(j) *
                             spatial_velocity->get(j) *
                             spatial_metric.get(j, j);
    for (size_t k = j + 1; k < Dim; ++k) {
      magnitude_of_velocity += 2.0 * spatial_velocity->get(j) *
                               spatial_velocity->get(k) *
                               spatial_metric.get(j, k);
    }
  }
  if (UNLIKELY(rest_mass_density <
               velocity_limiting.atmosphere_density_cutoff)) {
    // Note: magnitude_of_velocity is squared still.
    *lorentz_factor = 1.0 / sqrt(1.0 - magnitude_of_velocity);
    return;
  }
  magnitude_of_velocity = sqrt(magnitude_of_velocity);
  const double scale_factor =
      (rest_mass_density - velocity_limiting.atmosphere_density_cutoff) /
      (velocity_limiting.transition_density_bound -
       velocity_limiting.atmosphere_density_cutoff);
  if (const double max_mag_of_velocity =
          scale_factor * velocity_limiting.near_atmosphere_max_velocity;
      magnitude_of_velocity > max_mag_of_velocity) {
    const double one_over_max_mag_of_velocity = 1.0 / magnitude_of_velocity;
    for (size_t j = 0; j < Dim; ++j) {
      spatial_velocity->get(j) *=
          max_mag_of_velocity * one_over_max_mag_of_velocity;
    }
    *lorentz_factor =
        1.0 / sqrt(1.0 - max_mag_of_velocity * max_mag_of_velocity);
  }
}

template <size_t Dim>
template <size_t ThermodynamicDim, typename EosType>
KOKKOS_FUNCTION bool FixToAtmosphere<Dim>::DeviceView::apply_kappa_limit(
    const gsl::not_null<double*> temperature, const double rest_mass_density,
    const double electron_fraction, const EosType& equation_of_state,
    bool* const root_find_failed) const {
  const KappaLimiting& opts = kappa_limiting;
  double& temp = *temperature;
  const double& density = rest_mass_density;
  const double& y_e = electron_fraction;
  const double min_temperature =
      opts.min_temperature_from_eos
          ? equation_of_state.temperature_lower_bound()
          : opts.min_temperature;

  const auto get_pressure = [&density, &equation_of_state,
                             &y_e](const double local_temperature) -> double {
    if constexpr (ThermodynamicDim == 2) {
      (void)y_e;
      return get(equation_of_state.pressure_from_density_and_energy(
          Scalar<double>{density},
          equation_of_state
              .specific_internal_energy_from_density_and_temperature(
                  Scalar<double>{density}, Scalar<double>{local_temperature})));
    } else {
      return get(equation_of_state.pressure_from_density_and_temperature(
          Scalar<double>{density}, Scalar<double>{local_temperature},
          Scalar<double>{y_e}));
    }
  };

  const auto impl = [&get_pressure, &min_temperature, &temp, &y_e,
                     &root_find_failed](const double local_kappa_max) -> bool {
    const double p_temp = get_pressure(temp);
    const double p_min = get_pressure(min_temperature);
    const double p_max = p_min * local_kappa_max;
    if (p_temp > p_max) {
      if (UNLIKELY((p_temp - p_max) * (p_min - p_max) > 0.0)) {
        if (root_find_failed != nullptr) {
          *root_find_failed = true;
          return false;
        }
        ERROR(
            "The root for the pressure function while applying the kappa "
            "limiting strategy was not bound.\n"
            "T_min="
            << min_temperature << "\np_min=" << p_min << "\nT=" << temp
            << "\np_temp=" << p_temp << "\np_max=" << p_max << "\nYe=" << y_e);
      }
      if (temp - min_temperature > 1.0e-13) {
        const auto f = [&get_pressure, &p_max](const double local_temperature) {
          return get_pressure(local_temperature) - p_max;
        };
        const auto root =
            RootFinder::toms748(f, min_temperature, temp, 1e-50, 1.0e-13);
        if (root_find_failed != nullptr and not root.has_value()) {
          *root_find_failed = true;
          return false;
        }
        temp = root.value();
      } else {
        temp = 0.5 * (temp + min_temperature);
      }
      return true;
    }
    return false;
  };

  using std::abs;
  using std::min;
  if (density < opts.density_lower_bound) {
    if (abs(temp - min_temperature) > opts.eplison_kappa_minus * abs(temp)) {
      temp = min_temperature;
    }
    return true;
  } else if (density < opts.density_upper_bound) {
    const double kappa_max =
        1.0 + opts.epsilon_kappa_max *
                  min((density - opts.density_lower_bound) /
                          (opts.density_upper_bound - opts.density_lower_bound),
                      1.0);
    return impl(kappa_max);
  } else if (opts.limit_above_density_upper_bound) {
    return impl(1.0 + opts.epsilon_kappa_max);
  }
  return false;
}
}  // namespace VariableFixing
