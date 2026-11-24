// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace EquationsOfState {
namespace detail {
/*!
 * \brief The cold part `ColdEos` of a `HybridEos` at one rest mass density,
 * evaluated once.
 *
 * Provides the functions of the cold EOS that `HybridEosFunctions` calls with
 * the rest mass density, but only at that density.
 */
template <typename ColdEos>
struct ColdEosAtFixedDensity {
  static constexpr bool is_relativistic =
      std::remove_cvref_t<ColdEos>::is_relativistic;

  double rest_mass_density;
  double pressure;
  double specific_internal_energy;
  double chi;

  KOKKOS_FUNCTION static ColdEosAtFixedDensity evaluate(
      const ColdEos& cold_eos, const double rest_mass_density) {
    const Scalar<double> density{rest_mass_density};
    return {rest_mass_density, get(cold_eos.pressure_from_density(density)),
            get(cold_eos.specific_internal_energy_from_density(density)),
            get(cold_eos.chi_from_density(density))};
  }

  KOKKOS_FUNCTION Scalar<double> pressure_from_density(
      const Scalar<double>& density) const {
    assert_density(density);
    return Scalar<double>{pressure};
  }

  KOKKOS_FUNCTION Scalar<double> specific_internal_energy_from_density(
      const Scalar<double>& density) const {
    assert_density(density);
    return Scalar<double>{specific_internal_energy};
  }

  KOKKOS_FUNCTION Scalar<double> chi_from_density(
      const Scalar<double>& density) const {
    assert_density(density);
    return Scalar<double>{chi};
  }

 private:
  KOKKOS_FUNCTION void assert_density(
      [[maybe_unused]] const Scalar<double>& density) const {
    ASSERT(get(density) == rest_mass_density,
           "The cold EOS was evaluated at the rest mass density "
               << rest_mass_density << ", not at " << get(density));
  }
};
}  // namespace detail

/// @{
/*!
 * \brief The device view `equation_of_state` for evaluations at the rest mass
 * density `rest_mass_density` only.
 *
 * For an `Equilibrium3D<HybridEos<ColdEos>>` the cold EOS is evaluated once
 * here, instead of in every function of the view (`sound_speed_squared_from_\
 * density_and_temperature` alone evaluates the cold specific internal energy
 * four times). The results are the same, because the functions of the view
 * combine the same cold values. Other views are returned as they are.
 *
 * All functions of the result must be called with `rest_mass_density`.
 */
template <typename DeviceView>
KOKKOS_FUNCTION const DeviceView& at_fixed_density(
    const DeviceView& equation_of_state, const double /*rest_mass_density*/) {
  return equation_of_state;
}

template <typename ColdEos>
KOKKOS_FUNCTION auto at_fixed_density(
    const detail::Equilibrium3DFunctions<detail::HybridEosFunctions<ColdEos>>&
        equation_of_state,
    const double rest_mass_density)
    -> detail::Equilibrium3DFunctions<
        detail::HybridEosFunctions<detail::ColdEosAtFixedDensity<ColdEos>>> {
  const auto& hybrid = equation_of_state.underlying;
  return {{detail::ColdEosAtFixedDensity<ColdEos>::evaluate(hybrid.cold_eos,
                                                            rest_mass_density),
           hybrid.thermal_adiabatic_index, hybrid.min_temperature}};
}
/// @}
}  // namespace EquationsOfState
