// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <type_traits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDeviceCapable.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
namespace detail {
/*!
 * \brief The functions of `HybridEos`, implemented with the cold EOS
 * `ColdEos`, which is a reference to the cold EOS on the host or its
 * `DeviceView` on the device (see `HybridEos::device_view`).
 */
template <typename ColdEos>
struct HybridEosFunctions {
  static constexpr size_t thermodynamic_dim = 2;
  static constexpr bool is_relativistic =
      std::decay_t<ColdEos>::is_relativistic;

  ColdEos cold_eos;
  double thermal_adiabatic_index;
  double min_temperature;

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    using std::max;
    return Scalar<DataType>{
        get(cold_eos.pressure_from_density(rest_mass_density)) +
        get(rest_mass_density) * (thermal_adiabatic_index - 1.0) *
            max((get(specific_internal_energy) -
                 get(cold_eos.specific_internal_energy_from_density(
                     rest_mass_density))),
                0.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_enthalpy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_enthalpy) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    using std::max;
    if constexpr (is_relativistic) {
      return Scalar<DataType>{
          (get(cold_eos.pressure_from_density(rest_mass_density)) +
           get(rest_mass_density) * (thermal_adiabatic_index - 1.0) *
               max((get(specific_enthalpy) - 1.0 -
                    get(cold_eos.specific_internal_energy_from_density(
                        rest_mass_density))),
                   0.0)) /
          thermal_adiabatic_index};
    } else {
      return Scalar<DataType>{
          (get(cold_eos.pressure_from_density(rest_mass_density)) +
           get(rest_mass_density) * (thermal_adiabatic_index - 1.0) *
               max((get(specific_enthalpy) -
                    get(cold_eos.specific_internal_energy_from_density(
                        rest_mass_density))),
                   0.0)) /
          thermal_adiabatic_index};
    }
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_internal_energy_from_density_and_pressure(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& pressure) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    using std::max;
    return Scalar<DataType>{
        get(cold_eos.specific_internal_energy_from_density(rest_mass_density)) +
        1.0 / (thermal_adiabatic_index - 1.0) *
            max((get(pressure) -
                 get(cold_eos.pressure_from_density(rest_mass_density))),
                0.0) /
            get(rest_mass_density)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> temperature_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    using std::max;
    return Scalar<DataType>{
        (thermal_adiabatic_index - 1.0) *
        max((get(specific_internal_energy) -
             get(cold_eos.specific_internal_energy_from_density(
                 rest_mass_density))),
            0.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_entropy_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    using std::max;
    DataType thermal_specific_internal_energy =
        max(get(specific_internal_energy) -
                get(cold_eos.specific_internal_energy_from_density(
                    rest_mass_density)),
            0.0);
    if constexpr (std::is_same_v<DataType, double>) {
      return Scalar<double>{specific_entropy_from_density_and_thermal_energy(
          get(rest_mass_density), thermal_specific_internal_energy)};
    } else {
      auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);
      for (size_t i = 0; i < get(result).size(); ++i) {
        get(result)[i] = specific_entropy_from_density_and_thermal_energy(
            get(rest_mass_density)[i], thermal_specific_internal_energy[i]);
      }
      return result;
    }
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_entropy_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    if constexpr (std::is_same_v<DataType, double>) {
      return Scalar<double>{specific_entropy_from_density_and_thermal_energy(
          get(rest_mass_density),
          get(temperature) / (thermal_adiabatic_index - 1.0))};
    } else {
      auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);
      for (size_t i = 0; i < get(result).size(); ++i) {
        get(result)[i] = specific_entropy_from_density_and_thermal_energy(
            get(rest_mass_density)[i],
            get(temperature)[i] / (thermal_adiabatic_index - 1.0));
      }
      return result;
    }
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_internal_energy_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    return Scalar<DataType>{
        get(cold_eos.specific_internal_energy_from_density(rest_mass_density)) +
        get(temperature) / (thermal_adiabatic_index - 1.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> chi_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    return Scalar<DataType>{
        get(cold_eos.chi_from_density(rest_mass_density)) +
        (thermal_adiabatic_index - 1.0) *
            (get(specific_internal_energy) -
             get(cold_eos.specific_internal_energy_from_density(
                 rest_mass_density)) -
             get(cold_eos.pressure_from_density(rest_mass_density)) /
                 get(rest_mass_density))};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  kappa_times_p_over_rho_squared_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<ColdEos>);
    using std::max;
    return Scalar<DataType>{
        (thermal_adiabatic_index - 1.0) *
            get(cold_eos.pressure_from_density(rest_mass_density)) /
            get(rest_mass_density) +
        square(thermal_adiabatic_index - 1.0) *
            max((get(specific_internal_energy) -
                 get(cold_eos.specific_internal_energy_from_density(
                     rest_mass_density))),
                0.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double rest_mass_density_lower_bound() const {
    ASSERT_DEVICE_CAPABLE(std::decay_t<ColdEos>);
    return cold_eos.rest_mass_density_lower_bound();
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double rest_mass_density_upper_bound() const {
    ASSERT_DEVICE_CAPABLE(std::decay_t<ColdEos>);
    return cold_eos.rest_mass_density_upper_bound();
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double specific_internal_energy_lower_bound(
      const double rest_mass_density) const {
    ASSERT_DEVICE_CAPABLE(std::decay_t<ColdEos>);
    return get(cold_eos.specific_internal_energy_from_density(
               Scalar<double>{rest_mass_density})) +
           (min_temperature) / (thermal_adiabatic_index - 1.0);
  }

  KOKKOS_FUNCTION double specific_internal_energy_upper_bound(
      const double /*rest_mass_density*/) const {
    return std::numeric_limits<double>::max();
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double specific_enthalpy_lower_bound() const {
    ASSERT_DEVICE_CAPABLE(std::decay_t<ColdEos>);
    return cold_eos.specific_enthalpy_lower_bound() +
           (thermal_adiabatic_index * min_temperature) /
               (thermal_adiabatic_index - 1.0);
  }

  KOKKOS_FUNCTION double temperature_lower_bound() const {
    return min_temperature;
  }

  /// Like `EquationOfState::temperature_upper_bound`, which this EOS doesn't
  /// override
  KOKKOS_FUNCTION double temperature_upper_bound() const {
    return std::numeric_limits<double>::max();
  }

  KOKKOS_FUNCTION double specific_entropy_from_density_and_thermal_energy(
      const double rest_mass_density,
      const double thermal_specific_internal_energy) const {
    using std::max;
    const double floored_thermal_specific_internal_energy =
        max(min_temperature / (thermal_adiabatic_index - 1.0),
            thermal_specific_internal_energy);
    ASSERT(floored_thermal_specific_internal_energy > 0.0,
           "The entropy is only well defined for positive nonzero "
           "temperatures. Try setting the minimum temperature to be greater "
           "than 0.");
    return log(floored_thermal_specific_internal_energy /
               pow(rest_mass_density, thermal_adiabatic_index - 1.0)) /
           (thermal_adiabatic_index - 1.0);
  }
};
}  // namespace detail

/*!
 * \ingroup EquationsOfStateGroup
 *
 * \brief Hybrid equation of state combining a barotropic EOS for cold
 * (zero-temperature) part with a simple thermal part
 *
 * The hybrid equation of state:
 *
 * \f[
 * p = p_{cold}(\rho) + \rho (\Gamma_{th}-1) (\epsilon - \epsilon_{cold}(\rho))
 * \f]
 *
 * where \f$p\f$ is the pressure, \f$\rho\f$ is the rest mass density,
 * \f$\epsilon\f$ is the specific internal energy, \f$p_{cold}\f$ and
 * \f$\epsilon_{cold}\f$ are the pressure and specific internal energy evaluated
 * using the cold EOS, and \f$\Gamma_{th}\f$ is the adiabatic index for the
 * thermal part.
 *
 * The temperature \f$T\f$ is defined as
 *
 * \f[
 * T = (\Gamma_{th} - 1) (\epsilon - \epsilon_{cold})
 * \f]
 *
 * This is the amount of internal energy above that of the cold EOS.
 *
 * For a hybrid EOS with the cold EOS being a polytrope we have
 *
 * \f{align}
 * p &= \kappa \rho^\Gamma + \rho T
 *     =\kappa\rho^\Gamma+\rho
 *     (\Gamma-1)\left(\epsilon-\frac{K\rho^{\Gamma-1}}{\Gamma-1}\right), \\
 * T&=(\Gamma-1)\left(\epsilon-\frac{K\rho^{\Gamma-1}}{\Gamma-1}\right), \\
 * \epsilon &= \frac{1}{(\Gamma-1)}\frac{p}{\rho}
 * \f}
 */
template <typename ColdEquationOfState>
class HybridEos
    : public EquationOfState<ColdEquationOfState::is_relativistic, 2> {
 public:
  static constexpr size_t thermodynamic_dim = 2;
  static constexpr bool is_relativistic = ColdEquationOfState::is_relativistic;

  struct ColdEos {
    using type = ColdEquationOfState;
    static constexpr Options::String help = {"Cold equation of state"};
    static std::string name() {
      return pretty_type::short_name<ColdEquationOfState>();
    }
  };

  struct ThermalAdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {"Adiabatic index Gamma_th"};
  };

  struct MinTemperature {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Minimum temperature. "
        "This value must be non-negative."};
  };

  static constexpr Options::String help = {
      "A hybrid equation of state combining a cold EOS with a simple thermal "
      "part.  The pressure is related to the rest mass density by "
      " p = p_cold(rho) + rho * (Gamma_th - 1) * (epsilon - "
      "epsilon_cold(rho)), where p is the pressure, rho is the rest mass "
      "density, epsilon is the specific internal energy, p_cold and "
      "epsilon_cold are the pressure and specific internal energy evaluated "
      "using the cold EOS and Gamma_th is the adiabatic index for the thermal "
      "part."};

  using options = tmpl::list<ColdEos, ThermalAdiabaticIndex, MinTemperature>;

  HybridEos() = default;
  HybridEos(const HybridEos&) = default;
  HybridEos& operator=(const HybridEos&) = default;
  HybridEos(HybridEos&&) = default;
  HybridEos& operator=(HybridEos&&) = default;
  ~HybridEos() override = default;

  HybridEos(ColdEquationOfState cold_eos, double thermal_adiabatic_index,
            double min_temperature = 0.0);

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(HybridEos, 2)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<is_relativistic, 2>), HybridEos);

  std::unique_ptr<EquationOfState<is_relativistic, 2>> get_clone()
      const override;

  std::unique_ptr<EquationOfState<is_relativistic, 3>> promote_to_3d_eos()
      const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return false; }

  bool operator==(const HybridEos<ColdEquationOfState>& rhs) const;

  bool operator!=(const HybridEos<ColdEquationOfState>& rhs) const;

  bool is_equal(const EquationOfState<is_relativistic, 2>& rhs) const override;

  static std::string name() {
    return "HybridEos(" + pretty_type::name<ColdEquationOfState>() + ")";
  }

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override {
    return cold_eos_.rest_mass_density_lower_bound();
  }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return cold_eos_.rest_mass_density_upper_bound();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_lower_bound(
      const double rest_mass_density) const override {
    return functions().specific_internal_energy_lower_bound(rest_mass_density);
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double rest_mass_density) const override {
    return functions().specific_internal_energy_upper_bound(rest_mass_density);
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return functions().specific_enthalpy_lower_bound();
  }

  /// The lower bound of the temperature that is valid for this EOS.
  /// Non-zero lower bound could be set to impose floor on the specific
  /// internal energy.
  double temperature_lower_bound() const override {
    return functions().temperature_lower_bound();
  }

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override { return cold_eos_.baryon_mass(); }

  /// The parameters and functions of this EOS in a trivially copyable type
  /// that can be passed to Kokkos kernels (see `IdealFluid::DeviceView`).
  /// Only available if the cold EOS has a `device_view()`.
  template <typename T = ColdEquationOfState>
    requires has_device_view_v<T>
  auto device_view() const {
    using ColdDeviceView = decltype(cold_eos_.device_view());
    return detail::HybridEosFunctions<ColdDeviceView>{
        cold_eos_.device_view(), thermal_adiabatic_index_, min_temperature_};
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(2)

  // The functions of this EOS, implemented with the cold EOS
  detail::HybridEosFunctions<const ColdEquationOfState&> functions() const {
    return {cold_eos_, thermal_adiabatic_index_, min_temperature_};
  }

  ColdEquationOfState cold_eos_;
  double thermal_adiabatic_index_ =
      std::numeric_limits<double>::signaling_NaN();
  double min_temperature_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename ColdEquationOfState>
PUP::able::PUP_ID EquationsOfState::HybridEos<ColdEquationOfState>::my_PUP_ID =
    0;
/// \endcond
}  // namespace EquationsOfState
