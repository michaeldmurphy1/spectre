// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <cmath>
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
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Equation of state for an ideal fluid
 *
 * An ideal fluid equation of state:
 *
 * \f[
 * p = \rho \epsilon(\gamma-1)
 * \f]
 *
 * where \f$\rho\f$ is the rest mass density, \f$\epsilon\f$ is the specific
 * internal energy, and \f$\gamma\f$ is the adiabatic index.
 *
 * The temperature \f$T\f$ is defined as
 *
 * \f[
 * T = (\gamma - 1) \epsilon
 * \f]
 */
template <bool IsRelativistic>
class IdealFluid : public EquationOfState<IsRelativistic, 2> {
 public:
  static constexpr size_t thermodynamic_dim = 2;
  static constexpr bool is_relativistic = IsRelativistic;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {"Adiabatic index gamma"};
  };

  struct MinTemperature {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Minimum temperature. "
        "This value must be non-negative."};
  };

  static constexpr Options::String help = {
      "An ideal fluid equation of state.\n"
      "The pressure is related to the rest mass density by p = rho * epsilon * "
      "(gamma - 1), where p is the pressure, rho is the rest mass density, "
      "epsilon is the specific internal energy, and gamma is the adiabatic "
      "index.\n"
      "The temperature T is defined as T=(gamma-1) * epsilon."};

  using options = tmpl::list<AdiabaticIndex, MinTemperature>;

  IdealFluid() = default;
  IdealFluid(const IdealFluid&) = default;
  IdealFluid& operator=(const IdealFluid&) = default;
  IdealFluid(IdealFluid&&) = default;
  IdealFluid& operator=(IdealFluid&&) = default;
  ~IdealFluid() override = default;

  explicit IdealFluid(double adiabatic_index, double min_temperature = 0.0);

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(IdealFluid, 2)

  std::unique_ptr<EquationOfState<IsRelativistic, 2>> get_clone()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 3>> promote_to_3d_eos()
      const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return false; }

  bool operator==(const IdealFluid<IsRelativistic>& rhs) const;

  bool operator!=(const IdealFluid<IsRelativistic>& rhs) const;

  bool is_equal(const EquationOfState<IsRelativistic, 2>& rhs) const override;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 2>), IdealFluid);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override { return 0.0; }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  /// If non-zero lower bound for temperature is provided, then the lower bound
  /// for specific internal energy is also non-zero accordingly.
  double specific_internal_energy_lower_bound(
      const double /* rest_mass_density */) const override {
    return (min_temperature_) / (adiabatic_index_ - 1.0);
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      double rest_mass_density) const override;

  /// The lower bound of the specific enthalpy that is valid for this EOS
  /// If non-zero lower bound for temperature is provided, then the lower bound
  /// for specific internal enthalpy is also non-zero accordingly.
  double specific_enthalpy_lower_bound() const override {
    return IsRelativistic ? 1.0 + (adiabatic_index_ * min_temperature_) /
                                      (adiabatic_index_ - 1.0)
                          : (adiabatic_index_ * min_temperature_) /
                                (adiabatic_index_ - 1.0);
  }

  /// The lower bound of the temperature that is valid for this EOS.
  /// Non-zero lower bound could be set to impose floor on the specific
  /// internal energy.
  double temperature_lower_bound() const override { return min_temperature_; }

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::default_baryon_mass;
  }

  /*!
   * \brief The parameters and functions of this EOS in a trivially copyable
   * type that can be passed to Kokkos kernels.
   *
   * Has the same functions as the EOS, but without the virtual interface. The
   * EOS is implemented with these functions.
   */
  struct DeviceView;

  /// The `DeviceView` of this EOS
  DeviceView device_view() const;

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(2)

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double min_temperature_ = std::numeric_limits<double>::signaling_NaN();
};

template <bool IsRelativistic>
struct IdealFluid<IsRelativistic>::DeviceView {
  double adiabatic_index;
  double min_temperature;

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy) const {
    return Scalar<DataType>{get(rest_mass_density) *
                            get(specific_internal_energy) *
                            (adiabatic_index - 1.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_enthalpy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_enthalpy) const {
    if constexpr (IsRelativistic) {
      return Scalar<DataType>{get(rest_mass_density) *
                              (get(specific_enthalpy) - 1.0) *
                              (adiabatic_index - 1.0) / adiabatic_index};
    } else {
      return Scalar<DataType>{get(rest_mass_density) * get(specific_enthalpy) *
                              (adiabatic_index - 1.0) / adiabatic_index};
    }
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_entropy_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy) const {
    if constexpr (std::is_same_v<DataType, double>) {
      return Scalar<double>{specific_entropy(get(rest_mass_density),
                                             get(specific_internal_energy))};
    } else {
      auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);
      for (size_t i = 0; i < get(result).size(); ++i) {
        get(result)[i] = specific_entropy(get(rest_mass_density)[i],
                                          get(specific_internal_energy)[i]);
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
    return specific_entropy_from_density_and_energy(
        rest_mass_density,
        specific_internal_energy_from_density_and_temperature(rest_mass_density,
                                                              temperature));
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_internal_energy_from_density_and_pressure(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& pressure) const {
    return Scalar<DataType>{1.0 / (adiabatic_index - 1.0) * get(pressure) /
                            get(rest_mass_density)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> temperature_from_density_and_energy(
      const Scalar<DataType>& /*rest_mass_density*/,
      const Scalar<DataType>& specific_internal_energy) const {
    return Scalar<DataType>{(adiabatic_index - 1.0) *
                            get(specific_internal_energy)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_internal_energy_from_density_and_temperature(
      const Scalar<DataType>& /*rest_mass_density*/,
      const Scalar<DataType>& temperature) const {
    return Scalar<DataType>{get(temperature) / (adiabatic_index - 1.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> chi_from_density_and_energy(
      const Scalar<DataType>& /*rest_mass_density*/,
      const Scalar<DataType>& specific_internal_energy) const {
    return Scalar<DataType>{get(specific_internal_energy) *
                            (adiabatic_index - 1.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  kappa_times_p_over_rho_squared_from_density_and_energy(
      const Scalar<DataType>& /*rest_mass_density*/,
      const Scalar<DataType>& specific_internal_energy) const {
    return Scalar<DataType>{square(adiabatic_index - 1.0) *
                            get(specific_internal_energy)};
  }

  /// The lower bound of the specific internal energy
  KOKKOS_FUNCTION double specific_internal_energy_lower_bound() const {
    return min_temperature / (adiabatic_index - 1.0);
  }

 private:
  KOKKOS_FUNCTION double specific_entropy(
      const double rest_mass_density,
      const double specific_internal_energy) const {
    using std::max;
    const double floored_specific_internal_energy =
        max(specific_internal_energy_lower_bound(), specific_internal_energy);
    ASSERT(
        floored_specific_internal_energy > 0.0,
        "The entropy is only well defined for positive nonzero temperatures. "
        "Try setting the minimum temperature to be greater than 0.");
    return log(floored_specific_internal_energy /
               pow(rest_mass_density, adiabatic_index - 1.0)) /
           (adiabatic_index - 1.0);
  }
};

template <bool IsRelativistic>
auto IdealFluid<IsRelativistic>::device_view() const -> DeviceView {
  return DeviceView{adiabatic_index_, min_temperature_};
}

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::IdealFluid<IsRelativistic>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
