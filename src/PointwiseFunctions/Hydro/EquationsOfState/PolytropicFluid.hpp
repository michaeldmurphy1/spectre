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

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Equation of state for a polytropic fluid
 *
 * A polytropic equation of state \f$p=K\rho^{\Gamma}\f$ where \f$K\f$ is the
 * polytropic constant and \f$\Gamma\f$ is the polytropic exponent. The
 * polytropic exponent is related to the polytropic index \f$N_p\f$ by
 * \f$N_p=1/(\Gamma-1)\f$.
 *
 * We also have
 *
 * \f{align}{
 * \epsilon&=\frac{K\rho^{\Gamma-1}}{\Gamma-1}\\
 * h&=1+\epsilon+\frac{p}{\rho}=1+\frac{K\Gamma}{\Gamma-1}\rho^{\Gamma-1} \\
 * T&=0 \\
 * c_s^2&=\frac{\Gamma p}{\rho h}
 * =\frac{\Gamma(\Gamma-1)p}{\rho(\Gamma-1)+\Gamma p}
 * =\left(\frac{1}{\Gamma K\rho^{\Gamma-1}}+\frac{1}{\Gamma-1}\right)^{-1}
 * \f}
 */
template <bool IsRelativistic>
class PolytropicFluid : public EquationOfState<IsRelativistic, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {"Polytropic constant K"};
    static double lower_bound() { return 0.0; }
  };

  struct PolytropicExponent {
    using type = double;
    static constexpr Options::String help = {"Polytropic exponent Gamma"};
    static double lower_bound() { return 1.0; }
  };

  static constexpr Options::String help = {
      "A polytropic fluid equation of state.\n"
      "The pressure is related to the rest mass density by p = K rho ^ Gamma, "
      "where p is the pressure, rho is the rest mass density, K is the "
      "polytropic constant, and Gamma is the polytropic exponent. The "
      "polytropic index N is defined as Gamma = 1 + 1 / N."};

  using options = tmpl::list<PolytropicConstant, PolytropicExponent>;

  PolytropicFluid() = default;
  PolytropicFluid(const PolytropicFluid&) = default;
  PolytropicFluid& operator=(const PolytropicFluid&) = default;
  PolytropicFluid(PolytropicFluid&&) = default;
  PolytropicFluid& operator=(PolytropicFluid&&) = default;
  ~PolytropicFluid() override = default;

  PolytropicFluid(double polytropic_constant, double polytropic_exponent);

  std::unique_ptr<EquationOfState<IsRelativistic, 1>> get_clone()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 3>> promote_to_3d_eos()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 2>> promote_to_2d_eos()
      const override;

  bool is_equal(const EquationOfState<IsRelativistic, 1>& rhs) const override;

  bool operator==(const PolytropicFluid<IsRelativistic>& rhs) const;

  bool operator!=(const PolytropicFluid<IsRelativistic>& rhs) const;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(PolytropicFluid, 1)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 1>), PolytropicFluid);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override {
    return device_view().rest_mass_density_lower_bound();
  }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return device_view().rest_mass_density_upper_bound();
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return device_view().specific_enthalpy_lower_bound();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_lower_bound() const override {
    return device_view().specific_internal_energy_lower_bound();
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_upper_bound() const override {
    return device_view().specific_internal_energy_upper_bound();
  }

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::default_baryon_mass;
  }

  /// \copydoc IdealFluid::DeviceView
  struct DeviceView;

  /// The `DeviceView` of this EOS
  DeviceView device_view() const;

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1)

  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
};

template <bool IsRelativistic>
struct PolytropicFluid<IsRelativistic>::DeviceView {
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  double polytropic_constant;
  double polytropic_exponent;

  KOKKOS_FUNCTION double rest_mass_density_lower_bound() const { return 0.0; }

  KOKKOS_FUNCTION double rest_mass_density_upper_bound() const {
    // this bound comes from the dominant energy condition which implies
    // that the pressure is bounded by the total energy density,
    // i.e. p < e = rho * (1 + eps)
    if (IsRelativistic and polytropic_exponent > 2.0) {
      return pow((polytropic_exponent - 1.0) /
                     (polytropic_constant * (polytropic_exponent - 2.0)),
                 1.0 / (polytropic_exponent - 1.0));
    }
    return std::numeric_limits<double>::max();
  }

  KOKKOS_FUNCTION double specific_enthalpy_lower_bound() const {
    return IsRelativistic ? 1.0 : 0.0;
  }

  KOKKOS_FUNCTION double specific_internal_energy_lower_bound() const {
    return 0.0;
  }

  KOKKOS_FUNCTION double specific_internal_energy_upper_bound() const {
    // this bound comes from the dominant energy condition which implies
    // that the pressure is bounded by the total energy density,
    // i.e. p < e = rho * (1 + eps)
    if (IsRelativistic and polytropic_exponent > 2.0) {
      return 1.0 / (polytropic_exponent - 2.0);
    }
    return std::numeric_limits<double>::max();
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    return Scalar<DataType>{polytropic_constant *
                            pow(get(rest_mass_density), polytropic_exponent)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> rest_mass_density_from_enthalpy(
      const Scalar<DataType>& specific_enthalpy) const {
    if constexpr (IsRelativistic) {
      return Scalar<DataType>{
          pow(((polytropic_exponent - 1.0) /
               (polytropic_constant * polytropic_exponent)) *
                  (get(specific_enthalpy) - 1.0),
              1.0 / (polytropic_exponent - 1.0))};
    } else {
      return Scalar<DataType>{
          pow(((polytropic_exponent - 1.0) /
               (polytropic_constant * polytropic_exponent)) *
                  get(specific_enthalpy),
              1.0 / (polytropic_exponent - 1.0))};
    }
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_internal_energy_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    return Scalar<DataType>{
        polytropic_constant / (polytropic_exponent - 1.0) *
        pow(get(rest_mass_density), polytropic_exponent - 1.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> chi_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    return Scalar<DataType>{
        polytropic_constant * polytropic_exponent *
        pow(get(rest_mass_density), polytropic_exponent - 1.0)};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> kappa_times_p_over_rho_squared_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    return detail::zero_like(rest_mass_density);
  }

  /// Like `EquationOfState::specific_entropy_from_density`, which this EOS
  /// doesn't override
  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_entropy_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    return detail::zero_like(rest_mass_density);
  }
};

template <bool IsRelativistic>
auto PolytropicFluid<IsRelativistic>::device_view() const -> DeviceView {
  return DeviceView{polytropic_constant_, polytropic_exponent_};
}

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::PolytropicFluid<IsRelativistic>::my_PUP_ID =
    0;
/// \endcond
}  // namespace EquationsOfState
