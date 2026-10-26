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
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief A 3D equation of state representing a fluid in compositional
 * equalibrium.
 *
 *
 * The equation of state takes the form
 *
 * \f[
 * p = p (T, rho, Y_e) = p(T, rho, Y_e= Y_{e, \beta})
 * \f]
 *
 * where \f$\rho\f$ is the rest mass density and  \f$T\f$ is the
 * temperaturee;   \f$Y_e\f$  the electron fraction,  is not
 * used, and therefore this evaluating this EoS at any arbtirary
 * electron fraction is equivalent to evaluating it  in beta equalibrium
 *
 */
namespace detail {
// The functions of `Equilibrium3D` in terms of the functions of the 2D EOS
// `underlying`, which is either the EOS itself (on the host) or its
// `DeviceView`. This way the same code implements both.
template <typename Underlying>
struct Equilibrium3DFunctions {
  static constexpr size_t thermodynamic_dim = 3;
  static constexpr bool is_relativistic =
      std::remove_cvref_t<Underlying>::is_relativistic;

  Underlying underlying;

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.pressure_from_density_and_energy(
        rest_mass_density,
        underlying.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature));
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> temperature_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.temperature_from_density_and_energy(
        rest_mass_density, specific_internal_energy);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_entropy_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.specific_entropy_from_density_and_energy(
        rest_mass_density, specific_internal_energy);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_entropy_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.specific_entropy_from_density_and_energy(
        rest_mass_density,
        underlying.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature));
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_internal_energy_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.specific_internal_energy_from_density_and_temperature(
        rest_mass_density, temperature);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  sound_speed_squared_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature,
      const Scalar<DataType>& /*electron_fraction*/) const {
    const Scalar<DataType> specific_internal_energy =
        underlying.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature);
    const DataType pressure = get(underlying.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    const DataType enthalpy_density =
        pressure +
        (1.0 + get(specific_internal_energy)) * get(rest_mass_density);
    // Cold part plus temperature dependent part, see the 3D EoS documentation
    // for expression.
    return Scalar<DataType>{
        get(rest_mass_density) / enthalpy_density *
        (get(underlying.chi_from_density_and_energy(rest_mass_density,
                                                    specific_internal_energy)) +
         get(underlying.kappa_times_p_over_rho_squared_from_density_and_energy(
             rest_mass_density, specific_internal_energy)))};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> kappa_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature,
      const Scalar<DataType>& /*electron_fraction*/) const {
    // The wrapped 2D EOS exposes kappa * p / rho^2 rather than kappa itself.
    // Undo that here so the 3D interface returns kappa = dp/deps directly.
    const Scalar<DataType> specific_internal_energy =
        underlying.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature);
    const Scalar<DataType> pressure =
        underlying.pressure_from_density_and_energy(rest_mass_density,
                                                    specific_internal_energy);
    const Scalar<DataType> kappa_p_over_rho_squared =
        underlying.kappa_times_p_over_rho_squared_from_density_and_energy(
            rest_mass_density, specific_internal_energy);
    return Scalar<DataType>{get(kappa_p_over_rho_squared) / get(pressure) *
                            square(get(rest_mass_density))};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> zeta_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*temperature*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return zero_like(rest_mass_density);
  }

  KOKKOS_FUNCTION double electron_fraction_lower_bound() const { return 0.0; }

  KOKKOS_FUNCTION double electron_fraction_upper_bound() const { return 1.0; }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double rest_mass_density_lower_bound() const {
    return underlying.rest_mass_density_lower_bound();
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double rest_mass_density_upper_bound() const {
    return underlying.rest_mass_density_upper_bound();
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double specific_internal_energy_lower_bound(
      const double rest_mass_density,
      const double /*electron_fraction*/) const {
    return underlying.specific_internal_energy_lower_bound(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double specific_internal_energy_upper_bound(
      const double rest_mass_density,
      const double /*electron_fraction*/) const {
    return underlying.specific_internal_energy_upper_bound(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double specific_enthalpy_lower_bound() const {
    return underlying.specific_enthalpy_lower_bound();
  }
};
}  // namespace detail

template <typename EquilEos>
class Equilibrium3D : public EquationOfState<EquilEos::is_relativistic, 3> {
 public:
  static constexpr size_t thermodynamic_dim = 3;
  static constexpr bool is_relativistic = EquilEos::is_relativistic;
  static std::string name() {
    return "Equilibrium3D(" + pretty_type::name<EquilEos>() + ")";
  }
  static constexpr Options::String help = {
      "An 3D EoS which is independent of electron fraction. "
      "Contains an underlying 2D EoS which is dependent only "
      "on rest mass density and temperature/internal energy."};

  struct UnderlyingEos {
    using type = EquilEos;
    static std::string name() { return pretty_type::short_name<EquilEos>(); }
    static constexpr Options::String help{
        "The underlying Eos which is being represented as a "
        "3D Eos.  Must be a 2D EoS"};
  };

  using options = tmpl::list<UnderlyingEos>;

  Equilibrium3D() = default;
  Equilibrium3D(const Equilibrium3D&) = default;
  Equilibrium3D& operator=(const Equilibrium3D&) = default;
  Equilibrium3D(Equilibrium3D&&) = default;
  Equilibrium3D& operator=(Equilibrium3D&&) = default;
  ~Equilibrium3D() override = default;

  explicit Equilibrium3D(const EquilEos& underlying_eos)
      : underlying_eos_(underlying_eos){};

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Equilibrium3D, 3)

  std::unique_ptr<EquationOfState<EquilEos::is_relativistic, 3>> get_clone()
      const override;

  bool is_equal(
      const EquationOfState<EquilEos::is_relativistic, 3>& rhs) const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return false; }

  /// \brief Returns `true` if the EOS is in beta-equilibrium
  bool is_equilibrium() const override { return true; }

  bool operator==(const Equilibrium3D<EquilEos>& rhs) const;

  bool operator!=(const Equilibrium3D<EquilEos>& rhs) const;
  /// @{
  /*!
   * Computes the electron fraction in beta-equilibrium \f$Y_e^{\rm eq}\f$ from
   * the rest mass density \f$\rho\f$ and the temperature \f$T\f$.
   */
  Scalar<double> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& temperature) const override{
    return underlying_eos_
        .equilibrium_electron_fraction_from_density_temperature(
            rest_mass_density, temperature);
  }

  Scalar<DataVector> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& temperature) const override {
    return underlying_eos_
        .equilibrium_electron_fraction_from_density_temperature(
            rest_mass_density, temperature);
  }
  /// @}
  //

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<EquilEos::is_relativistic, 3>), Equilibrium3D);

  /// The lower bound of the electron fraction that is valid for this EOS
  double electron_fraction_lower_bound() const override {
    return functions().electron_fraction_lower_bound();
  }

  /// The upper bound of the electron fraction that is valid for this EOS
  double electron_fraction_upper_bound() const override {
    return functions().electron_fraction_upper_bound();
  }

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override {
    return functions().rest_mass_density_lower_bound();
  }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return functions().rest_mass_density_upper_bound();
  }

  /// The lower bound of the temperature that is valid for this EOS
  double temperature_lower_bound() const override {
    return underlying_eos_.temperature_lower_bound();
  }

  /// The upper bound of the temperature that is valid for this EOS
  double temperature_upper_bound() const override {
    return underlying_eos_.temperature_upper_bound();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$ and electron fraction \f$Y_e\f$
  double specific_internal_energy_lower_bound(
      const double rest_mass_density,
      const double electron_fraction) const override {
    return functions().specific_internal_energy_lower_bound(rest_mass_density,
                                                            electron_fraction);
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double rest_mass_density,
      const double electron_fraction) const override {
    return functions().specific_internal_energy_upper_bound(rest_mass_density,
                                                            electron_fraction);
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return functions().specific_enthalpy_lower_bound();
  }

  /// The baryon mass for this EoS
  double baryon_mass() const override { return underlying_eos_.baryon_mass(); }

  /// The parameters and functions of this EOS in a trivially copyable type
  /// that can be passed to Kokkos kernels (see `IdealFluid::DeviceView`).
  /// Only available if the underlying EOS has a `device_view()`.
  template <typename T = EquilEos>
    requires has_device_view_v<T>
  auto device_view() const {
    using UnderlyingDeviceView = decltype(underlying_eos_.device_view());
    return detail::Equilibrium3DFunctions<UnderlyingDeviceView>{
        underlying_eos_.device_view()};
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(3)

  // The functions of this EOS, implemented with the underlying EOS
  detail::Equilibrium3DFunctions<const EquilEos&> functions() const {
    return {underlying_eos_};
  }

  EquilEos underlying_eos_;
};
/// \cond
template <typename EquilEos>
PUP::able::PUP_ID EquationsOfState::Equilibrium3D<EquilEos>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
