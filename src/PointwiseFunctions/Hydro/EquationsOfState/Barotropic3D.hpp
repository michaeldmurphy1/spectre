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

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief A 3D equation of state representing a barotropic fluid.
 *
 *
 * The equation of state takes the form
 *
 * \f[
 * p = p (\rho , T, Y_e) = p(\rho, 0, Y_e= Y_{e, \beta})
 * \f]
 *
 * where \f$\rho\f$ is the rest mass density, \f$T\f$  the
 * temperature , and \f$Y_e\f$ the electron fraction. The temperature and
 * electron fraction are not used, so evaluating this EoS at any arbtirary
 * temeperature or electron fraction is equivalent to evaluating it at
 * zero temperature and in beta equalibrium.
 */
namespace detail {
// The functions of `Barotropic3D` in terms of the functions of the 1D EOS
// `underlying`, which is either the EOS itself (on the host) or its
// `DeviceView`. This way the same code implements both.
template <typename Underlying>
struct Barotropic3DFunctions {
  Underlying underlying;

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*temperature*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.pressure_from_density(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*specific_internal_energy*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.pressure_from_density(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> temperature_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*specific_internal_energy*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return zero_like(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_entropy_from_density_and_energy(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*specific_internal_energy*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.specific_entropy_from_density(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_entropy_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*temperature*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.specific_entropy_from_density(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  specific_internal_energy_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*temperature*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return underlying.specific_internal_energy_from_density(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType>
  sound_speed_squared_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*temperature*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    // We have to do this to avoid dividing by zero
    const DataType enthalpy_density =
        get(underlying.pressure_from_density(rest_mass_density)) +
        (1.0 + get(underlying.specific_internal_energy_from_density(
                   rest_mass_density))) *
            get(rest_mass_density);
    return Scalar<DataType>{
        get(rest_mass_density) *
        get(underlying.chi_from_density(rest_mass_density)) / enthalpy_density};
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> kappa_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*temperature*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    // Barotropic EOS: p = p(rho), so kappa = dp/deps|_rho = 0.
    return zero_like(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> zeta_from_density_and_temperature(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& /*temperature*/,
      const Scalar<DataType>& /*electron_fraction*/) const {
    return zero_like(rest_mass_density);
  }
};
}  // namespace detail

template <typename ColdEquilEos>
class Barotropic3D : public EquationOfState<ColdEquilEos::is_relativistic, 3> {
 public:
  static constexpr size_t thermodynamic_dim = 3;
  static constexpr bool is_relativistic = ColdEquilEos::is_relativistic;

  static std::string name() {
    return "Barotropic3D(" + pretty_type::name<ColdEquilEos>() + ")";
  }
  static constexpr Options::String help = {
      "An 3D EoS which is independent of electron fraction and temperature. "
      "Contains an underlying 1D EoS which is dependent only "
      "on rest mass density."};
  struct UnderlyingEos {
    using type = ColdEquilEos;
    static std::string name() {
      return pretty_type::short_name<ColdEquilEos>();
    }
    static constexpr Options::String help{
        "The underlying Eos which is being represented as a "
        "3D Eos.  Must be a 1D EoS"};
  };

  using options = tmpl::list<UnderlyingEos>;

  Barotropic3D() = default;
  Barotropic3D(const Barotropic3D&) = default;
  Barotropic3D& operator=(const Barotropic3D&) = default;
  Barotropic3D(Barotropic3D&&) = default;
  Barotropic3D& operator=(Barotropic3D&&) = default;
  ~Barotropic3D() override = default;

  explicit Barotropic3D(const ColdEquilEos& underlying_eos)
      : underlying_eos_(underlying_eos){};

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Barotropic3D, 3)

  std::unique_ptr<EquationOfState<ColdEquilEos::is_relativistic, 3>> get_clone()
      const override;

  bool is_equal(const EquationOfState<ColdEquilEos::is_relativistic, 3>& rhs)
      const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return true; }

  /// \brief Returns `true` if the EOS is in beta-equilibrium
  bool is_equilibrium() const override { return false; }

  bool operator==(const Barotropic3D<ColdEquilEos>& rhs) const;

  bool operator!=(const Barotropic3D<ColdEquilEos>& rhs) const;
  /// @{
  /*!
   * Computes the electron fraction in beta-equilibrium \f$Y_e^{\rm eq}\f$ from
   * the rest mass density \f$\rho\f$ and the temperature \f$T\f$.
   */
  Scalar<double> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& temperature) const override {
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
      SINGLE_ARG(EquationOfState<ColdEquilEos::is_relativistic, 3>),
      Barotropic3D);

  /// The lower bound of the electron fraction that is valid for this EOS
  double electron_fraction_lower_bound() const override { return 0.0; }

  /// The upper bound of the electron fraction that is valid for this EOS
  double electron_fraction_upper_bound() const override { return 1.0; }

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override {
    return underlying_eos_.rest_mass_density_lower_bound();
  }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return underlying_eos_.rest_mass_density_upper_bound();
  }

  /// The lower bound of the temperature that is valid for this EOS
  double temperature_lower_bound() const override { return 0.0; }

  /// The upper bound of the temperature that is valid for this EOS
  double temperature_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$ and electron fraction \f$Y_e\f$
  double specific_internal_energy_lower_bound(
      const double /*rest_mass_density*/,
      const double /*electron_fraction*/) const override {
    return underlying_eos_.specific_internal_energy_lower_bound();
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double /*rest_mass_density*/,
      const double /*electron_fraction*/) const override {
    return underlying_eos_.specific_internal_energy_upper_bound();
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return underlying_eos_.specific_enthalpy_lower_bound();
  }

  /// The baryon mass for this EoS
  double baryon_mass() const override { return underlying_eos_.baryon_mass(); }

  /// The parameters and functions of this EOS in a trivially copyable type
  /// that can be passed to Kokkos kernels (see `IdealFluid::DeviceView`).
  /// Only available if the underlying EOS has a `device_view()`.
  template <typename T = ColdEquilEos>
    requires has_device_view_v<T>
  auto device_view() const {
    using UnderlyingDeviceView = decltype(underlying_eos_.device_view());
    return detail::Barotropic3DFunctions<UnderlyingDeviceView>{
        underlying_eos_.device_view()};
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(3)

  // The functions of this EOS, implemented with the underlying EOS
  detail::Barotropic3DFunctions<const ColdEquilEos&> functions() const {
    return {underlying_eos_};
  }

  ColdEquilEos underlying_eos_;
};
/// \cond
template <typename ColdEquilEos>
PUP::able::PUP_ID EquationsOfState::Barotropic3D<ColdEquilEos>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
