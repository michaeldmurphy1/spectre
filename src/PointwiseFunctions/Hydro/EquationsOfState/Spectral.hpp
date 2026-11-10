// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
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
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDeviceCapable.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
namespace Spectral_detail {
/// The largest number of coefficients a `Spectral::DeviceView` holds
constexpr size_t max_number_of_coefficients = 16;
/// The largest number of tabulated specific internal energies a
/// `Spectral::DeviceView` holds
constexpr size_t max_table_size = 128;

/// Values with a fixed capacity, so they are trivially copyable
template <size_t Capacity>
struct FixedArray {
  std::array<double, Capacity> values{};
  size_t number_of_values = 0;

  KOKKOS_FUNCTION size_t size() const { return number_of_values; }
  KOKKOS_FUNCTION double operator[](const size_t i) const {
    return gsl::at(values, i);
  }
};

/*!
 * \brief The functions of `Spectral`, for its parameters on the host (with
 * `Array` a reference to a `std::vector`) or on the device (with `Array` a
 * `FixedArray`, see `Spectral::DeviceView`).
 *
 * \details `evaluate_polynomial` sums the terms in the same order as
 * `::evaluate_polynomial`, so both give the same results.
 */
template <typename CoefficientArray, typename TableArray>
struct SpectralFunctions {
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = true;

  double reference_density;
  double reference_pressure;
  CoefficientArray gamma_coefficients;
  CoefficientArray integral_coefficients;
  double x_max;
  double gamma_of_x_max;
  double integral_of_gamma_of_x_max;
  TableArray table_of_specific_energies;
  std::array<double, 3> quadrature_weights;
  std::array<double, 3> quadrature_points;

  KOKKOS_FUNCTION double rest_mass_density_lower_bound() const { return 0.0; }
  KOKKOS_FUNCTION double rest_mass_density_upper_bound() const {
    return std::numeric_limits<double>::max();
  }
  KOKKOS_FUNCTION double specific_enthalpy_lower_bound() const { return 1.0; }
  KOKKOS_FUNCTION double specific_internal_energy_lower_bound() const {
    return 0.0;
  }
  KOKKOS_FUNCTION double specific_internal_energy_upper_bound() const {
    return std::numeric_limits<double>::max();
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> pressure_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    return pointwise(rest_mass_density, [this](const double density) {
      return pressure_from_density_at_point(density);
    });
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> rest_mass_density_from_enthalpy(
      const Scalar<DataType>& specific_enthalpy) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    return pointwise(specific_enthalpy, [this](const double enthalpy) {
      return rest_mass_density_from_enthalpy_at_point(enthalpy);
    });
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_internal_energy_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    return pointwise(rest_mass_density, [this](const double density) {
      return specific_internal_energy_from_density_at_point(density);
    });
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> chi_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    return pointwise(rest_mass_density, [this](const double density) {
      return chi_from_density_at_point(density);
    });
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> kappa_times_p_over_rho_squared_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    return detail::zero_like(rest_mass_density);
  }

  /// Like `EquationOfState::specific_entropy_from_density`, which this EOS
  /// doesn't override
  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType>
  KOKKOS_FUNCTION Scalar<DataType> specific_entropy_from_density(
      const Scalar<DataType>& rest_mass_density) const {
    ASSERT_DEVICE_CAPABLE(DataType, std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    return detail::zero_like(rest_mass_density);
  }

  // `f` at each grid point
  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <class DataType, typename F>
  KOKKOS_FUNCTION static Scalar<DataType> pointwise(const Scalar<DataType>& x,
                                                    const F& f) {
    ASSERT_DEVICE_CAPABLE(DataType);
    if constexpr (std::is_same_v<DataType, double>) {
      return Scalar<double>{f(get(x))};
    } else {
      auto result = make_with_value<Scalar<DataType>>(x, 0.0);
      for (size_t i = 0; i < get(result).size(); ++i) {
        get(result)[i] = f(get(x)[i]);
      }
      return result;
    }
  }

  // Horner's method, like `::evaluate_polynomial`
  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION static double evaluate_polynomial(
      const CoefficientArray& coefficients, const double x) {
    ASSERT_DEVICE_CAPABLE(std::decay_t<CoefficientArray>);
    double result = 0.0;
    for (size_t i = coefficients.size(); i > 0; --i) {
      result = result * x + coefficients[i - 1];
    }
    return result;
  }

  KOKKOS_FUNCTION double gamma(const double x) const {
    return evaluate_polynomial(gamma_coefficients, x);
  }

  KOKKOS_FUNCTION double integral_of_gamma(const double x) const {
    return evaluate_polynomial(integral_coefficients, x) * x;
  }

  KOKKOS_FUNCTION double pressure_from_log_density(const double x) const {
    const double integral_of_gamma_of_x =
        x <= 0.0
            ? integral_coefficients[0] * x
            : (x < x_max
                   ? integral_of_gamma(x)
                   : integral_of_gamma_of_x_max + gamma_of_x_max * (x - x_max));
    return reference_pressure * exp(integral_of_gamma_of_x);
  }

  KOKKOS_FUNCTION double pressure_from_density_at_point(
      const double rest_mass_density) const {
    const double x = log(rest_mass_density / reference_density);
    return pressure_from_log_density(x);
  }

  KOKKOS_FUNCTION double chi_from_density_at_point(
      const double rest_mass_density) const {
    const double x = log(rest_mass_density / reference_density);
    const double P = pressure_from_log_density(x);
    const double chi = P / rest_mass_density *
                       (x <= 0.0 ? integral_coefficients[0]
                                 : (x < x_max ? gamma(x) : gamma(x_max)));
    return chi;
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double specific_internal_energy_from_density_at_point(
      const double rest_mass_density) const {
    ASSERT_DEVICE_CAPABLE(std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    const double x = log(rest_mass_density / reference_density);
    if (x <= 0.) {
      return reference_pressure / reference_density /
             (gamma_coefficients[0] - 1.0) *
             exp((gamma_coefficients[0] - 1.0) * x);
    }
    const size_t n_points_epsilon = table_of_specific_energies.size();
    const double delta_x = x_max / (n_points_epsilon - 1.0);
    double specific_energy = 0.0;
    if (x >= x_max) {
      // Analytic integral at constant gamma_coefficient above x_max
      specific_energy = table_of_specific_energies[n_points_epsilon - 1];
      specific_energy += pressure_from_log_density(x_max) / reference_density *
                         exp(-x_max) *
                         std::expm1((gamma_of_x_max - 1.0) * (x - x_max)) /
                         (gamma_of_x_max - 1.0);
      return specific_energy;
    } else {
      const auto table_index = static_cast<size_t>(floor(x / delta_x));
      const double x0 = delta_x * table_index;
      specific_energy = table_of_specific_energies[table_index];
      // Gaussian quadrature integral
      for (size_t q = 0; q < quadrature_weights.size(); q++) {
        const double xp =
            (1.0 + gsl::at(quadrature_points, q)) * (x - x0) / 2. + x0;
        const double xm =
            (1.0 - gsl::at(quadrature_points, q)) * (x - x0) / 2. + x0;
        const double Pp = pressure_from_log_density(xp);
        const double Pm = pressure_from_log_density(xm);
        specific_energy += gsl::at(quadrature_weights, q) * (x - x0) / 2. *
                           (Pp * exp(-xp) + Pm * exp(-xm)) / reference_density;
      }
      return specific_energy;
    }
  }

  KOKKOS_FUNCTION double specific_enthalpy_from_density_at_point(
      const double rest_mass_density) const {
    return 1.0 +
           pressure_from_density_at_point(rest_mass_density) /
               rest_mass_density +
           specific_internal_energy_from_density_at_point(rest_mass_density);
  }

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  KOKKOS_FUNCTION double rest_mass_density_from_enthalpy_at_point(
      const double specific_enthalpy) const {
    ASSERT_DEVICE_CAPABLE(std::decay_t<CoefficientArray>,
                          std::decay_t<TableArray>);
    const double reference_enthalpy =
        specific_enthalpy_from_density_at_point(reference_density);
    const double upper_density = reference_density * exp(x_max);
    const double enthalpy_of_x_max =
        specific_enthalpy_from_density_at_point(upper_density);
    if (specific_enthalpy <= reference_enthalpy) {
      double rest_mass_density =
          (specific_enthalpy - 1.0) * (gamma_coefficients[0] - 1.0) /
          gamma_coefficients[0] *
          pow(reference_density, gamma_coefficients[0]) / reference_pressure;
      rest_mass_density =
          pow(rest_mass_density, 1.0 / (gamma_coefficients[0] - 1.0));
      return rest_mass_density;
    } else if (specific_enthalpy >= enthalpy_of_x_max) {
      // Above maximum density, we also have an analytical expression
      // (h-1-eps_max)*(rho_max/P_max)+1/(Gamma_max-1) = Gamma_max /
      // (Gamma_max-1) * exp((Gamma_max-1)*(x-x_max)) [Can be derived by
      // combining expressions for epsilon(x) and P(x) with x = log(rho)]
      const size_t n_points_epsilon = table_of_specific_energies.size();
      const double specific_internal_energy_of_x_max =
          table_of_specific_energies[n_points_epsilon - 1];
      const double pressure_of_x_max = pressure_from_log_density(x_max);
      double x_target =
          ((specific_enthalpy - 1.0 - specific_internal_energy_of_x_max) *
               upper_density / pressure_of_x_max +
           1.0 / (gamma_of_x_max - 1.0)) *
          (gamma_of_x_max - 1.0) / gamma_of_x_max;
      x_target = log(x_target) / (gamma_of_x_max - 1.0) + x_max;
      return reference_density * exp(x_target);
    } else {
      // Root-finding appropriate between reference density and maximum
      // density. We can use x=0 and x=x_max as bounds
      const auto f = [this, &specific_enthalpy](const double density) {
        return this->specific_enthalpy_from_density_at_point(density) -
               specific_enthalpy;
      };
      return RootFinder::toms748(f, reference_density, upper_density, 1.0e-14,
                                 1.0e-15)
          .value();
    }
  }
};
}  // namespace Spectral_detail

/*!
 * \ingroup EquationsOfStateGroup
 * \brief A spectral equation of state
 *
 * This equation of state is determined as a function of \f$x =
 * \ln(\rho/\rho_0)\f$ where \f$\rho\f$ is the rest mass density and
 * \f$\rho_0\f$ is the provided reference density.  The adiabatic
 * index \f$\Gamma(x)\f$ is defined such that
 * \f{equation}{
 * \frac{d \ln p}{dx} = \Gamma(x) = \sum_{n=0}^N
 * \gamma_n x^n
 * \f}
 *
 * for the set of spectral coefficinets \f$\gamma_n\f$ when
 * \f$0 < x < x_u = \ln(\rho_u/\rho_0)\f$, where \f$\rho_u\f$ is the provided
 * upper density.
 *
 * For \f$ x < 0 \f$, \f$ \Gamma(x) = \gamma_0 \f$.
 *
 * For \f$ x > x_u \f$, \f$ \Gamma(x) = \Gamma(x_u) \f$
 *
 *
 */
class Spectral : public EquationOfState<true, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = true;

  struct ReferenceDensity {
    using type = double;
    static constexpr Options::String help = {"Reference density rho_0"};
    static double lower_bound() { return 0.0; }
  };

  struct ReferencePressure {
    using type = double;
    static constexpr Options::String help = {"Reference pressure p_0"};
    static double lower_bound() { return 0.0; }
  };

  struct Coefficients {
    using type = std::vector<double>;
    static constexpr Options::String help = {"Spectral coefficients gamma_i"};
  };

  struct UpperDensity {
    using type = double;
    static constexpr Options::String help = {"Upper density rho_u"};
    static double lower_bound() { return 0.0; }
  };

  static constexpr Options::String help = {
      "A spectral equation of state.  Defining x = log(rho/rho_0), Gamma(x) = "
      "Sum_i gamma_i x^i, then the pressure is determined from d(log P)/dx = "
      "Gamma(x) for x > 0.  For x < 0 the EOS is a polytrope with "
      "Gamma(x)=Gamma(0).  For x > x_u = log(rho_u/rho_0), Gamma(x) = "
      "Gamma(x_u).\n"
      "To get smooth equations of state, it is recommended that the second "
      "and third supplied coefficient should be 0. It is up to the user to "
      "choose coefficients that are physically reasonable, e.g. that "
      "satisfy causality."};

  using options = tmpl::list<ReferenceDensity, ReferencePressure, Coefficients,
                             UpperDensity>;

  Spectral() = default;
  Spectral(const Spectral&) = default;
  Spectral& operator=(const Spectral&) = default;
  Spectral(Spectral&&) = default;
  Spectral& operator=(Spectral&&) = default;
  ~Spectral() override = default;

  Spectral(double reference_density, double reference_pressure,
           std::vector<double> coefficients, double upper_density);

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Spectral, 1)

  std::unique_ptr<EquationOfState<true, 1>> get_clone() const override;

  std::unique_ptr<EquationOfState<true, 3>> promote_to_3d_eos() const override;

  std::unique_ptr<EquationOfState<true, 2>> promote_to_2d_eos() const override;

  bool operator==(const Spectral& rhs) const;

  bool operator!=(const Spectral& rhs) const;

  bool is_equal(const EquationOfState<true, 1>& rhs) const override;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<true, 1>), Spectral);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override { return 0.0; }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override { return 1.0; }

  /// The lower bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_lower_bound() const override { return 0.0; }

  /// The upper bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::default_baryon_mass;
  }

  /// The parameters and functions of this EOS in a trivially copyable type
  /// that can be passed to Kokkos kernels (see `IdealFluid::DeviceView`). It
  /// holds at most `Spectral_detail::max_number_of_coefficients` coefficients
  /// and `Spectral_detail::max_table_size` tabulated specific internal
  /// energies; more are an error.
  using DeviceView = Spectral_detail::SpectralFunctions<
      Spectral_detail::FixedArray<Spectral_detail::max_number_of_coefficients>,
      Spectral_detail::FixedArray<Spectral_detail::max_table_size>>;

  /// The `DeviceView` of this EOS
  DeviceView device_view() const;

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1)

  // The functions of this EOS, which the `DeviceView` shares
  Spectral_detail::SpectralFunctions<const std::vector<double>&,
                                     const std::vector<double>&>
  functions() const;

  double reference_density_ = std::numeric_limits<double>::signaling_NaN();
  double reference_pressure_ = std::numeric_limits<double>::signaling_NaN();
  std::vector<double> integral_coefficients_{};
  std::vector<double> gamma_coefficients_{};
  double x_max_ = std::numeric_limits<double>::signaling_NaN();
  double gamma_of_x_max_ = std::numeric_limits<double>::signaling_NaN();
  double integral_of_gamma_of_x_max_ =
      std::numeric_limits<double>::signaling_NaN();
  std::vector<double> table_of_specific_energies_{};
  // Information for Gaussian quadrature
  size_t number_of_quadrature_coefs_ =
      std::numeric_limits<size_t>::signaling_NaN();
  std::vector<double> quadrature_weights_{};
  std::vector<double> quadrature_points_{};
};

}  // namespace EquationsOfState
