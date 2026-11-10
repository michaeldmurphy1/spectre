// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// The `values` for a `Spectral::DeviceView`
template <size_t Capacity>
EquationsOfState::Spectral_detail::FixedArray<Capacity> to_fixed_array(
    const std::vector<double>& values, const std::string& name) {
  if (values.size() > Capacity) {
    ERROR("The DeviceView of the Spectral EOS holds at most "
          << Capacity << " " << name << ", but the EOS has " << values.size()
          << ". Increase the capacity in Spectral_detail.");
  }
  EquationsOfState::Spectral_detail::FixedArray<Capacity> result{};
  for (size_t i = 0; i < values.size(); ++i) {
    gsl::at(result.values, i) = values[i];
  }
  result.number_of_values = values.size();
  return result;
}

std::vector<double> compute_integral_coefficients(
    const std::vector<double>& gamma_coefficients) {
  std::vector<double> result = gamma_coefficients;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] /= (1.0 + i);
  }
  return result;
}
}  // namespace

namespace EquationsOfState {
Spectral::Spectral(const double reference_density,
                   const double reference_pressure,
                   std::vector<double> coefficients, const double upper_density)
    : reference_density_(reference_density),
      reference_pressure_(reference_pressure),
      integral_coefficients_(compute_integral_coefficients(coefficients)),
      gamma_coefficients_(std::move(coefficients)),
      x_max_(log(upper_density / reference_density)) {
  // Setup table of specific energies
  // Use 6th order Gauss-Legendre quadrature.
  // Weights and collocation points can be calculated following
  // e.g. the gauleg method from Numerical Recipes
  // (Sec. 4.6.1 in 3rd edition). We only need to consider
  // collocation points with x>0, as the other points are
  // just symmetric with respect to the origin.
  quadrature_weights_ = {0.3607615730481386, 0.4679139345726910,
                         0.1713244923791704};
  quadrature_points_ = {0.6612093864662645, 0.2386191860831969,
                        0.9324695142031521};
  number_of_quadrature_coefs_ = quadrature_weights_.size();
  gamma_of_x_max_ = functions().gamma(x_max_);
  integral_of_gamma_of_x_max_ = functions().integral_of_gamma(x_max_);
  const auto n_points_epsilon = static_cast<size_t>(ceil(2.0 * x_max_) + 1.0);
  const double delta_x = x_max_ / (n_points_epsilon - 1.0);
  table_of_specific_energies_.resize(n_points_epsilon);
  table_of_specific_energies_[0] =
      reference_pressure_ / reference_density_ / (gamma_coefficients_[0] - 1.0);
  for (size_t i = 1; i < n_points_epsilon; i++) {
    table_of_specific_energies_[i] = table_of_specific_energies_[i - 1];
    // Use Gaussian quadrature to calculate the specific internal
    // energy at the next point, separated from the old point
    // by delta_x in log(density)
    // From the 1st law of thermodynamics,
    // d(epsilon)/dx = (Pressure)/(reference_density)*exp(-x)
    const double x0 = (i - 1.) * delta_x;
    for (size_t q = 0; q < number_of_quadrature_coefs_; q++) {
      const double xp = x0 + (1.0 + quadrature_points_[q]) * (delta_x / 2.0);
      const double xm = x0 + (1.0 - quadrature_points_[q]) * (delta_x / 2.0);
      table_of_specific_energies_[i] +=
          quadrature_weights_[q] * delta_x / (2.0 * reference_density) *
          (exp(-xp) * functions().pressure_from_log_density(xp) +
           exp(-xm) * functions().pressure_from_log_density(xm));
    }
  }
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(, Spectral, double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(, Spectral, DataVector, 1)

std::unique_ptr<EquationOfState<true, 1>> Spectral::get_clone() const {
  auto clone = std::make_unique<Spectral>(*this);
  return std::unique_ptr<EquationOfState<true, 1>>(std::move(clone));
}

std::unique_ptr<EquationOfState<true, 3>> Spectral::promote_to_3d_eos() const {
  return std::make_unique<Barotropic3D<Spectral>>(*this);
}

std::unique_ptr<EquationOfState<true, 2>> Spectral::promote_to_2d_eos() const {
  return std::make_unique<Barotropic2D<Spectral>>(*this);
}

bool Spectral::operator==(const Spectral& rhs) const {
  return reference_density_ == rhs.reference_density_ and
         reference_pressure_ == rhs.reference_pressure_;
}

bool Spectral::operator!=(const Spectral& rhs) const {
  return not(*this == rhs);
}

bool Spectral::is_equal(const EquationOfState<true, 1>& rhs) const {
  const auto& derived_ptr = dynamic_cast<const Spectral* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

Spectral::Spectral(CkMigrateMessage* msg) : EquationOfState<true, 1>(msg) {}

void Spectral::pup(PUP::er& p) {
  EquationOfState<true, 1>::pup(p);
  p | reference_density_;
  p | reference_pressure_;
  p | integral_coefficients_;
  p | gamma_coefficients_;
  p | x_max_;
  p | gamma_of_x_max_;
  p | integral_of_gamma_of_x_max_;
  p | number_of_quadrature_coefs_;
  p | quadrature_weights_;
  p | quadrature_points_;
  p | table_of_specific_energies_;
}

// this evaluates the power series
// Int_0^x Gamma(xx) dxx = Sum_{n=0}^N \frac{gamma_n}{n+1} xx^{n+1}
Spectral_detail::SpectralFunctions<const std::vector<double>&,
                                   const std::vector<double>&>
Spectral::functions() const {
  ASSERT(quadrature_weights_.size() == 3 and quadrature_points_.size() == 3,
         "Expected 3 quadrature points but got " << quadrature_points_.size());
  return {
      reference_density_,
      reference_pressure_,
      gamma_coefficients_,
      integral_coefficients_,
      x_max_,
      gamma_of_x_max_,
      integral_of_gamma_of_x_max_,
      table_of_specific_energies_,
      {{quadrature_weights_[0], quadrature_weights_[1],
        quadrature_weights_[2]}},
      {{quadrature_points_[0], quadrature_points_[1], quadrature_points_[2]}}};
}

auto Spectral::device_view() const -> DeviceView {
  constexpr size_t max_coefficients =
      Spectral_detail::max_number_of_coefficients;
  constexpr size_t max_table = Spectral_detail::max_table_size;
  const auto host = functions();
  return {
      reference_density_,
      reference_pressure_,
      to_fixed_array<max_coefficients>(gamma_coefficients_, "coefficients"),
      to_fixed_array<max_coefficients>(integral_coefficients_, "coefficients"),
      x_max_,
      gamma_of_x_max_,
      integral_of_gamma_of_x_max_,
      to_fixed_array<max_table>(table_of_specific_energies_,
                                "tabulated specific internal energies"),
      host.quadrature_weights,
      host.quadrature_points};
}

EQUATION_OF_STATE_DELEGATE_IMPLS(, Spectral, 1, functions())

PUP::able::PUP_ID EquationsOfState::Spectral::my_PUP_ID = 0;

}  // namespace EquationsOfState
