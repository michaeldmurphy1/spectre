// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace ConstraintDamping {
/*!
 * \brief A constant plus a sum of up to three Gaussians with fixed centers,
 * \f$f(x) = C + \sum_k A_k \exp(-|x - c_k|^2 s_k)\f$.
 *
 * \details This is a damping function at a fixed time (see
 * `DampingFunction::as_gaussian_sum`): all supported damping functions are of
 * this form once their functions of time are evaluated. It is trivially
 * copyable, so it can be copied to the device (e.g. in a buffer of parameters
 * that are updated every step) and evaluated in Kokkos kernels. The terms are
 * added in the same order as by the damping functions, so the values are the
 * same.
 */
template <size_t Dim>
struct GaussianSum {
  struct Gaussian {
    /// \f$A_k\f$
    double amplitude;
    /// \f$s_k\f$, the square of the inverse width (times the expansion factor)
    double exponent_factor;
    /// \f$c_k\f$
    std::array<double, Dim> center;
  };
  static constexpr size_t maximum_number_of_gaussians = 3;

  /// \f$C\f$
  double constant;
  size_t number_of_gaussians;
  /// Only the first `number_of_gaussians` are used
  std::array<Gaussian, maximum_number_of_gaussians> gaussians;

  /// The value at the point `x`
  KOKKOS_FUNCTION double operator()(const std::array<double, Dim>& x) const {
    double result = constant;
    for (size_t k = 0; k < number_of_gaussians; ++k) {
      const Gaussian& gaussian = gaussians[k];
      std::array<double, Dim> centered_coords{};
      for (size_t i = 0; i < Dim; ++i) {
        centered_coords[i] = x[i] - gaussian.center[i];
      }
      // As `dot_product`
      double squared_distance = centered_coords[0] * centered_coords[0];
      for (size_t i = 1; i < Dim; ++i) {
        squared_distance += centered_coords[i] * centered_coords[i];
      }
      result += gaussian.amplitude *
                exp(-squared_distance * gaussian.exponent_factor);
    }
    return result;
  }
};
}  // namespace ConstraintDamping
