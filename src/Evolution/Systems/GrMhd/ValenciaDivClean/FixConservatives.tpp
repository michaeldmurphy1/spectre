// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>

#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Simd/Simd.hpp"

// The fixing at a grid point, so that the device kernels that fix the
// conservatives inline it (see "Definitions that kernels call" in the GPU
// guide)

namespace grmhd::ValenciaDivClean {
namespace FixConservatives_detail {

// This class codes Eq. (B.34), rewritten as a standard-form
// polynomial in (W - 1) or (W - ((tau/D) - (B^2/D) + 1)) for better
// numerical behavior.
//
// If lower_bound is 0, the implemented function is
// ((B^2/D) / 2 - (tau/D)) (1 + 2 (B^2/D) mu^2 + (B^2/D)^2 mu^2)
// + (W-1) (2 ((B^2/D) - (tau/D)) (1 + (B^2/D) mu^2) + (B^2/D) mu^2 + 1)
// + (W-1)^2 ((B^2/D) - (tau/D) + 3/2 (B^2/D) mu^2 + 2)
// + (W-1)^3
// where mu^2 = (B.S)^2 / (B^2 S^2).  A nice property of this form is
// that, because we've already modified tau to satisfy B.39, its value
// at W-1 = 0 is guaranteed to be negative, even in the presence of
// roundoff error.
//
// If lower_bound is LB = ((tau/D) - (B^2/D)), the implemented function is
// - 1/2 (B^2/D) (1 + mu^2 (tau/D) ((tau/D) + 2))
// + (W-[LB+1]) (1 + (B^2/D) mu^2 + LB ((B^2/D) mu^2 + LB + 2))
// + (W-[LB+1])^2 (2 LB + 3/2 (B^2/D) mu^2 + 2)
// + (W-[LB+1])^3
// This form is only used if LB > 0, so, as in the previous case, it
// is guaranteed to be negative at W-[LB+1] = 0.
template <bool AssumeNonZeroMagneticField, typename T>
class FunctionOfLorentzFactor {
 public:
  KOKKOS_FUNCTION FunctionOfLorentzFactor(
      const T b_squared_over_d, const T tau_over_d,
      [[maybe_unused]] const T normalized_s_dot_b, const T lower_bound) {
    const auto mask = lower_bound == 0.0;
    T sqr_normalized_s_dot_b_times_b_squared_over_d{};
    if constexpr (AssumeNonZeroMagneticField) {
      sqr_normalized_s_dot_b_times_b_squared_over_d =
          square(normalized_s_dot_b) * b_squared_over_d;
    }
    const auto zero_bound_values =
        zero_bound(b_squared_over_d, tau_over_d,
                   sqr_normalized_s_dot_b_times_b_squared_over_d);
    const auto nonzero_bound_values =
        nonzero_bound(b_squared_over_d, tau_over_d, lower_bound,
                      sqr_normalized_s_dot_b_times_b_squared_over_d);
    coefficients_ = {
        {simd::select(mask, zero_bound_values[0], nonzero_bound_values[0]),
         simd::select(mask, zero_bound_values[1], nonzero_bound_values[1]),
         simd::select(mask, zero_bound_values[2], nonzero_bound_values[2]),
         simd::select(mask, zero_bound_values[3], nonzero_bound_values[3])}};
  }

  KOKKOS_FUNCTION T operator()(const T excess_lorentz_factor) const {
    // Horner's method, like `evaluate_polynomial` (which can't be used in a
    // Kokkos kernel)
    return ((coefficients_[3] * excess_lorentz_factor + coefficients_[2]) *
                excess_lorentz_factor +
            coefficients_[1]) *
               excess_lorentz_factor +
           coefficients_[0];
  }

 private:
  KOKKOS_FUNCTION static std::array<T, 4> zero_bound(
      const T& b_squared_over_d, const T& tau_over_d,
      const T& sqr_normalized_s_dot_b_times_b_squared_over_d) {
    if constexpr (AssumeNonZeroMagneticField) {
      const T temp0 = b_squared_over_d - tau_over_d;
      return std::array{
          (0.5 * b_squared_over_d - tau_over_d) *
              (sqr_normalized_s_dot_b_times_b_squared_over_d *
                   (b_squared_over_d + 2.0) +
               1.0),
          2.0 * (sqr_normalized_s_dot_b_times_b_squared_over_d + 1.0) * temp0 +
              sqr_normalized_s_dot_b_times_b_squared_over_d + 1.0,
          temp0 + 1.5 * sqr_normalized_s_dot_b_times_b_squared_over_d + 2.0,
          static_cast<T>(1.0)};
    } else {
      return std::array{-tau_over_d, -2.0 * tau_over_d + 1.0, -tau_over_d + 2.0,
                        static_cast<T>(1.0)};
    }
  }

  KOKKOS_FUNCTION static std::array<T, 4> nonzero_bound(
      const T& b_squared_over_d, const T& tau_over_d, const T& lower_bound,
      const T& sqr_normalized_s_dot_b_times_b_squared_over_d) {
    if constexpr (AssumeNonZeroMagneticField) {
      return std::array{
          -0.5 * (b_squared_over_d +
                  sqr_normalized_s_dot_b_times_b_squared_over_d * tau_over_d *
                      (tau_over_d + 2.0)),
          1.0 + sqr_normalized_s_dot_b_times_b_squared_over_d +
              lower_bound *
                  (2.0 + sqr_normalized_s_dot_b_times_b_squared_over_d +
                   lower_bound),
          2.0 + 1.5 * sqr_normalized_s_dot_b_times_b_squared_over_d +
              2.0 * lower_bound,
          static_cast<T>(1.0)};
    } else {
      return std::array{static_cast<T>(0),
                        1.0 + lower_bound * (2.0 + lower_bound),
                        2.0 + 2.0 * lower_bound, static_cast<T>(1.0)};
    }
  }

  std::array<T, 4> coefficients_;
};
template <bool AssumeNonZeroMagneticField, typename T>
KOKKOS_FUNCTION FunctionOfLorentzFactor<AssumeNonZeroMagneticField, T>
make_function_of_lorentz_factor(T b_squared_over_d, T tau_over_d,
                                T normalized_s_dot_b, T lower_bound) {
  return {b_squared_over_d, tau_over_d, normalized_s_dot_b, lower_bound};
}
}  // namespace FixConservatives_detail

template <typename T>
struct FixConservatives::RootFindFailure {
  T upper_bound{};
  T lower_bound_of_lorentz_factor_minus_one{};
  T s_tilde_squared{};
  T d_tilde{};
  T sqrt_det_g{};
  T tau_tilde{};
  T b_tilde_squared{};
  T b_squared_over_d{};
  T tau_over_d{};
  T normalized_s_dot_b{};
  RootFinder::Toms748Error<T> error{};
};

// WARNING!
// Notation of Foucart is not that of SpECTRE
// SpECTRE           Foucart
// {\tilde D}        \rho_*
// {\tilde \tau}     \tau
// {\tilde S}_k      S_k
// {\tilde B}^k      B^k \sqrt{g}
// \rho              \rho_0
// \gamma_{mn}       g_{mn}
template <bool NonZeroMagneticField, typename T, typename Mask>
KOKKOS_FUNCTION bool FixConservatives::fix_impl(
    const gsl::not_null<T*> d_tilde_ptr, const gsl::not_null<T*> ye_tilde_ptr,
    const gsl::not_null<T*> tau_tilde_ptr,
    const gsl::not_null<T*> s_tilde_rescaling_factor, const T& sqrt_det_g,
    const T& s_tilde_squared, const T& b_tilde_squared_in,
    const T& s_tilde_dot_b_tilde_in, const Mask& completion_mask,
    const gsl::not_null<std::optional<RootFindFailure<T>>*> root_find_failure)
    const {
  using SimdType = T;
  bool needed_fixing = false;
  T& d_tilde = *d_tilde_ptr;
  T& ye_tilde = *ye_tilde_ptr;
  T& tau_tilde = *tau_tilde_ptr;
  *s_tilde_rescaling_factor = SimdType{1.0};

  const double one_over_one_minus_safety_factor_for_magnetic_field =
      1.0 / one_minus_safety_factor_for_magnetic_field_;
  const double one_over_safety_factor_for_momentum_density_cutoff_d =
      1.0 / safety_factor_for_momentum_density_cutoff_d_;

  const SimdType one_over_sqrt_det_g = 1.0 / sqrt_det_g;
  SimdType one_over_d_tilde = 1.0 / d_tilde;
  SimdType rest_mass_density_times_lorentz_factor =
      d_tilde * one_over_sqrt_det_g;

  // Increase electron fraction if necessary
  if (const auto ye_mask = ye_tilde < electron_fraction_cutoff_ * d_tilde;
      simd::any(ye_mask)) {
    needed_fixing = true;
    ye_tilde =
        simd::select(ye_mask, minimum_electron_fraction_ * d_tilde, ye_tilde);
  }

  // Increase mass density if necessary
  if (const auto tilde_d_mask = rest_mass_density_times_lorentz_factor <
                                rest_mass_density_times_lorentz_factor_cutoff_;
      simd::any(tilde_d_mask)) {
    needed_fixing = true;
    d_tilde = simd::select(
        tilde_d_mask,
        minimum_rest_mass_density_times_lorentz_factor_ * sqrt_det_g, d_tilde);
    ye_tilde = d_tilde * one_over_d_tilde * ye_tilde;
    one_over_d_tilde = 1.0 / d_tilde;
    rest_mass_density_times_lorentz_factor = simd::select(
        tilde_d_mask,
        static_cast<SimdType>(minimum_rest_mass_density_times_lorentz_factor_),
        rest_mass_density_times_lorentz_factor);
  }

  // Increase internal energy if necessary
  auto b_tilde_squared = static_cast<SimdType>(0);
  if constexpr (NonZeroMagneticField) {
    b_tilde_squared = b_tilde_squared_in;
    // Equation B.39 of Foucart
    if (const auto tilde_tau_mask =
            b_tilde_squared > one_minus_safety_factor_for_magnetic_field_ * 2. *
                                  tau_tilde * sqrt_det_g;
        simd::any(tilde_tau_mask)) {
      needed_fixing = true;
      tau_tilde =
          simd::select(tilde_tau_mask,
                       0.5 * b_tilde_squared *
                           one_over_one_minus_safety_factor_for_magnetic_field *
                           one_over_sqrt_det_g,
                       tau_tilde);
    }
  } else {
    auto zeroes = static_cast<SimdType>(0);
    if (const auto tilde_tau_mask = tau_tilde < zeroes;
        simd::any(tilde_tau_mask)) {
      needed_fixing = true;
      tau_tilde = simd::select(tilde_tau_mask, zeroes, tau_tilde);
    }
  }

  // Decrease momentum density if necessary
  // Equation B.24 of Foucart
  const SimdType tau_over_d = tau_tilde * one_over_d_tilde;
  // Equation B.23 of Foucart
  const SimdType b_squared_over_d =
      NonZeroMagneticField
          ? b_tilde_squared * one_over_sqrt_det_g * one_over_d_tilde
          : static_cast<SimdType>(0);
  const SimdType s_tilde_dot_b_tilde =
      NonZeroMagneticField ? s_tilde_dot_b_tilde_in : static_cast<SimdType>(0);

  // Equation B.27 of Foucart
  //
  // We avoid division by zero in the SIMD case using a mask and then set the
  // result to zero using the same mask.
  const auto normaled_s_dot_b_mask =
      (b_tilde_squared > 1.e-16 * d_tilde and
       s_tilde_squared > 1.e-16 * square(d_tilde));
  const SimdType normalized_s_dot_b =
      NonZeroMagneticField
          ? simd::select(
                normaled_s_dot_b_mask,
                s_tilde_dot_b_tilde /
                    simd::select(normaled_s_dot_b_mask,
                                 sqrt(b_tilde_squared * s_tilde_squared),
                                 SimdType{1.0}),
                SimdType{0.0})
          : static_cast<SimdType>(0);

  // Equation B.40 of Foucart
  const auto lower_bound_of_lorentz_factor_minus_one =
      simd::max(tau_over_d - b_squared_over_d, SimdType{0.});
  // Equation B.31 of Foucart
  const auto upper_bound_for_s_tilde_squared =
      [&b_squared_over_d, &d_tilde, &lower_bound_of_lorentz_factor_minus_one,
       &normalized_s_dot_b](const auto local_excess_lorentz_factor) {
        const auto local_lorentz_factor_minus_one =
            lower_bound_of_lorentz_factor_minus_one +
            local_excess_lorentz_factor;
        if constexpr (NonZeroMagneticField) {
          return square(1.0 + local_lorentz_factor_minus_one +
                        b_squared_over_d) *
                 local_lorentz_factor_minus_one *
                 (2.0 + local_lorentz_factor_minus_one) /
                 (square(1.0 + local_lorentz_factor_minus_one) +
                  square(normalized_s_dot_b) * b_squared_over_d *
                      (b_squared_over_d +
                       2.0 * (1.0 + local_lorentz_factor_minus_one))) *
                 square(d_tilde);
        } else {
          (void)b_squared_over_d;
          (void)normalized_s_dot_b;
          return local_lorentz_factor_minus_one *
                 (2.0 + local_lorentz_factor_minus_one) * square(d_tilde);
        }
      };
  const SimdType simple_upper_bound_for_s_tilde_squared =
      upper_bound_for_s_tilde_squared(SimdType{0.});

  // If s_tilde_squared is small enough, no fix is needed. Otherwise, we need
  // to do some real work.
  const auto one_minus_safety_factor_for_momentum_density_at_density =
      simd::select(
          rest_mass_density_times_lorentz_factor >
              safety_factor_for_momentum_density_cutoff_d_,
          SimdType{one_minus_safety_factor_for_momentum_density_},
          one_minus_safety_factor_for_momentum_density_ +
              safety_factor_for_momentum_density_slope_ *
                  log10(rest_mass_density_times_lorentz_factor *
                        one_over_safety_factor_for_momentum_density_cutoff_d));
  if (const auto tilde_s_mask =
          s_tilde_squared >
              one_minus_safety_factor_for_momentum_density_at_density *
                  simple_upper_bound_for_s_tilde_squared and
          not completion_mask;
      simd::any(tilde_s_mask)) {
    // Find root of Equation B.34 of Foucart
    // NOTE:
    // - This assumes minimum specific enthalpy is 1.
    // - SpEC implements a more complicated formula (B.32) which is equivalent
    // - Bounds on root are given by Equation  B.40 of Foucart
    // - In regions where the solution is just above atmosphere we sometimes
    //   obtain an upper bound on the Lorentz factor somewhere around ~1e5,
    //   while the actual Lorentz factor is only 1+1e-6. This leads to
    //   situations where the solver must perform many (over 50) iterations to
    //   converge. A simple way of avoiding this is to check that
    //   [W_{lower_bound}, 10 * W_{lower_bound}] brackets the root and then
    //   use 10 * W_{lower_bound} as the upper bound. This reduces the number
    //   of iterations for the TOMS748 algorithm to converge to less than 10.
    //   Note that the factor 10 is chosen arbitrarily and could probably be
    //   reduced if required. The reasoning behind 10 is that it is unlikely
    //   the Lorentz factor will increase by a factor of 10 from one time step
    //   to the next in a physically meaning situation, and so 10 provides a
    //   reasonable bound.
    const auto f_of_lorentz_factor =
        FixConservatives_detail::make_function_of_lorentz_factor<NonZeroMagneticField>(
            b_squared_over_d, tau_over_d, normalized_s_dot_b,
            lower_bound_of_lorentz_factor_minus_one);
    SimdType upper_bound =
        simd::select(lower_bound_of_lorentz_factor_minus_one == 0.0, tau_over_d,
                     b_squared_over_d);

    SimdType excess_lorentz_factor = 0.0;
    if (const auto upper_not_zero_mask = upper_bound != 0.0;
        simd::any(upper_not_zero_mask)) {
      const SimdType f_at_lower = f_of_lorentz_factor(SimdType{0.0});
      const SimdType candidate_upper_bound =
          9.0 * (lower_bound_of_lorentz_factor_minus_one + 1.0);
      // The if-based implementation is here as a reference since it's
      // likely easier to understand.
      //
      // auto f_at_upper = std::numeric_limits<SimdType>::signaling_NaN();
      // if (upper_bound < candidate_upper_bound) {
      //   f_at_upper = f_of_lorentz_factor(upper_bound);
      // } else {
      //   f_at_upper = f_of_lorentz_factor(candidate_upper_bound);
      //   if (f_at_upper > 0.0) {
      //     upper_bound = candidate_upper_bound;
      //   } else {
      //     f_at_upper = f_of_lorentz_factor(upper_bound);
      //   }
      // }
      auto f_at_upper = f_of_lorentz_factor(upper_bound);
      if (const auto not_upper_less_candidate_bound =
              not(upper_bound < candidate_upper_bound);
          simd::any(not_upper_less_candidate_bound)) {
        const SimdType f_at_candidate_upper_bound =
            f_of_lorentz_factor(candidate_upper_bound);
        upper_bound = simd::select((f_at_candidate_upper_bound > 0.0) and
                                       not_upper_less_candidate_bound,
                                   candidate_upper_bound, upper_bound);
        f_at_upper = simd::select((f_at_candidate_upper_bound > 0.0) and
                                      not_upper_less_candidate_bound,
                                  f_at_candidate_upper_bound, f_at_upper);
      }

      const auto root = RootFinder::toms748<true>(
          f_of_lorentz_factor, SimdType{0.0}, upper_bound, f_at_lower,
          f_at_upper, 1.e-14, 1.e-14, 100, not tilde_s_mask);
      if (not root.has_value()) {
        *root_find_failure = RootFindFailure<T>{
            upper_bound,     lower_bound_of_lorentz_factor_minus_one,
            s_tilde_squared, d_tilde,
            sqrt_det_g,      tau_tilde,
            b_tilde_squared, b_squared_over_d,
            tau_over_d,      normalized_s_dot_b,
            root.error()};
        return needed_fixing;
      }
      excess_lorentz_factor = root.value();
    }

    const auto rescaling_factor = simd::select(
        tilde_s_mask,
        simd::min(sqrt(one_minus_safety_factor_for_momentum_density_ *
                       upper_bound_for_s_tilde_squared(excess_lorentz_factor) /
                       (s_tilde_squared + 1.e-16 * square(d_tilde))),
                  SimdType{1.}),
        SimdType{1.});
    if (UNLIKELY(simd::any(rescaling_factor < 1.))) {
      needed_fixing = true;
      *s_tilde_rescaling_factor = rescaling_factor;
    }
  }
  return needed_fixing;
}

KOKKOS_INLINE_FUNCTION bool FixConservatives::fix_at_point(
    const gsl::not_null<Scalar<double>*> tilde_d,
    const gsl::not_null<Scalar<double>*> tilde_ye,
    const gsl::not_null<Scalar<double>*> tilde_tau,
    const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<double, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<double, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<double>& sqrt_det_spatial_metric,
    const bool assume_non_zero_magnetic_field,
    const gsl::not_null<std::optional<RootFindFailure<double>>*>
        root_find_failure) const {
  if (not enable_) {
    return false;
  }
  Scalar<double> tilde_s_squared{};
  dot_product(make_not_null(&tilde_s_squared), *tilde_s, *tilde_s,
              inv_spatial_metric);
  double s_tilde_rescaling_factor = 1.0;
  bool needed_fixing = false;
  if (assume_non_zero_magnetic_field) {
    Scalar<double> tilde_b_squared{};
    dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
                spatial_metric);
    Scalar<double> tilde_s_dot_tilde_b{};
    dot_product(make_not_null(&tilde_s_dot_tilde_b), *tilde_s, tilde_b);
    needed_fixing = fix_impl<true>(
        make_not_null(&get(*tilde_d)), make_not_null(&get(*tilde_ye)),
        make_not_null(&get(*tilde_tau)),
        make_not_null(&s_tilde_rescaling_factor), get(sqrt_det_spatial_metric),
        get(tilde_s_squared), get(tilde_b_squared), get(tilde_s_dot_tilde_b),
        false, root_find_failure);
  } else {
    needed_fixing = fix_impl<false>(
        make_not_null(&get(*tilde_d)), make_not_null(&get(*tilde_ye)),
        make_not_null(&get(*tilde_tau)),
        make_not_null(&s_tilde_rescaling_factor), get(sqrt_det_spatial_metric),
        get(tilde_s_squared), 0.0, 0.0, false, root_find_failure);
  }
  if (s_tilde_rescaling_factor < 1.0) {
    for (size_t i = 0; i < 3; i++) {
      tilde_s->get(i) *= s_tilde_rescaling_factor;
    }
  }
  return needed_fixing;
}

KOKKOS_INLINE_FUNCTION PointwiseStatus FixConservatives::AtGridPoint::apply(
    const gsl::not_null<Scalar<double>*> tilde_d,
    const gsl::not_null<Scalar<double>*> tilde_ye,
    const gsl::not_null<Scalar<double>*> tilde_tau,
    const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*> tilde_s,
    const Scalar<double>& unfixed_tilde_d,
    const Scalar<double>& unfixed_tilde_ye,
    const Scalar<double>& unfixed_tilde_tau,
    const tnsr::i<double, 3, Frame::Inertial>& unfixed_tilde_s,
    const tnsr::I<double, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<double, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<double>& sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    const bool assume_non_zero_magnetic_field) {
  *tilde_d = unfixed_tilde_d;
  *tilde_ye = unfixed_tilde_ye;
  *tilde_tau = unfixed_tilde_tau;
  *tilde_s = unfixed_tilde_s;
  std::optional<RootFindFailure<double>> root_find_failure{};
  const bool needed_fixing = fix_conservatives.fix_at_point(
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, spatial_metric,
      inv_spatial_metric, sqrt_det_spatial_metric,
      assume_non_zero_magnetic_field, make_not_null(&root_find_failure));
  if (root_find_failure.has_value()) {
    // Leave the variables unfixed, so the caller can report them
    *tilde_d = unfixed_tilde_d;
    *tilde_ye = unfixed_tilde_ye;
    *tilde_tau = unfixed_tilde_tau;
    *tilde_s = unfixed_tilde_s;
    return {false, true};
  }
  return {needed_fixing, false};
}
}  // namespace grmhd::ValenciaDivClean
