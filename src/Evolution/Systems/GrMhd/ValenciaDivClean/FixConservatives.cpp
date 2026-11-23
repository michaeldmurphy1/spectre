// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.tpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <limits>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Simd/Simd.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// The state at a failed root find for the Lorentz factor, for error messages.
// The error of the root finder is streamed separately.
template <typename RootFindFailure>
std::string root_find_failure_message(const RootFindFailure& failure) {
  // clang-format makes the streamed text hard to read in code...
  // clang-format off
  return MakeString{}
      << std::scientific << std::setprecision(18)
      << "  upper_bound = " << failure.upper_bound
      << "\n  lower_bound_of_lorentz_factor_minus_one = "
      << failure.lower_bound_of_lorentz_factor_minus_one
      << "\n  s_tilde_squared = " << failure.s_tilde_squared
      << "\n  d_tilde = " << failure.d_tilde
      << "\n  sqrt_det_g = " << failure.sqrt_det_g
      << "\n  tau_tilde = " << failure.tau_tilde
      << "\n  b_tilde_squared = " << failure.b_tilde_squared
      << "\n  b_squared_over_d = " << failure.b_squared_over_d
      << "\n  tau_over_d = " << failure.tau_over_d
      << "\n  normalized_s_dot_b = " << failure.normalized_s_dot_b;
  // clang-format on
}
}  // namespace

namespace grmhd::ValenciaDivClean {
FixConservatives::FixConservatives(
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    double minimum_electron_fraction, double electron_fraction_cutoff,
    const double safety_factor_for_magnetic_field,
    const double safety_factor_for_momentum_density,
    const double safety_factor_for_momentum_density_cutoff_d,
    const double safety_factor_for_momentum_density_slope, const bool enable,
    const hydro::MagneticFieldTreatment magnetic_field_treatment,
    const Options::Context& context)
    : minimum_rest_mass_density_times_lorentz_factor_(
          minimum_rest_mass_density_times_lorentz_factor),
      rest_mass_density_times_lorentz_factor_cutoff_(
          rest_mass_density_times_lorentz_factor_cutoff),
      minimum_electron_fraction_(minimum_electron_fraction),
      electron_fraction_cutoff_(electron_fraction_cutoff),
      one_minus_safety_factor_for_magnetic_field_(
          1.0 - safety_factor_for_magnetic_field),
      one_minus_safety_factor_for_momentum_density_(
          1.0 - safety_factor_for_momentum_density),
      safety_factor_for_momentum_density_cutoff_d_(
          safety_factor_for_momentum_density_cutoff_d),
      safety_factor_for_momentum_density_slope_(
          safety_factor_for_momentum_density_slope),
      enable_(enable),
      magnetic_field_treatment_(magnetic_field_treatment) {
  if (minimum_rest_mass_density_times_lorentz_factor_ >
      rest_mass_density_times_lorentz_factor_cutoff_) {
    PARSE_ERROR(context,
                "The minimum value of D (a.k.a. rest mass density times "
                "Lorentz factor) ("
                    << minimum_rest_mass_density_times_lorentz_factor_
                    << ") must be less than or equal to the cutoff value of D ("
                    << rest_mass_density_times_lorentz_factor_cutoff_ << ')');
  }
  if (minimum_electron_fraction_ > electron_fraction_cutoff_) {
    PARSE_ERROR(context,
                "The minimum value of electron fraction Y_e ("
                    << minimum_electron_fraction_
                    << ") must be less than or equal to the cutoff value ("
                    << electron_fraction_cutoff_ << ')');
  }
  if (safety_factor_for_momentum_density_slope_ < 0.0) {
    PARSE_ERROR(context,
                "The option SafetyFactorForSSlope cannot be negative.");
  }
  if (safety_factor_for_momentum_density +
          safety_factor_for_momentum_density_slope_ *
              log10(safety_factor_for_momentum_density_cutoff_d /
                    minimum_rest_mass_density_times_lorentz_factor_) >=
      1.0) {
    PARSE_ERROR(context,
                "SafetyFactorForSSlope too large, will lead to unphysical "
                "cutoff at low density");
  }
}

// NOLINTNEXTLINE(google-runtime-references)
void FixConservatives::pup(PUP::er& p) {
  p | minimum_rest_mass_density_times_lorentz_factor_;
  p | rest_mass_density_times_lorentz_factor_cutoff_;
  p | minimum_electron_fraction_;
  p | electron_fraction_cutoff_;
  p | one_minus_safety_factor_for_magnetic_field_;
  p | one_minus_safety_factor_for_momentum_density_;
  p | safety_factor_for_momentum_density_cutoff_d_;
  p | safety_factor_for_momentum_density_slope_;
  p | enable_;
  p | magnetic_field_treatment_;
}

bool FixConservatives::assume_non_zero_magnetic_field(
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b) const {
  if (magnetic_field_treatment_ == hydro::MagneticFieldTreatment::CheckIfZero) {
    return max(max(abs(get<0>(tilde_b)), abs(get<1>(tilde_b)),
                   abs(get<2>(tilde_b)))) > 0.0;
  }
  return magnetic_field_treatment_ ==
         hydro::MagneticFieldTreatment::AssumeNonZero;
}

bool FixConservatives::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) const {
  bool needed_fixing = false;
  if (not enable_) {
    return needed_fixing;
  }
  const size_t size = get<0>(tilde_b).size();
  Variables<tmpl::list<::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
                       ::Tags::TempScalar<3>>>
      temp_buffer(size);

  Scalar<DataVector>& tilde_s_squared = get<::Tags::TempScalar<2>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_squared), *tilde_s, *tilde_s,
              inv_spatial_metric);

  const bool non_zero_mag = assume_non_zero_magnetic_field(tilde_b);

  Scalar<DataVector>& tilde_b_squared = get<::Tags::TempScalar<1>>(temp_buffer);
  Scalar<DataVector>& tilde_s_dot_tilde_b =
      get<::Tags::TempScalar<3>>(temp_buffer);

  if (non_zero_mag) {
    dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
                spatial_metric);
    dot_product(make_not_null(&tilde_s_dot_tilde_b), *tilde_s, tilde_b);
  }

  const auto fix_at = [&](const size_t grid_index, auto use_simd,
                          const auto completion_mask,
                          auto assume_non_zero_magnetic_field_v) {
    constexpr bool assume_non_zero_magnetic_field_at_point =
        decltype(assume_non_zero_magnetic_field_v)::value;
    using SimdType =
        tmpl::conditional_t<std::decay_t<decltype(use_simd)>::value,
                            simd::batch<double>, double>;
    const auto load = [&grid_index, &use_simd](const auto& data_vector) {
      (void)use_simd;
      if constexpr (std::decay_t<decltype(use_simd)>::value) {
        return simd::load_unaligned(&data_vector[grid_index]);
      } else {
        return data_vector[grid_index];
      }
    };
    const auto store = [&grid_index, &use_simd](
                           const gsl::not_null<DataVector*> data_vector,
                           const SimdType& value) {
      (void)use_simd;
      if constexpr (std::decay_t<decltype(use_simd)>::value) {
        simd::store_unaligned(&(*data_vector)[grid_index], value);
      } else {
        (*data_vector)[grid_index] = value;
      }
    };

    SimdType d_tilde = load(get(*tilde_d));
    SimdType ye_tilde = load(get(*tilde_ye));
    SimdType tau_tilde = load(get(*tilde_tau));
    SimdType s_tilde_rescaling_factor{1.0};
    const auto zeroes = static_cast<SimdType>(0);
    std::optional<RootFindFailure<SimdType>> root_find_failure{};
    if (fix_impl<assume_non_zero_magnetic_field_at_point>(
            make_not_null(&d_tilde), make_not_null(&ye_tilde),
            make_not_null(&tau_tilde), make_not_null(&s_tilde_rescaling_factor),
            load(get(sqrt_det_spatial_metric)), load(get(tilde_s_squared)),
            assume_non_zero_magnetic_field_at_point ? load(get(tilde_b_squared))
                                                    : zeroes,
            assume_non_zero_magnetic_field_at_point
                ? load(get(tilde_s_dot_tilde_b))
                : zeroes,
            completion_mask, make_not_null(&root_find_failure))) {
      needed_fixing = true;
    }
    if (root_find_failure.has_value()) {
      // clang-format makes the streamed text hard to read in code...
      // clang-format off
      ERROR(
          "Failed to fix conserved variables because the root finder failed "
          "to find the lorentz factor.\n"
          << root_find_failure_message(*root_find_failure)
          << "\n  tilde_s =\n" << extract_point(*tilde_s, grid_index)
          << "\n  tilde_b =\n" << extract_point(tilde_b, grid_index)
          << "\n  spatial_metric =\n"
          << extract_point(spatial_metric, grid_index)
          << "\n  inv_spatial_metric =\n"
          << extract_point(inv_spatial_metric, grid_index) << "\n"
          << "The error of the root finder is:\n"
          << root_find_failure->error);
      // clang-format on
    }

    if (UNLIKELY(simd::any(s_tilde_rescaling_factor < 1.))) {
      for (size_t i = 0; i < 3; i++) {
        store(make_not_null(&tilde_s->get(i)),
              load(tilde_s->get(i)) * s_tilde_rescaling_factor);
      }
    }
    store(make_not_null(&get(*tilde_d)), d_tilde);
    store(make_not_null(&get(*tilde_ye)), ye_tilde);
    store(make_not_null(&get(*tilde_tau)), tau_tilde);
  };

#ifdef SPECTRE_USE_XSIMD
  constexpr size_t simd_width = simd::size<simd::batch<double>>();
  if (size < simd_width) {
    if (non_zero_mag) {
      for (size_t s = 0; s < size; s++) {
        fix_at(s, std::false_type{}, false, std::true_type{});
      }
    } else {
      for (size_t s = 0; s < size; s++) {
        fix_at(s, std::false_type{}, false, std::false_type{});
      }
    }
  } else {
    const size_t vectorized_size = size - size % simd_width;
    if (non_zero_mag) {
      for (size_t s = 0; s < vectorized_size; s += simd_width) {
        fix_at(s, std::true_type{},
               simd::mask_type_t<simd::batch<double>>{false}, std::true_type{});
      }
      if (const size_t remainder = size - vectorized_size; remainder > 0) {
        const auto complete_mask =
            simd::make_sequence<simd::batch<double>>() <
            (simd_width - static_cast<double>(remainder));
        fix_at(size - simd_width, std::true_type{}, complete_mask,
               std::true_type{});
      }
    } else {
      for (size_t s = 0; s < vectorized_size; s += simd_width) {
        fix_at(s, std::true_type{},
               simd::mask_type_t<simd::batch<double>>{false},
               std::false_type{});
      }
      if (const size_t remainder = size - vectorized_size; remainder > 0) {
        const auto complete_mask =
            simd::make_sequence<simd::batch<double>>() <
            (simd_width - static_cast<double>(remainder));
        fix_at(size - simd_width, std::true_type{}, complete_mask,
               std::false_type{});
      }
    }
  }
#else
  if (non_zero_mag) {
    for (size_t s = 0; s < size; s++) {
      fix_at(s, std::false_type{}, false, std::true_type{});
    }
  } else {
    for (size_t s = 0; s < size; s++) {
      fix_at(s, std::false_type{}, false, std::false_type{});
    }
  }
#endif
  return needed_fixing;
}

bool operator==(const FixConservatives& lhs, const FixConservatives& rhs) {
  return lhs.minimum_rest_mass_density_times_lorentz_factor_ ==
             rhs.minimum_rest_mass_density_times_lorentz_factor_ and
         lhs.rest_mass_density_times_lorentz_factor_cutoff_ ==
             rhs.rest_mass_density_times_lorentz_factor_cutoff_ and
         lhs.one_minus_safety_factor_for_magnetic_field_ ==
             rhs.one_minus_safety_factor_for_magnetic_field_ and
         lhs.one_minus_safety_factor_for_momentum_density_ ==
             rhs.one_minus_safety_factor_for_momentum_density_ and
         lhs.safety_factor_for_momentum_density_cutoff_d_ ==
             rhs.safety_factor_for_momentum_density_cutoff_d_ and
         lhs.safety_factor_for_momentum_density_slope_ ==
             rhs.safety_factor_for_momentum_density_slope_ and
         lhs.enable_ == rhs.enable_;
}

bool operator!=(const FixConservatives& lhs, const FixConservatives& rhs) {
  return not(lhs == rhs);
}
}  // namespace grmhd::ValenciaDivClean
