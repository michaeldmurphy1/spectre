// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
template <typename DataType, typename EquationOfStateType>
KOKKOS_FUNCTION void Hll::dg_package_data_impl(
    const gsl::not_null<Scalar<DataType>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataType>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataType>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataType, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataType>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataType>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataType>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataType>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataType, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataType>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataType>*> packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataType>*> packaged_largest_ingoing_char_speed,

    const Scalar<DataType>& tilde_d, const Scalar<DataType>& tilde_ye,
    const Scalar<DataType>& tilde_tau,
    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataType>& tilde_phi,

    const tnsr::I<DataType, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataType, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataType, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataType, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataType, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataType, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::i<DataType, 3, Frame::Inertial>& spatial_velocity_one_form,

    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& electron_fraction,
    const Scalar<DataType>& temperature,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity,

    const tnsr::i<DataType, 3, Frame::Inertial>& normal_covector,
    const Scalar<DataType>* const normal_dot_mesh_velocity,
    const EquationOfStateType& equation_of_state, const bool use_hydro_speeds,
    const double light_speed_density_cutoff) {
  {
    Scalar<DataType> shift_dot_normal = tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);

    // Initialize characteristic speeds to light speed
    // (default choice)
    get(*packaged_largest_outgoing_char_speed) =
        get(lapse) - get(shift_dot_normal);
    get(*packaged_largest_ingoing_char_speed) =
        -get(lapse) - get(shift_dot_normal);

    if (use_hydro_speeds) {
      // Since we have no magnetic field (and we have grid points not in the
      // atmosphere), we can reduce the char speeds to the hydro speeds only.
      // This makes the scheme less dissipative.
      //
      // Calculate sound speed squared
      const size_t num_points = get_size(get(rest_mass_density));
      TempBuffer<tmpl::list<
          ::Tags::TempScalar<0, DataType>, ::Tags::TempScalar<1, DataType>,
          ::Tags::TempScalar<2, DataType>, ::Tags::TempScalar<3, DataType>,
          ::Tags::TempScalar<4, DataType>, ::Tags::TempScalar<5, DataType>,
          ::Tags::TempScalar<6, DataType>>>
          temp_buffer{num_points};
      auto& v_dot_normal = get<::Tags::TempScalar<0, DataType>>(temp_buffer);
      auto& v_squared = get<::Tags::TempScalar<1, DataType>>(temp_buffer);
      auto& discriminant =
          get(get<::Tags::TempScalar<2, DataType>>(temp_buffer));
      auto& one_minus_v2_cs2 =
          get<::Tags::TempScalar<3, DataType>>(temp_buffer);
      auto& one_minus_cs2 = get<::Tags::TempScalar<4, DataType>>(temp_buffer);
      auto& lapse_over_one_minus_v2_cs2 =
          get<::Tags::TempScalar<5, DataType>>(temp_buffer);
      auto& v_dot_normal_times_one_minus_cs2 =
          get<::Tags::TempScalar<6, DataType>>(temp_buffer);

      Scalar<DataType> sound_speed_squared =
          equation_of_state.sound_speed_squared_from_density_and_temperature(
              rest_mass_density, temperature, electron_fraction);
      if constexpr (std::is_same_v<DataType, double>) {
        get(sound_speed_squared) =
            std::clamp(get(sound_speed_squared), 0.0, 1.0);
      } else {
        get(sound_speed_squared) = clamp(get(sound_speed_squared), 0.0, 1.0);
      }

      // Compute v_dot_normal, v^i n_i
      dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                  normal_covector);

      // Compute v^2=v^i v_i
      dot_product(make_not_null(&v_squared), spatial_velocity,
                  spatial_velocity_one_form);
      if constexpr (std::is_same_v<DataType, double>) {
        get(v_squared) = std::clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);
      } else {
        get(v_squared) = clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);
      }

      // Calculate characteristic speeds in inertial frame
      //
      // Ideally we'd use the Lorentz factor instead of 1-v^2, but I (Nils
      // Deppe) don't have the bandwidth to change this right now.
      get(one_minus_v2_cs2) = 1.0 - get(v_squared) * get(sound_speed_squared);
      get(one_minus_cs2) = 1.0 - get(sound_speed_squared);
      discriminant =
          get(sound_speed_squared) * (1.0 - get(v_squared)) *
          (get(one_minus_v2_cs2) -
           get(v_dot_normal) * get(v_dot_normal) * get(one_minus_cs2));
      if constexpr (std::is_same_v<DataType, double>) {
        discriminant = std::max(discriminant, 0.0);
      } else {
        discriminant = max(discriminant, 0.0);
      }
      discriminant = sqrt(discriminant);

      get(lapse_over_one_minus_v2_cs2) = get(lapse) / get(one_minus_v2_cs2);
      get(v_dot_normal_times_one_minus_cs2) =
          get(v_dot_normal) * get(one_minus_cs2);

      for (size_t i = 0; i < num_points; ++i) {
        if (get_element(get(rest_mass_density), i) >
            light_speed_density_cutoff) {
          get_element(get(*packaged_largest_outgoing_char_speed), i) =
              get_element(get(lapse_over_one_minus_v2_cs2), i) *
                  (get_element(get(v_dot_normal_times_one_minus_cs2), i) +
                   get_element(discriminant, i)) -
              get_element(get(shift_dot_normal), i);
          get_element(get(*packaged_largest_ingoing_char_speed), i) =
              get_element(get(lapse_over_one_minus_v2_cs2), i) *
                  (get_element(get(v_dot_normal_times_one_minus_cs2), i) -
                   get_element(discriminant, i)) -
              get_element(get(shift_dot_normal), i);
        }
      }
    }

    // Correct for mesh velocity
    if (normal_dot_mesh_velocity != nullptr) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

  normal_dot_flux(packaged_normal_dot_flux_tilde_d, normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_ye, normal_covector,
                  flux_tilde_ye);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, normal_covector,
                  flux_tilde_phi);
}

template <typename DataType>
KOKKOS_FUNCTION void Hll::dg_boundary_terms_impl(
    const gsl::not_null<Scalar<DataType>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataType>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataType>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataType, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataType>*> boundary_correction_tilde_phi,
    const Scalar<DataType>& tilde_d_int, const Scalar<DataType>& tilde_ye_int,
    const Scalar<DataType>& tilde_tau_int,
    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataType>& tilde_phi_int,
    const Scalar<DataType>& normal_dot_flux_tilde_d_int,
    const Scalar<DataType>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataType>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataType, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataType, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataType>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataType>& largest_outgoing_char_speed_int,
    const Scalar<DataType>& largest_ingoing_char_speed_int,
    const Scalar<DataType>& tilde_d_ext, const Scalar<DataType>& tilde_ye_ext,
    const Scalar<DataType>& tilde_tau_ext,
    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataType>& tilde_phi_ext,
    const Scalar<DataType>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataType>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataType>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataType, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataType, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataType>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataType>& largest_outgoing_char_speed_ext,
    const Scalar<DataType>& largest_ingoing_char_speed_ext,
    const dg::Formulation dg_formulation) {
  // Allocate a temp buffer with four tags.
  TempBuffer<tmpl::list<
      ::Tags::TempScalar<0, DataType>, ::Tags::TempScalar<1, DataType>,
      ::Tags::TempScalar<2, DataType>, ::Tags::TempScalar<3, DataType>>>
      temps{get_size(get(tilde_d_int))};

  // Determine lambda_max and lambda_min from the characteristic speeds info
  // from interior and exterior
  if constexpr (std::is_same_v<DataType, double>) {
    get(get<::Tags::TempScalar<0, DataType>>(temps)) =
        std::max(std::max(0., get(largest_outgoing_char_speed_int)),
                 -get(largest_ingoing_char_speed_ext));
    get(get<::Tags::TempScalar<1, DataType>>(temps)) =
        std::min(std::min(0., get(largest_ingoing_char_speed_int)),
                 -get(largest_outgoing_char_speed_ext));
  } else {
    get(get<::Tags::TempScalar<0, DataType>>(temps)) =
        max(0., get(largest_outgoing_char_speed_int),
            -get(largest_ingoing_char_speed_ext));
    get(get<::Tags::TempScalar<1, DataType>>(temps)) =
        min(0., get(largest_ingoing_char_speed_int),
            -get(largest_outgoing_char_speed_ext));
  }
  const DataType& lambda_max = get(get<::Tags::TempScalar<0, DataType>>(temps));
  const DataType& lambda_min = get(get<::Tags::TempScalar<1, DataType>>(temps));

  // Pre-compute two expressions made out of lambda_max and lambda_min
  get(get<::Tags::TempScalar<2, DataType>>(temps)) = lambda_max * lambda_min;
  const DataType& lambdas_product =
      get(get<::Tags::TempScalar<2, DataType>>(temps));
  get(get<::Tags::TempScalar<3, DataType>>(temps)) =
      1. / (lambda_max - lambda_min);
  const DataType& one_over_lambda_max_minus_min =
      get(get<::Tags::TempScalar<3, DataType>>(temps));

  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_tilde_d) =
        ((lambda_max * get(normal_dot_flux_tilde_d_int) +
          lambda_min * get(normal_dot_flux_tilde_d_ext)) +
         lambdas_product * (get(tilde_d_ext) - get(tilde_d_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_ye) =
        ((lambda_max * get(normal_dot_flux_tilde_ye_int) +
          lambda_min * get(normal_dot_flux_tilde_ye_ext)) +
         lambdas_product * (get(tilde_ye_ext) - get(tilde_ye_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_tau) =
        ((lambda_max * get(normal_dot_flux_tilde_tau_int) +
          lambda_min * get(normal_dot_flux_tilde_tau_ext)) +
         lambdas_product * (get(tilde_tau_ext) - get(tilde_tau_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_phi) =
        ((lambda_max * get(normal_dot_flux_tilde_phi_int) +
          lambda_min * get(normal_dot_flux_tilde_phi_ext)) +
         lambdas_product * (get(tilde_phi_ext) - get(tilde_phi_int))) *
        one_over_lambda_max_minus_min;

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          ((lambda_max * normal_dot_flux_tilde_s_int.get(i) +
            lambda_min * normal_dot_flux_tilde_s_ext.get(i)) +
           lambdas_product * (tilde_s_ext.get(i) - tilde_s_int.get(i))) *
          one_over_lambda_max_minus_min;
      boundary_correction_tilde_b->get(i) =
          ((lambda_max * normal_dot_flux_tilde_b_int.get(i) +
            lambda_min * normal_dot_flux_tilde_b_ext.get(i)) +
           lambdas_product * (tilde_b_ext.get(i) - tilde_b_int.get(i))) *
          one_over_lambda_max_minus_min;
    }
  } else {
    get(*boundary_correction_tilde_d) =
        (lambda_min * (get(normal_dot_flux_tilde_d_int) +
                       get(normal_dot_flux_tilde_d_ext)) +
         lambdas_product * (get(tilde_d_ext) - get(tilde_d_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_ye) =
        (lambda_min * (get(normal_dot_flux_tilde_ye_int) +
                       get(normal_dot_flux_tilde_ye_ext)) +
         lambdas_product * (get(tilde_ye_ext) - get(tilde_ye_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_tau) =
        (lambda_min * (get(normal_dot_flux_tilde_tau_int) +
                       get(normal_dot_flux_tilde_tau_ext)) +
         lambdas_product * (get(tilde_tau_ext) - get(tilde_tau_int))) *
        one_over_lambda_max_minus_min;
    get(*boundary_correction_tilde_phi) =
        (lambda_min * (get(normal_dot_flux_tilde_phi_int) +
                       get(normal_dot_flux_tilde_phi_ext)) +
         lambdas_product * (get(tilde_phi_ext) - get(tilde_phi_int))) *
        one_over_lambda_max_minus_min;

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          (lambda_min * (normal_dot_flux_tilde_s_int.get(i) +
                         normal_dot_flux_tilde_s_ext.get(i)) +
           lambdas_product * (tilde_s_ext.get(i) - tilde_s_int.get(i))) *
          one_over_lambda_max_minus_min;
      boundary_correction_tilde_b->get(i) =
          (lambda_min * (normal_dot_flux_tilde_b_int.get(i) +
                         normal_dot_flux_tilde_b_ext.get(i)) +
           lambdas_product * (tilde_b_ext.get(i) - tilde_b_int.get(i))) *
          one_over_lambda_max_minus_min;
    }
  }
}

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
