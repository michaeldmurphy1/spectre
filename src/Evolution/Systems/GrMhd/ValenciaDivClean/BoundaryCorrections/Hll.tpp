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
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
