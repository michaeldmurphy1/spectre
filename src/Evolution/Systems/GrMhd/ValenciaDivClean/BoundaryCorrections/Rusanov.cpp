// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"

#include <pup.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Rusanov::Rusanov(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Rusanov::get_clone() const {
  return std::make_unique<Rusanov>(*this);
}

void Rusanov::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

template <typename DataType>
KOKKOS_FUNCTION void Rusanov::dg_package_data_impl(
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
    const gsl::not_null<Scalar<DataType>*> packaged_abs_char_speed,

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
    const tnsr::i<DataType, 3, Frame::Inertial>& normal_covector,
    const Scalar<DataType>* const normal_dot_mesh_velocity) {
  {
    // Compute max abs char speed
    Scalar<DataType>& shift_dot_normal = *packaged_tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
    if constexpr (std::is_same_v<DataType, double>) {
      // Subtracting zero doesn't change the result
      const double normal_dot_mesh_velocity_or_zero =
          normal_dot_mesh_velocity != nullptr ? get(*normal_dot_mesh_velocity)
                                              : 0.0;
      get(*packaged_abs_char_speed) =
          std::max(std::abs(-get(lapse) - get(shift_dot_normal) -
                            normal_dot_mesh_velocity_or_zero),
                   std::abs(get(lapse) - get(shift_dot_normal) -
                            normal_dot_mesh_velocity_or_zero));
    } else if (normal_dot_mesh_velocity != nullptr) {
      get(*packaged_abs_char_speed) =
          max(abs(-get(lapse) - get(shift_dot_normal) -
                  get(*normal_dot_mesh_velocity)),
              abs(get(lapse) - get(shift_dot_normal) -
                  get(*normal_dot_mesh_velocity)));
    } else {
      get(*packaged_abs_char_speed) =
          max(abs(-get(lapse) - get(shift_dot_normal)),
              abs(get(lapse) - get(shift_dot_normal)));
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
KOKKOS_FUNCTION void Rusanov::dg_boundary_terms_impl(
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
    const Scalar<DataType>& abs_char_speed_int,
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
    const Scalar<DataType>& abs_char_speed_ext,
    const dg::Formulation dg_formulation) {
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataType>>> temps{
      get_size(get(abs_char_speed_int))};
  DataType& max_abs_char_speed =
      get(get<::Tags::TempScalar<0, DataType>>(temps));
  if constexpr (std::is_same_v<DataType, double>) {
    max_abs_char_speed =
        std::max(get(abs_char_speed_int), get(abs_char_speed_ext));
  } else {
    max_abs_char_speed = max(get(abs_char_speed_int), get(abs_char_speed_ext));
  }

  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_tilde_d) =
        0.5 * (get(normal_dot_flux_tilde_d_int) -
               get(normal_dot_flux_tilde_d_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_d_ext) - get(tilde_d_int));
    get(*boundary_correction_tilde_ye) =
        0.5 * (get(normal_dot_flux_tilde_ye_int) -
               get(normal_dot_flux_tilde_ye_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_ye_ext) - get(tilde_ye_int));
    get(*boundary_correction_tilde_tau) =
        0.5 * (get(normal_dot_flux_tilde_tau_int) -
               get(normal_dot_flux_tilde_tau_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_tau_ext) - get(tilde_tau_int));
    get(*boundary_correction_tilde_phi) =
        0.5 * (get(normal_dot_flux_tilde_phi_int) -
               get(normal_dot_flux_tilde_phi_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_phi_ext) - get(tilde_phi_int));

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          0.5 * (normal_dot_flux_tilde_s_int.get(i) -
                 normal_dot_flux_tilde_s_ext.get(i)) -
          0.5 * max_abs_char_speed * (tilde_s_ext.get(i) - tilde_s_int.get(i));
      boundary_correction_tilde_b->get(i) =
          0.5 * (normal_dot_flux_tilde_b_int.get(i) -
                 normal_dot_flux_tilde_b_ext.get(i)) -
          0.5 * max_abs_char_speed * (tilde_b_ext.get(i) - tilde_b_int.get(i));
    }
  } else {
    get(*boundary_correction_tilde_d) =
        -0.5 * (get(normal_dot_flux_tilde_d_int) +
                get(normal_dot_flux_tilde_d_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_d_ext) - get(tilde_d_int));
    get(*boundary_correction_tilde_ye) =
        -0.5 * (get(normal_dot_flux_tilde_ye_int) +
                get(normal_dot_flux_tilde_ye_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_ye_ext) - get(tilde_ye_int));
    get(*boundary_correction_tilde_tau) =
        -0.5 * (get(normal_dot_flux_tilde_tau_int) +
                get(normal_dot_flux_tilde_tau_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_tau_ext) - get(tilde_tau_int));
    get(*boundary_correction_tilde_phi) =
        -0.5 * (get(normal_dot_flux_tilde_phi_int) +
                get(normal_dot_flux_tilde_phi_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_phi_ext) - get(tilde_phi_int));

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          -0.5 * (normal_dot_flux_tilde_s_int.get(i) +
                  normal_dot_flux_tilde_s_ext.get(i)) -
          0.5 * max_abs_char_speed * (tilde_s_ext.get(i) - tilde_s_int.get(i));
      boundary_correction_tilde_b->get(i) =
          -0.5 * (normal_dot_flux_tilde_b_int.get(i) +
                  normal_dot_flux_tilde_b_ext.get(i)) -
          0.5 * max_abs_char_speed * (tilde_b_ext.get(i) - tilde_b_int.get(i));
    }
  }
}

#ifndef SPECTRE_KOKKOS_DEVICE_PASS
double Rusanov::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3,
                  Frame::Inertial>& /*spatial_velocity_one_form*/,

    const Scalar<DataVector>& /*rest_mass_density*/,
    const Scalar<DataVector>& /*electron_fraction*/,
    const Scalar<DataVector>& /*temperature*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*spatial_velocity*/,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>&
    /*equation_of_state*/) {
  dg_package_data_impl(
      packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau, packaged_tilde_s,
      packaged_tilde_b, packaged_tilde_phi, packaged_normal_dot_flux_tilde_d,
      packaged_normal_dot_flux_tilde_ye, packaged_normal_dot_flux_tilde_tau,
      packaged_normal_dot_flux_tilde_s, packaged_normal_dot_flux_tilde_b,
      packaged_normal_dot_flux_tilde_phi, packaged_abs_char_speed, tilde_d,
      tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
      flux_tilde_ye, flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi,
      lapse, shift, normal_covector,
      normal_dot_mesh_velocity.has_value() ? &normal_dot_mesh_velocity.value()
                                           : nullptr);
  return max(get(*packaged_abs_char_speed));
}

void Rusanov::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& abs_char_speed_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_ye_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    const dg::Formulation dg_formulation) {
  dg_boundary_terms_impl(
      boundary_correction_tilde_d, boundary_correction_tilde_ye,
      boundary_correction_tilde_tau, boundary_correction_tilde_s,
      boundary_correction_tilde_b, boundary_correction_tilde_phi, tilde_d_int,
      tilde_ye_int, tilde_tau_int, tilde_s_int, tilde_b_int, tilde_phi_int,
      normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_ye_int,
      normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_s_int,
      normal_dot_flux_tilde_b_int, normal_dot_flux_tilde_phi_int,
      abs_char_speed_int, tilde_d_ext, tilde_ye_ext, tilde_tau_ext, tilde_s_ext,
      tilde_b_ext, tilde_phi_ext, normal_dot_flux_tilde_d_ext,
      normal_dot_flux_tilde_ye_ext, normal_dot_flux_tilde_tau_ext,
      normal_dot_flux_tilde_s_ext, normal_dot_flux_tilde_b_ext,
      normal_dot_flux_tilde_phi_ext, abs_char_speed_ext, dg_formulation);
}
#endif  // SPECTRE_KOKKOS_DEVICE_PASS

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void Rusanov::dg_package_data_impl(                                 \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_tilde_d,                    \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_tilde_ye,                   \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_tilde_tau,                  \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*>                 \
          packaged_tilde_s,                                                    \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*>                 \
          packaged_tilde_b,                                                    \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_tilde_phi,                  \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_normal_dot_flux_tilde_d,    \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_normal_dot_flux_tilde_ye,   \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_normal_dot_flux_tilde_tau,  \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*>                 \
          packaged_normal_dot_flux_tilde_s,                                    \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*>                 \
          packaged_normal_dot_flux_tilde_b,                                    \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_normal_dot_flux_tilde_phi,  \
      gsl::not_null<Scalar<DTYPE(data)>*> packaged_abs_char_speed,             \
      const Scalar<DTYPE(data)>& tilde_d, const Scalar<DTYPE(data)>& tilde_ye, \
      const Scalar<DTYPE(data)>& tilde_tau,                                    \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi,                                    \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& flux_tilde_d,            \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& flux_tilde_ye,           \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& flux_tilde_tau,          \
      const tnsr::Ij<DTYPE(data), 3, Frame::Inertial>& flux_tilde_s,           \
      const tnsr::IJ<DTYPE(data), 3, Frame::Inertial>& flux_tilde_b,           \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& flux_tilde_phi,          \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& normal_covector,         \
      const Scalar<DTYPE(data)>* normal_dot_mesh_velocity);                    \
  template void Rusanov::dg_boundary_terms_impl(                               \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_d,         \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_ye,        \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_tau,       \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*>                 \
          boundary_correction_tilde_s,                                         \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*>                 \
          boundary_correction_tilde_b,                                         \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_phi,       \
      const Scalar<DTYPE(data)>& tilde_d_int,                                  \
      const Scalar<DTYPE(data)>& tilde_ye_int,                                 \
      const Scalar<DTYPE(data)>& tilde_tau_int,                                \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s_int,             \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b_int,             \
      const Scalar<DTYPE(data)>& tilde_phi_int,                                \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_d_int,                  \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_ye_int,                 \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_tau_int,                \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>&                          \
          normal_dot_flux_tilde_s_int,                                         \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&                          \
          normal_dot_flux_tilde_b_int,                                         \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_phi_int,                \
      const Scalar<DTYPE(data)>& abs_char_speed_int,                           \
      const Scalar<DTYPE(data)>& tilde_d_ext,                                  \
      const Scalar<DTYPE(data)>& tilde_ye_ext,                                 \
      const Scalar<DTYPE(data)>& tilde_tau_ext,                                \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s_ext,             \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b_ext,             \
      const Scalar<DTYPE(data)>& tilde_phi_ext,                                \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_d_ext,                  \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_ye_ext,                 \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_tau_ext,                \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>&                          \
          normal_dot_flux_tilde_s_ext,                                         \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&                          \
          normal_dot_flux_tilde_b_ext,                                         \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_phi_ext,                \
      const Scalar<DTYPE(data)>& abs_char_speed_ext,                           \
      dg::Formulation dg_formulation);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)))

#undef INSTANTIATE
#undef DTYPE

bool operator==(const Rusanov& /*lhs*/, const Rusanov& /*rhs*/) { return true; }

bool operator!=(const Rusanov& lhs, const Rusanov& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Rusanov::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
