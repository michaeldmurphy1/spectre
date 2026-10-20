// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"

#include <algorithm>
#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.tpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Hll::Hll(const double magnetic_field_magnitude_for_hydro,
         const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

Hll::Hll(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Hll::get_clone() const {
  return std::make_unique<Hll>(*this);
}

void Hll::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

bool Hll::use_hydro_speeds(
    const Scalar<DataVector>& rest_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b) const {
  const bool has_b_field =
      max(get(magnitude(tilde_b))) > magnetic_field_magnitude_for_hydro_;
  return not has_b_field and
         max(get(rest_mass_density)) > light_speed_density_cutoff_;
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

#ifndef SPECTRE_KOKKOS_DEVICE_PASS
double Hll::dg_package_data(
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
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_ingoing_char_speed,

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
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  dg_package_data_impl(
      packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau, packaged_tilde_s,
      packaged_tilde_b, packaged_tilde_phi, packaged_normal_dot_flux_tilde_d,
      packaged_normal_dot_flux_tilde_ye, packaged_normal_dot_flux_tilde_tau,
      packaged_normal_dot_flux_tilde_s, packaged_normal_dot_flux_tilde_b,
      packaged_normal_dot_flux_tilde_phi, packaged_largest_outgoing_char_speed,
      packaged_largest_ingoing_char_speed, tilde_d, tilde_ye, tilde_tau,
      tilde_s, tilde_b, tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau,
      flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
      spatial_velocity_one_form, rest_mass_density, electron_fraction,
      temperature, spatial_velocity, normal_covector,
      normal_dot_mesh_velocity.has_value() ? &normal_dot_mesh_velocity.value()
                                           : nullptr,
      equation_of_state, use_hydro_speeds(rest_mass_density, tilde_b),
      light_speed_density_cutoff_);

  return std::max(max(abs(get(*packaged_largest_outgoing_char_speed))),
                  max(abs(get(*packaged_largest_ingoing_char_speed))));
}

void Hll::dg_boundary_terms(
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
    const Scalar<DataVector>& largest_outgoing_char_speed_int,
    const Scalar<DataVector>& largest_ingoing_char_speed_int,
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
    const Scalar<DataVector>& largest_outgoing_char_speed_ext,
    const Scalar<DataVector>& largest_ingoing_char_speed_ext,
    const dg::Formulation dg_formulation) {
  dg_boundary_terms_impl(
      boundary_correction_tilde_d, boundary_correction_tilde_ye,
      boundary_correction_tilde_tau, boundary_correction_tilde_s,
      boundary_correction_tilde_b, boundary_correction_tilde_phi, tilde_d_int,
      tilde_ye_int, tilde_tau_int, tilde_s_int, tilde_b_int, tilde_phi_int,
      normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_ye_int,
      normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_s_int,
      normal_dot_flux_tilde_b_int, normal_dot_flux_tilde_phi_int,
      largest_outgoing_char_speed_int, largest_ingoing_char_speed_int,
      tilde_d_ext, tilde_ye_ext, tilde_tau_ext, tilde_s_ext, tilde_b_ext,
      tilde_phi_ext, normal_dot_flux_tilde_d_ext, normal_dot_flux_tilde_ye_ext,
      normal_dot_flux_tilde_tau_ext, normal_dot_flux_tilde_s_ext,
      normal_dot_flux_tilde_b_ext, normal_dot_flux_tilde_phi_ext,
      largest_outgoing_char_speed_ext, largest_ingoing_char_speed_ext,
      dg_formulation);
}
#endif  // SPECTRE_KOKKOS_DEVICE_PASS

bool operator==(const Hll& lhs, const Hll& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const Hll& lhs, const Hll& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hll::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template void                                                              \
  grmhd::ValenciaDivClean::BoundaryCorrections::Hll::dg_boundary_terms_impl( \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_d,       \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_ye,      \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_tau,     \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*>               \
          boundary_correction_tilde_s,                                       \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*>               \
          boundary_correction_tilde_b,                                       \
      gsl::not_null<Scalar<DTYPE(data)>*> boundary_correction_tilde_phi,     \
      const Scalar<DTYPE(data)>& tilde_d_int,                                \
      const Scalar<DTYPE(data)>& tilde_ye_int,                               \
      const Scalar<DTYPE(data)>& tilde_tau_int,                              \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s_int,           \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b_int,           \
      const Scalar<DTYPE(data)>& tilde_phi_int,                              \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_d_int,                \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_ye_int,               \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_tau_int,              \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>&                        \
          normal_dot_flux_tilde_s_int,                                       \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&                        \
          normal_dot_flux_tilde_b_int,                                       \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_phi_int,              \
      const Scalar<DTYPE(data)>& largest_outgoing_char_speed_int,            \
      const Scalar<DTYPE(data)>& largest_ingoing_char_speed_int,             \
      const Scalar<DTYPE(data)>& tilde_d_ext,                                \
      const Scalar<DTYPE(data)>& tilde_ye_ext,                               \
      const Scalar<DTYPE(data)>& tilde_tau_ext,                              \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s_ext,           \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b_ext,           \
      const Scalar<DTYPE(data)>& tilde_phi_ext,                              \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_d_ext,                \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_ye_ext,               \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_tau_ext,              \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>&                        \
          normal_dot_flux_tilde_s_ext,                                       \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&                        \
          normal_dot_flux_tilde_b_ext,                                       \
      const Scalar<DTYPE(data)>& normal_dot_flux_tilde_phi_ext,              \
      const Scalar<DTYPE(data)>& largest_outgoing_char_speed_ext,            \
      const Scalar<DTYPE(data)>& largest_ingoing_char_speed_ext,             \
      dg::Formulation dg_formulation);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)))

#undef INSTANTIATE
#undef DTYPE
