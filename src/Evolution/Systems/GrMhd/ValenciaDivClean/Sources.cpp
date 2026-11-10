// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.tpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void grmhd::ValenciaDivClean::detail::sources_impl(                 \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_tau,                    \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*> source_tilde_s, \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> source_tilde_b, \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_phi,                    \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_s_up,     \
      gsl::not_null<tnsr::II<DTYPE(data), 3, Frame::Inertial>*>                \
          densitized_stress,                                                   \
      gsl::not_null<Scalar<DTYPE(data)>*> h_rho_w_squared_plus_b_squared,      \
      const Scalar<DTYPE(data)>& magnetic_field_dot_spatial_velocity,          \
      const Scalar<DTYPE(data)>& magnetic_field_squared,                       \
      const Scalar<DTYPE(data)>& one_over_w_squared,                           \
      const Scalar<DTYPE(data)>& pressure_star,                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&                          \
          trace_spatial_christoffel_second,                                    \
      const Scalar<DTYPE(data)>& tilde_d, const Scalar<DTYPE(data)>& tilde_ye, \
      const Scalar<DTYPE(data)>& tilde_tau,                                    \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi, const Scalar<DTYPE(data)>& lapse,  \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                      \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& d_lapse,                 \
      const tnsr::iJ<DTYPE(data), 3, Frame::Inertial>& d_shift,                \
      const tnsr::ijj<DTYPE(data), 3, Frame::Inertial>& d_spatial_metric,      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,        \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& magnetic_field,          \
      const Scalar<DTYPE(data)>& rest_mass_density,                            \
      const Scalar<DTYPE(data)>& electron_fraction,                            \
      const Scalar<DTYPE(data)>& pressure,                                     \
      const Scalar<DTYPE(data)>& specific_internal_energy,                     \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& extrinsic_curvature,    \
      double constraint_damping_parameter);                                    \
  template void grmhd::ValenciaDivClean::ComputeSources::apply(                \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_tau,                    \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*> source_tilde_s, \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> source_tilde_b, \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_phi,                    \
      const Scalar<DTYPE(data)>& tilde_d, const Scalar<DTYPE(data)>& tilde_ye, \
      const Scalar<DTYPE(data)>& tilde_tau,                                    \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi,                                    \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& magnetic_field,          \
      const Scalar<DTYPE(data)>& rest_mass_density,                            \
      const Scalar<DTYPE(data)>& electron_fraction,                            \
      const Scalar<DTYPE(data)>& specific_internal_energy,                     \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const Scalar<DTYPE(data)>& pressure, const Scalar<DTYPE(data)>& lapse,   \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& d_lapse,                 \
      const tnsr::iJ<DTYPE(data), 3, Frame::Inertial>& d_shift,                \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::ijj<DTYPE(data), 3, Frame::Inertial>& d_spatial_metric,      \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                      \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& extrinsic_curvature,    \
      double constraint_damping_parameter,                                     \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& inertial_coords,         \
      const Mesh<3>& dg_mesh);                                                 \
  template void grmhd::ValenciaDivClean::detail::cartoon_sources_impl(         \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*> source_tilde_s, \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> source_tilde_b, \
      const Scalar<DTYPE(data)>& pressure_star,                                \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& magnetic_field_one_form, \
      const Scalar<DTYPE(data)>& magnetic_field_dot_spatial_velocity,          \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi,                                    \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,        \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& inertial_coords,         \
      Spectral::Quadrature cartoon_quadrature);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)))

#undef INSTANTIATE
#undef DTYPE
