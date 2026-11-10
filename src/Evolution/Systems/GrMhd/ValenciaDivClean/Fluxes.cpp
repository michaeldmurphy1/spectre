// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.tpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void grmhd::ValenciaDivClean::detail::fluxes_impl(                  \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_d_flux,   \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_ye_flux,  \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_tau_flux, \
      gsl::not_null<tnsr::Ij<DTYPE(data), 3, Frame::Inertial>*> tilde_s_flux,  \
      gsl::not_null<tnsr::IJ<DTYPE(data), 3, Frame::Inertial>*> tilde_b_flux,  \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_phi_flux, \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*>                 \
          transport_velocity,                                                  \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& lapse_b_over_w,          \
      const Scalar<DTYPE(data)>& magnetic_field_dot_spatial_velocity,          \
      const Scalar<DTYPE(data)>& pressure_star_lapse_sqrt_det_spatial_metric,  \
      const Scalar<DTYPE(data)>& tilde_d, const Scalar<DTYPE(data)>& tilde_ye, \
      const Scalar<DTYPE(data)>& tilde_tau,                                    \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi, const Scalar<DTYPE(data)>& lapse,  \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity);       \
  template void grmhd::ValenciaDivClean::ComputeFluxes::apply(                 \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_d_flux,   \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_ye_flux,  \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_tau_flux, \
      gsl::not_null<tnsr::Ij<DTYPE(data), 3, Frame::Inertial>*> tilde_s_flux,  \
      gsl::not_null<tnsr::IJ<DTYPE(data), 3, Frame::Inertial>*> tilde_b_flux,  \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_phi_flux, \
      const Scalar<DTYPE(data)>& tilde_d, const Scalar<DTYPE(data)>& tilde_ye, \
      const Scalar<DTYPE(data)>& tilde_tau,                                    \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi, const Scalar<DTYPE(data)>& lapse,  \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                      \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DTYPE(data)>& pressure,                                     \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,        \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& magnetic_field);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)))

#undef INSTANTIATE
#undef DTYPE
