// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.tpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template void grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply( \
      gsl::not_null<Scalar<DTYPE(data)>*> tilde_d,                         \
      gsl::not_null<Scalar<DTYPE(data)>*> tilde_ye,                        \
      gsl::not_null<Scalar<DTYPE(data)>*> tilde_tau,                       \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*> tilde_s,    \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_b,    \
      gsl::not_null<Scalar<DTYPE(data)>*> tilde_phi,                       \
      const Scalar<DTYPE(data)>& rest_mass_density,                        \
      const Scalar<DTYPE(data)>& electron_fraction,                        \
      const Scalar<DTYPE(data)>& specific_internal_energy,                 \
      const Scalar<DTYPE(data)>& pressure,                                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,    \
      const Scalar<DTYPE(data)>& lorentz_factor,                           \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& magnetic_field,      \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                  \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& spatial_metric,     \
      const Scalar<DTYPE(data)>& divergence_cleaning_field);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)))

#undef INSTANTIATE
#undef DTYPE
