// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace grmhd::ValenciaDivClean {
namespace detail {
template <typename DataType>
KOKKOS_FUNCTION void fluxes_impl(
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataType, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_phi_flux,

    // Temporaries
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*>
        transport_velocity,
    const tnsr::i<DataType, 3, Frame::Inertial>& lapse_b_over_w,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& pressure_star_lapse_sqrt_det_spatial_metric,

    // Extra args
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& tilde_ye,
    const Scalar<DataType>& tilde_tau,
    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataType>& tilde_phi, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::II<DataType, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity) {
  hydro::transport_velocity(transport_velocity, spatial_velocity, lapse, shift);
  for (size_t i = 0; i < 3; ++i) {
    tilde_d_flux->get(i) = get(tilde_d) * transport_velocity->get(i);
    tilde_ye_flux->get(i) = get(tilde_ye) * transport_velocity->get(i);
    tilde_tau_flux->get(i) =
        get(tilde_tau) * transport_velocity->get(i) +
        get(pressure_star_lapse_sqrt_det_spatial_metric) *
            spatial_velocity.get(i) -
        get(lapse) * get(magnetic_field_dot_spatial_velocity) * tilde_b.get(i);
    tilde_phi_flux->get(i) =
        get(lapse) * tilde_b.get(i) - get(tilde_phi) * shift.get(i);
    for (size_t j = 0; j < 3; ++j) {
      tilde_s_flux->get(i, j) = tilde_s.get(j) * transport_velocity->get(i) -
                                lapse_b_over_w.get(j) * tilde_b.get(i);
      tilde_b_flux->get(i, j) =
          tilde_b.get(j) * transport_velocity->get(i) +
          get(lapse) * (get(tilde_phi) * inv_spatial_metric.get(i, j) -
                        spatial_velocity.get(j) * tilde_b.get(i));
    }
    tilde_s_flux->get(i, i) += get(pressure_star_lapse_sqrt_det_spatial_metric);
  }
}
}  // namespace detail

template <typename DataType>
KOKKOS_FUNCTION void ComputeFluxes::apply(
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataType, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_phi_flux,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& tilde_ye,
    const Scalar<DataType>& tilde_tau,
    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataType>& tilde_phi, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataType>& pressure,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, 3, Frame::Inertial>& magnetic_field) {
  // A `Variables` for `DataVector` and a `TaggedTuple` on the stack for
  // `double`
  TempBuffer<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataType, 3>,
                        hydro::Tags::MagneticFieldOneForm<DataType, 3>,
                        hydro::Tags::MagneticFieldDotSpatialVelocity<DataType>,
                        hydro::Tags::MagneticFieldSquared<DataType>,
                        ::Tags::TempScalar<0, DataType>,
                        ::Tags::TempScalar<1, DataType>,
                        ::Tags::TempI<2, 3, Frame::Inertial, DataType>>>
      temp_tensors{get_size(get<0>(shift))};

  auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataType, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                       spatial_velocity, spatial_metric);
  auto& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataType, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&magnetic_field_one_form), magnetic_field,
                       spatial_metric);
  auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataType>>(temp_tensors);
  dot_product(make_not_null(&magnetic_field_dot_spatial_velocity),
              magnetic_field, spatial_velocity_one_form);
  auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataType>>(temp_tensors);
  dot_product(make_not_null(&magnetic_field_squared), magnetic_field,
              magnetic_field_one_form);

  DataType& one_over_w_squared =
      get(get<::Tags::TempScalar<0, DataType>>(temp_tensors));
  one_over_w_squared = 1.0 / square(get(lorentz_factor));
  // p_star = p + p_m = p + b^2/2 = p + ((B^m v_m)^2 + (B^m B_m)/W^2)/2
  Scalar<DataType>& pressure_star_lapse_sqrt_det_spatial_metric =
      get<::Tags::TempScalar<1, DataType>>(temp_tensors);
  get(pressure_star_lapse_sqrt_det_spatial_metric) =
      get(sqrt_det_spatial_metric) * get(lapse) *
      (get(pressure) + 0.5 * square(get(magnetic_field_dot_spatial_velocity)) +
       0.5 * get(magnetic_field_squared) * one_over_w_squared);

  // lapse b_i / W = lapse (B_i / W^2 + v_i (B^m v_m)
  tnsr::i<DataType, 3, Frame::Inertial>& lapse_b_over_w =
      get<hydro::Tags::SpatialVelocityOneForm<DataType, 3>>(temp_tensors);
  for (size_t i = 0; i < 3; ++i) {
    lapse_b_over_w.get(i) *= get(magnetic_field_dot_spatial_velocity);
    lapse_b_over_w.get(i) +=
        one_over_w_squared * magnetic_field_one_form.get(i);
    lapse_b_over_w.get(i) *= get(lapse);
  }

  detail::fluxes_impl(
      tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux,
      tilde_phi_flux,
      // Temporaries
      make_not_null(
          &get<::Tags::TempI<2, 3, Frame::Inertial, DataType>>(temp_tensors)),
      lapse_b_over_w, magnetic_field_dot_spatial_velocity,
      pressure_star_lapse_sqrt_det_spatial_metric,
      // Extra args
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
      inv_spatial_metric, spatial_velocity);
}
}  // namespace grmhd::ValenciaDivClean

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

GENERATE_INSTANTIATIONS(INSTANTIATE, (double))
// The `DataVector` versions are never called on the device
#ifndef SPECTRE_KOKKOS_DEVICE_PASS
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))
#endif  // SPECTRE_KOKKOS_DEVICE_PASS

#undef INSTANTIATE
#undef DTYPE
