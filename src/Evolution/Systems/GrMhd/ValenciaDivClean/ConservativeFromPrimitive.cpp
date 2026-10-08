// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace grmhd::ValenciaDivClean {

template <typename DataType>
KOKKOS_FUNCTION void ConservativeFromPrimitive::apply(
    const gsl::not_null<Scalar<DataType>*> tilde_d,
    const gsl::not_null<Scalar<DataType>*> tilde_ye,
    const gsl::not_null<Scalar<DataType>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataType, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataType>*> tilde_phi,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& electron_fraction,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const Scalar<DataType>& divergence_cleaning_field) {
  TempBuffer<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataType, 3>,
                        hydro::Tags::SpatialVelocitySquared<DataType>,
                        hydro::Tags::MagneticFieldOneForm<DataType, 3>,
                        hydro::Tags::MagneticFieldDotSpatialVelocity<DataType>,
                        hydro::Tags::MagneticFieldSquared<DataType>>>
      temp_tensors{get_size(get(rest_mass_density))};
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
  auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataType>>(temp_tensors);
  dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
              spatial_velocity_one_form);

  auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataType>>(temp_tensors);
  dot_product(make_not_null(&magnetic_field_squared), magnetic_field,
              magnetic_field_one_form);

  get(*tilde_d) = get(sqrt_det_spatial_metric) * get(rest_mass_density) *
                  get(lorentz_factor);

  get(*tilde_ye) = get(*tilde_d) * get(electron_fraction);

  get(*tilde_tau) = get(sqrt_det_spatial_metric) *
                      (square(get(lorentz_factor)) *
                        (get(rest_mass_density) *
                          (get(specific_internal_energy) +
                           get(spatial_velocity_squared) * get(lorentz_factor) /
                              (get(lorentz_factor) + 1.)) +
                         get(pressure) * get(spatial_velocity_squared)) +
                       0.5 * get(magnetic_field_squared) *
                        (1.0 + get(spatial_velocity_squared)) -
                       0.5 * square(get(magnetic_field_dot_spatial_velocity)));

  // Reuse allocation
  Scalar<DataType>& common_factor =
      get<hydro::Tags::MagneticFieldSquared<DataType>>(temp_tensors);
  get(common_factor) +=
      (get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
       get(pressure)) *
      square(get(lorentz_factor));
  get(common_factor) *= get(sqrt_det_spatial_metric);

  get(magnetic_field_dot_spatial_velocity) *= get(sqrt_det_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    tilde_s->get(i) = get(common_factor) * spatial_velocity_one_form.get(i) -
                      get(magnetic_field_dot_spatial_velocity) *
                          magnetic_field_one_form.get(i);
  }
  for (size_t i = 0; i < 3; ++i) {
    tilde_b->get(i) = get(sqrt_det_spatial_metric) * magnetic_field.get(i);
  }
  get(*tilde_phi) =
      get(sqrt_det_spatial_metric) * get(divergence_cleaning_field);
}

}  // namespace grmhd::ValenciaDivClean

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

GENERATE_INSTANTIATIONS(INSTANTIATE, (double))
// The `DataVector` version is never called on the device
#ifndef SPECTRE_KOKKOS_DEVICE_PASS
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))
#endif  // SPECTRE_KOKKOS_DEVICE_PASS

#undef INSTANTIATE
#undef DTYPE
