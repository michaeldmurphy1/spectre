// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace grmhd::GhValenciaDivClean {

namespace {
// spatial components of the four-velocity are:
// u^i = W(v^i - \beta^i / \alpha)
// so the down-index spatial components are
// u_i = W v_i
//
// u_0 = - \alpha W + \beta^i u_i
template <typename DataType>
KOKKOS_FUNCTION void four_velocity_one_form(
    const gsl::not_null<tnsr::a<DataType, 3>*> four_velocity_one_form_result,
    const tnsr::i<DataType, 3, Frame::Inertial>& spatial_velocity_one_form,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const Scalar<DataType>& lapse) {
  get<0>(*four_velocity_one_form_result) = -get(lapse) * get(lorentz_factor);
  for (size_t i = 0; i < 3; ++i) {
    four_velocity_one_form_result->get(i + 1) =
        get(lorentz_factor) * spatial_velocity_one_form.get(i);
    get<0>(*four_velocity_one_form_result) +=
        shift.get(i) * four_velocity_one_form_result->get(i + 1);
  }
}
}  // namespace

template <typename DataType>
KOKKOS_FUNCTION void add_stress_energy_term_to_dt_pi(
    const gsl::not_null<tnsr::aa<DataType, 3>*> dt_pi,
    const tnsr::aa<DataType, 3>& trace_reversed_stress_energy,
    const Scalar<DataType>& lapse) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      dt_pi->get(a, b) -=
          16.0 * M_PI * get(lapse) * trace_reversed_stress_energy.get(a, b);
    }
  }
}

template <typename DataType>
KOKKOS_FUNCTION void trace_reversed_stress_energy(
    const gsl::not_null<tnsr::aa<DataType, 3>*> stress_energy,
    const gsl::not_null<tnsr::a<DataType, 3>*> four_velocity_one_form_buffer,
    const gsl::not_null<tnsr::a<DataType, 3>*>
        comoving_magnetic_field_one_form_buffer,
    const Scalar<DataType>& rest_mass_density,
    const tnsr::i<DataType, 3, Frame::Inertial>& spatial_velocity_one_form,
    const tnsr::i<DataType, 3, Frame::Inertial>& magnetic_field_one_form,
    const Scalar<DataType>& magnetic_field_squared,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& one_over_w_squared,
    const Scalar<DataType>& pressure,
    const Scalar<DataType>& specific_internal_energy,
    const tnsr::aa<DataType, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const Scalar<DataType>& lapse) {
  four_velocity_one_form(four_velocity_one_form_buffer,
                         spatial_velocity_one_form, lorentz_factor, shift,
                         lapse);

  // 3,3 component will be the last component assigned, so we can use it as a
  // temporary buffer
  auto& modified_enthalpy_times_rest_mass = get<3, 3>(*stress_energy);
  modified_enthalpy_times_rest_mass =
      get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
      get(pressure) + square(get(magnetic_field_dot_spatial_velocity)) +
      get(magnetic_field_squared) * get(one_over_w_squared);

  hydro::comoving_magnetic_field_one_form(
      comoving_magnetic_field_one_form_buffer, spatial_velocity_one_form,
      magnetic_field_one_form, magnetic_field_dot_spatial_velocity,
      lorentz_factor, shift, lapse);
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      stress_energy->get(a, b) =
          modified_enthalpy_times_rest_mass *
              four_velocity_one_form_buffer->get(a) *
              four_velocity_one_form_buffer->get(b) +
          (0.5 * modified_enthalpy_times_rest_mass - get(pressure)) *
              spacetime_metric.get(a, b) -
          comoving_magnetic_field_one_form_buffer->get(a) *
              comoving_magnetic_field_one_form_buffer->get(b);
    }
  }
}
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                                 \
  template void add_stress_energy_term_to_dt_pi(                               \
      const gsl::not_null<tnsr::aa<DTYPE(data), 3>*> dt_pi,                    \
      const tnsr::aa<DTYPE(data), 3>& trace_reversed_stress_energy,            \
      const Scalar<DTYPE(data)>& lapse);                                       \
  template void trace_reversed_stress_energy(                                  \
      const gsl::not_null<tnsr::aa<DTYPE(data), 3>*> stress_energy,            \
      const gsl::not_null<tnsr::a<DTYPE(data), 3>*>                            \
          four_velocity_one_form_buffer,                                       \
      const gsl::not_null<tnsr::a<DTYPE(data), 3>*>                            \
          comoving_magnetic_field_one_form_buffer,                             \
      const Scalar<DTYPE(data)>& rest_mass_density,                            \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>&                          \
          spatial_velocity_one_form,                                           \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& magnetic_field_one_form, \
      const Scalar<DTYPE(data)>& magnetic_field_squared,                       \
      const Scalar<DTYPE(data)>& magnetic_field_dot_spatial_velocity,          \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const Scalar<DTYPE(data)>& one_over_w_squared,                           \
      const Scalar<DTYPE(data)>& pressure,                                     \
      const Scalar<DTYPE(data)>& specific_internal_energy,                     \
      const tnsr::aa<DTYPE(data), 3, Frame::Inertial>& spacetime_metric,       \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const Scalar<DTYPE(data)>& lapse);

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)))

#undef DTYPE
#undef INSTANTIATION
}  // namespace grmhd::GhValenciaDivClean
