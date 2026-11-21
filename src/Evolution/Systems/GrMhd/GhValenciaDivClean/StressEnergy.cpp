// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.tpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace grmhd::GhValenciaDivClean {

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
