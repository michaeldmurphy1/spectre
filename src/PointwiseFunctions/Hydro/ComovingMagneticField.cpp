// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.tpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace hydro {

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                               \
  template void comoving_magnetic_field(                                     \
      const gsl::not_null<tnsr::A<DTYPE(data), 3>*>                          \
          comoving_magnetic_field_result,                                    \
      const tnsr::I<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&,        \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const tnsr::I<DTYPE(data), 3>&, const Scalar<DTYPE(data)>&);           \
  template tnsr::A<DTYPE(data), 3> comoving_magnetic_field(                  \
      const tnsr::I<DTYPE(data), 3>&, const tnsr::I<DTYPE(data), 3>&,        \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const tnsr::I<DTYPE(data), 3>&, const Scalar<DTYPE(data)>&);           \
  template void comoving_magnetic_field_one_form(                            \
      const gsl::not_null<tnsr::a<DTYPE(data), 3>*>                          \
          comoving_magnetic_field_one_form_result,                           \
      const tnsr::i<DTYPE(data), 3>&, const tnsr::i<DTYPE(data), 3>&,        \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const tnsr::I<DTYPE(data), 3>&, const Scalar<DTYPE(data)>&);           \
  template tnsr::a<DTYPE(data), 3> comoving_magnetic_field_one_form(         \
      const tnsr::i<DTYPE(data), 3>&, const tnsr::i<DTYPE(data), 3>&,        \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const tnsr::I<DTYPE(data), 3>&, const Scalar<DTYPE(data)>&);           \
  template void comoving_magnetic_field_squared(                             \
      const gsl::not_null<Scalar<DTYPE(data)>*>, const Scalar<DTYPE(data)>&, \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&);               \
  template Scalar<DTYPE(data)> comoving_magnetic_field_squared(              \
      const Scalar<DTYPE(data)>&, const Scalar<DTYPE(data)>&,                \
      const Scalar<DTYPE(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)))

#undef DTYPE
#undef INSTANTIATION

}  // namespace hydro
