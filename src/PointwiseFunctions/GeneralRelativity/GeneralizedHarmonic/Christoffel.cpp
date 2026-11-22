// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.tpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                   \
  template void gh::christoffel_second_kind(                                   \
      const gsl::not_null<tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>*>     \
          christoffel,                                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,               \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_metric);        \
  template tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>                      \
  gh::christoffel_second_kind(                                                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,               \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_metric);        \
  template void gh::trace_christoffel(                                         \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>       \
          trace,                                                               \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_one_form,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_vector,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric,                                              \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spacetime_metric,                                            \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);              \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)> gh::trace_christoffel( \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_one_form,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_vector,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric,                                              \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spacetime_metric,                                            \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial,
                         Frame::Spherical<Frame::Inertial>,
                         Frame::Spherical<Frame::Grid>))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
