// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.tpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>   \
  gr::christoffel_first_kind(                                                \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric);                                                         \
  template void gr::christoffel_first_kind(                                  \
      const gsl::not_null<                                                   \
          tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>*>  \
          christoffel,                                                       \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric);                                                         \
  template tnsr::Abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>   \
  gr::christoffel_second_kind(                                               \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric,                                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          inverse_metric);                                                   \
  template void gr::christoffel_second_kind(                                 \
      const gsl::not_null<                                                   \
          tnsr::Abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>*>  \
          christoffel,                                                       \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric,                                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          inverse_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial,
                         Frame::Spherical<Frame::Inertial>,
                         Frame::Spherical<Frame::Distorted>,
                         Frame::Spherical<Frame::Grid>),
                        (IndexType::Spatial, IndexType::Spacetime))

#undef INSTANTIATE

#define INSTANTIATE_TRACE(_, data)                                            \
  template void gr::trace_spatial_christoffel_first_kind(                     \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*>      \
          result,                                                             \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& d_spatial_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE_TRACE, (1, 2, 3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE_TRACE
