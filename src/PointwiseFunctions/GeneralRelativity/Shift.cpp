// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.tpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> gr::shift(           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric);                                           \
  template void gr::shift(                                                   \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>     \
          shift,                                                             \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
