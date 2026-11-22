// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.tpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                             \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                  \
  gr::spacetime_normal_one_form(const Scalar<DTYPE(data)>& lapse);       \
  template void gr::spacetime_normal_one_form(                           \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*> \
          normal_one_form,                                               \
      const Scalar<DTYPE(data)>& lapse);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
