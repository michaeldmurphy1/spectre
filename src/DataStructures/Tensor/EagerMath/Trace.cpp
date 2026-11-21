// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.tpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)
#define UPORLO0(data) BOOST_PP_TUPLE_ELEM(4, data)
#define UPORLO1(data) BOOST_PP_TUPLE_ELEM(5, data)

// Note: currently only instantiate these functions for the cases
// where Index0 and Index1 have the same spatial dimension, Frame and
// IndexType.  The functions in Trace.tpp are generic and can handle any of
// them being different, if these cases are needed.
#define INDEX0(data) INDEXTYPE(data)<DIM(data), UPORLO0(data), FRAME(data)>
#define INDEX1(data) INDEXTYPE(data)<DIM(data), UPORLO1(data), FRAME(data)>

#define INSTANTIATE(_, data)                                               \
  template void trace_last_indices(                                        \
      const gsl::not_null<                                                 \
          Tensor<DTYPE(data), Symmetry<1>, index_list<INDEX0(data)>>*>     \
          result,                                                          \
      const Tensor<DTYPE(data), Symmetry<2, 1, 1>,                         \
                   index_list<INDEX0(data), INDEX1(data), INDEX1(data)>>&  \
          tensor,                                                          \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                            \
                   index_list<change_index_up_lo<INDEX1(data)>,            \
                              change_index_up_lo<INDEX1(data)>>>& metric); \
  template Tensor<DTYPE(data), Symmetry<1>, index_list<INDEX0(data)>>      \
  trace_last_indices(                                                      \
      const Tensor<DTYPE(data), Symmetry<2, 1, 1>,                         \
                   index_list<INDEX0(data), INDEX1(data), INDEX1(data)>>&  \
          tensor,                                                          \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                            \
                   index_list<change_index_up_lo<INDEX1(data)>,            \
                              change_index_up_lo<INDEX1(data)>>>& metric);

GENERATE_INSTANTIATIONS(
    INSTANTIATE, (1, 2, 3),
    SPECTRE_DEVICE_AND_HOST_TYPES((double), (double, DataVector)),
    (Frame::Grid, Frame::Distorted, Frame::Inertial,
     Frame::Spherical<Frame::Inertial>, Frame::Spherical<Frame::Distorted>,
     Frame::Spherical<Frame::Grid>),
    (SpatialIndex, SpacetimeIndex), (UpLo::Lo, UpLo::Up), (UpLo::Lo, UpLo::Up))

#undef FRAME1
#undef UPORLO1
#undef INDEXTYPE1
#undef INDEX1
#undef INSTANTIATE

#define INSTANTIATE2(_, data)                                              \
  template void trace(                                                     \
      const gsl::not_null<Scalar<DTYPE(data)>*> trace,                     \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                            \
                   index_list<INDEX0(data), INDEX0(data)>>& tensor,        \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                            \
                   index_list<change_index_up_lo<INDEX0(data)>,            \
                              change_index_up_lo<INDEX0(data)>>>& metric); \
  template Scalar<DTYPE(data)> trace(                                      \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                            \
                   index_list<INDEX0(data), INDEX0(data)>>& tensor,        \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                            \
                   index_list<change_index_up_lo<INDEX0(data)>,            \
                              change_index_up_lo<INDEX0(data)>>>& metric);

GENERATE_INSTANTIATIONS(INSTANTIATE2, (1, 2, 3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial),
                        (SpatialIndex, SpacetimeIndex), (UpLo::Lo, UpLo::Up))

#undef DIM
#undef DTYPE
#undef FRAME0
#undef UPORLO0
#undef INDEXTYPE0
#undef INDEX0
#undef INSTANTIATE2
