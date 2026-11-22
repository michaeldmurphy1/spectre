// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::A<DataType, SpatialDim, Frame> spacetime_normal_vector(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) {
  tnsr::A<DataType, SpatialDim, Frame> local_spacetime_normal_vector{
      get_size(get(lapse))};
  spacetime_normal_vector(make_not_null(&local_spacetime_normal_vector), lapse,
                          shift);
  return local_spacetime_normal_vector;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void spacetime_normal_vector(
    const gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>
        spacetime_normal_vector,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) {
  get<0>(*spacetime_normal_vector) = 1. / get(lapse);
  for (size_t i = 0; i < SpatialDim; i++) {
    spacetime_normal_vector->get(i + 1) =
        -shift.get(i) * get<0>(*spacetime_normal_vector);
  }
}
}  // namespace gr
