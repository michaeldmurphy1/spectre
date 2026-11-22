// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> lapse(
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) {
  Scalar<DataType> local_lapse{get_size(get<0, 0>(spacetime_metric))};
  lapse(make_not_null(&local_lapse), shift, spacetime_metric);
  return local_lapse;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void lapse(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) {
  get(*lapse) = -get<0, 0>(spacetime_metric);
  for (size_t i = 0; i < SpatialDim; ++i) {
    get(*lapse) += shift.get(i) * spacetime_metric.get(i + 1, 0);
  }
  get(*lapse) = sqrt(get(*lapse));
}
}  // namespace gr
