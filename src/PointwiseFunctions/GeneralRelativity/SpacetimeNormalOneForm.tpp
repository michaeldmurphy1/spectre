// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void spacetime_normal_one_form(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> normal_one_form,
    const Scalar<DataType>& lapse) {
  set_number_of_grid_points(normal_one_form, lapse);
  for (auto& component : *normal_one_form) {
    component = 0.0;
  }
  get<0>(*normal_one_form) = -get(lapse);
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse) {
  tnsr::a<DataType, SpatialDim, Frame> normal_one_form{};
  spacetime_normal_one_form(make_not_null(&normal_one_form), lapse);
  return normal_one_form;
}
}  // namespace gr
