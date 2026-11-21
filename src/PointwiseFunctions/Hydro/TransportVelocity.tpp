// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace hydro {
template <typename DataType, size_t Dim, typename Fr>
KOKKOS_FUNCTION void transport_velocity(
    const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift) {
  set_number_of_grid_points(result, lapse);
  for (size_t i = 0; i < Dim; i++) {
    result->get(i) = spatial_velocity.get(i) * get(lapse) - shift.get(i);
  }
}
}  // namespace hydro
