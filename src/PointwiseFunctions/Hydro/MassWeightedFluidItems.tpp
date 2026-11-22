// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.hpp"

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/Math.hpp"

namespace hydro {
template <typename DataType, size_t Dim, typename Frame>
KOKKOS_FUNCTION void u_lower_t(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift) {
  dot_product(result, spatial_velocity, shift, spatial_metric);
  result->get() = get(lorentz_factor) * (get(lapse) * (-1.0) + result->get());
}

template <typename DataType>
KOKKOS_FUNCTION void mass_weighted_internal_energy(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d,
    const Scalar<DataType>& specific_internal_energy) {
  result->get() = get(tilde_d) * get(specific_internal_energy);
}

template <typename DataType>
KOKKOS_FUNCTION void mass_weighted_kinetic_energy(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor) {
  result->get() = get(tilde_d) * (get(lorentz_factor) - 1.0);
}

template <typename DataType, size_t Dim, typename Fr>
KOKKOS_FUNCTION void tilde_d_unbound_ut_criterion(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const tnsr::ii<DataType, Dim, Fr>& spatial_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift) {
  u_lower_t(result, lorentz_factor, spatial_velocity, spatial_metric, lapse,
            shift);
  result->get() = get(tilde_d) * step_function(-1.0 - result->get());
}
}  // namespace hydro
