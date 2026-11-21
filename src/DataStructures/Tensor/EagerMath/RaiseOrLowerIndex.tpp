// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

template <typename DataType, typename Index0, typename Index1>
KOKKOS_FUNCTION void raise_or_lower_first_index(
    const gsl::not_null<
        Tensor<DataType, Symmetry<2, 1, 1>,
               index_list<change_index_up_lo<Index0>, Index1, Index1>>*>
        result,
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  constexpr auto dimension = Index0::dim;

  for (size_t i = 0; i < dimension; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      for (size_t k = j; k < dimension; ++k) {
        result->get(i, j, k) = tensor.get(0, j, k) * metric.get(i, 0);
        for (size_t m = 1; m < dimension; ++m) {
          result->get(i, j, k) += tensor.get(m, j, k) * metric.get(i, m);
        }
      }
    }
  }
}

template <typename DataTypeTensor, typename DataTypeMetric, typename Index0>
KOKKOS_FUNCTION void raise_or_lower_index(
    const gsl::not_null<Tensor<DataTypeTensor, Symmetry<1>,
                               index_list<change_index_up_lo<Index0>>>*>
        result,
    const Tensor<DataTypeTensor, Symmetry<1>, index_list<Index0>>& tensor,
    const Tensor<DataTypeMetric, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  constexpr auto dimension = Index0::dim;

  for (size_t i = 0; i < dimension; ++i) {
    result->get(i) = get<0>(tensor) * metric.get(i, 0);
    for (size_t m = 1; m < dimension; ++m) {
      result->get(i) += tensor.get(m) * metric.get(i, m);
    }
  }
}

template <typename DataType, typename Index0, typename Index1>
KOKKOS_FUNCTION Tensor<DataType, Symmetry<2, 1, 1>,
                       index_list<change_index_up_lo<Index0>, Index1, Index1>>
raise_or_lower_first_index(
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  Tensor<DataType, Symmetry<2, 1, 1>,
         index_list<change_index_up_lo<Index0>, Index1, Index1>>
      result{};
  raise_or_lower_first_index(make_not_null(&result), tensor, metric);
  return result;
}

template <typename DataTypeTensor, typename DataTypeMetric, typename Index0>
KOKKOS_FUNCTION
    Tensor<DataTypeTensor, Symmetry<1>, index_list<change_index_up_lo<Index0>>>
    raise_or_lower_index(
        const Tensor<DataTypeTensor, Symmetry<1>, index_list<Index0>>& tensor,
        const Tensor<DataTypeMetric, Symmetry<1, 1>,
                     index_list<change_index_up_lo<Index0>,
                                change_index_up_lo<Index0>>>& metric) {
  Tensor<DataTypeTensor, Symmetry<1>, index_list<change_index_up_lo<Index0>>>
      result{};
  raise_or_lower_index(make_not_null(&result), tensor, metric);
  return result;
}
