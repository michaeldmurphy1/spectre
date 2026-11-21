// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Trace.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

template <typename DataType, typename Index0, typename Index1>
KOKKOS_FUNCTION void trace_last_indices(
    const gsl::not_null<Tensor<DataType, Symmetry<1>, index_list<Index0>>*>
        trace_of_tensor,
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index1>,
                            change_index_up_lo<Index1>>>& metric) {
  constexpr auto dimension_of_trace = Index0::dim;
  constexpr auto dimension_of_metric = Index1::dim;
  for (size_t i = 0; i < dimension_of_trace; ++i) {
    trace_of_tensor->get(i) = tensor.get(i, 0, 0) * get<0, 0>(metric);
    for (size_t j = 0; j < dimension_of_metric; ++j) {
      if (j != 0) {
        trace_of_tensor->get(i) += tensor.get(i, j, j) * metric.get(j, j);
      }
      for (size_t k = j + 1; k < dimension_of_metric; ++k) {
        trace_of_tensor->get(i) += 2.0 * tensor.get(i, j, k) * metric.get(j, k);
      }
    }
  }
}

template <typename DataType, typename Index0>
KOKKOS_FUNCTION void trace(
    const gsl::not_null<Scalar<DataType>*> trace,
    const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  constexpr auto dimension = Index0::dim;
  get(*trace) = get<0, 0>(tensor) * get<0, 0>(metric);
  for (size_t i = 0; i < dimension; ++i) {
    if (i != 0) {
      get(*trace) += tensor.get(i, i) * metric.get(i, i);
    }
    for (size_t j = i + 1; j < dimension; ++j) {  // symmetry
      get(*trace) += 2.0 * tensor.get(i, j) * metric.get(i, j);
    }
  }
}

template <typename DataType, typename Index0, typename Index1>
KOKKOS_FUNCTION Tensor<DataType, Symmetry<1>, index_list<Index0>>
trace_last_indices(
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index1>,
                            change_index_up_lo<Index1>>>& metric) {
  Tensor<DataType, Symmetry<1>, index_list<Index0>> trace_of_tensor{};
  trace_last_indices(make_not_null(&trace_of_tensor), tensor, metric);
  return trace_of_tensor;
}

template <typename DataType, typename Index0>
KOKKOS_FUNCTION Scalar<DataType> trace(
    const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  Scalar<DataType> trace{};
  ::trace(make_not_null(&trace), tensor, metric);
  return trace;
}
