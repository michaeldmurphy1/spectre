// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DeviceArguments.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariables.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/VariablesKokkos.hpp"

namespace ApplyPointwiseOnDevice_detail {
// Kokkos functor that calls `Function::apply` at every grid point of every
// element of a batch. The members are exactly the data that is copied to the
// device.
template <typename Function, typename ResultTags, typename ArgsTags>
struct Functor;

template <typename Function, typename... ResultTags, typename... ArgsTags>
struct Functor<Function, tmpl::list<ResultTags...>, tmpl::list<ArgsTags...>> {
  // Returns what `Function::apply` returns (`void` or `bool`) at grid point
  // `i` of element `e`
  KOKKOS_FUNCTION auto apply_at(const size_t i, const size_t e) const {
    // Results at this grid point, on the stack of this thread
    tuples::TaggedTuple<::Tags::AtIndex<ResultTags>...> results_i{};
    const auto set_results = [this, &results_i, i, e]() {
      (...,
       set_at_index(make_not_null(&get<::Tags::BatchView<ResultTags>>(results)),
                    get<::Tags::AtIndex<ResultTags>>(results_i), i, e));
    };
    if constexpr (std::is_void_v<return_type>) {
      Function::apply(
          make_not_null(&get<::Tags::AtIndex<ResultTags>>(results_i))...,
          device_argument_at_index<ArgsTags>(args, i, e)...);
      set_results();
    } else {
      const bool result = Function::apply(
          make_not_null(&get<::Tags::AtIndex<ResultTags>>(results_i))...,
          device_argument_at_index<ArgsTags>(args, i, e)...);
      set_results();
      return result;
    }
  }

  // The flat index `k` runs over the grid points of each element in turn, so
  // consecutive threads access consecutive grid points
  // For `Kokkos::parallel_for`
  KOKKOS_FUNCTION void operator()(const size_t k) const {
    apply_at(k % number_of_grid_points, k / number_of_grid_points);
  }

  // For `Kokkos::parallel_reduce` with `Kokkos::LOr`
  KOKKOS_FUNCTION void operator()(const size_t k, bool& result) const {
    if (apply_at(k % number_of_grid_points, k / number_of_grid_points)) {
      result = true;
    }
  }

  using return_type = decltype(Function::apply(
      std::declval<
          gsl::not_null<typename ::Tags::AtIndex<ResultTags>::type*>>()...,
      device_argument_at_index<ArgsTags>(
          std::declval<const DeviceArguments<ArgsTags...>&>(), 0, 0)...));
  static_assert(std::is_void_v<return_type> or
                    std::is_same_v<return_type, bool>,
                "Function::apply must return void or bool");

  tuples::TaggedTuple<::Tags::BatchView<ResultTags>...> results;
  DeviceArguments<ArgsTags...> args;
  size_t number_of_grid_points;
};

// Launches the `Functor` over the batch `results`
template <typename Function, typename ExecSpace, typename... ResultTags,
          typename... ArgsTags>
auto apply_on_batch(
    const ExecSpace& exec,
    tuples::TaggedTuple<::Tags::BatchView<ResultTags>...> results,
    const size_t number_of_grid_points, const size_t number_of_elements,
    tmpl::list<ArgsTags...> /*meta*/,
    const DeviceArguments<ArgsTags...>& args) {
#ifdef SPECTRE_DEBUG
  tmpl::for_each<tmpl::list<ArgsTags...>>([&args, number_of_grid_points,
                                           number_of_elements](auto tag_v) {
    using Tag = tmpl::type_from<decltype(tag_v)>;
    if constexpr (DeviceArguments_detail::is_tensor_v<Tag>) {
      for (const auto& component : get<::Tags::BatchView<Tag>>(args)) {
        ASSERT(component.extent(0) == number_of_grid_points and
                   component.extent(1) == number_of_elements,
               "The argument " << pretty_type::get_name<Tag>() << " has "
                               << component.extent(0) << " grid points and "
                               << component.extent(1)
                               << " elements, but the results have "
                               << number_of_grid_points << " grid points and "
                               << number_of_elements << " elements");
      }
    } else if constexpr (DeviceArguments_detail::is_per_element<Tag>::value) {
      const auto& values = get<device_argument_tag<Tag>>(args);
      ASSERT(values.extent(0) == number_of_elements,
             "The argument "
                 << pretty_type::get_name<Tag>() << " has " << values.extent(0)
                 << " values, but the results have " << number_of_elements
                 << " elements");
    }
  });
#endif  // SPECTRE_DEBUG
  // The functor holds the views of the results (not copies of the data), so
  // the kernel writes directly into `results`
  const Functor<Function, tmpl::list<ResultTags...>, tmpl::list<ArgsTags...>>
      functor{std::move(results), args, number_of_grid_points};
  const Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<size_t>> policy(
      exec, 0, number_of_grid_points * number_of_elements);
  if constexpr (std::is_void_v<typename decltype(functor)::return_type>) {
    Kokkos::parallel_for("apply_pointwise_on_device", policy, functor);
  } else {
    bool result = false;
    Kokkos::parallel_reduce("apply_pointwise_on_device", policy, functor,
                            Kokkos::LOr<bool>(result));
    return result;
  }
}

// Enqueue the copy of the device tensor into the (already allocated) host
// tensor on `exec`
template <typename Tag, typename ExecSpace>
void copy_result_to_host(
    const ExecSpace& exec, const gsl::not_null<typename Tag::type*> host_tensor,
    const typename ::Tags::MirrorView<Tag>::type& device_tensor) {
  using HostUnmanaged =
      Kokkos::View<typename Tag::type::type::value_type*, Kokkos::HostSpace,
                   Kokkos::MemoryUnmanaged>;
  for (size_t i = 0; i < host_tensor->size(); ++i) {
    HostUnmanaged host_view((*host_tensor)[i].data(), (*host_tensor)[i].size());
    Kokkos::deep_copy(exec, host_view, device_tensor[i]);
  }
}
}  // namespace ApplyPointwiseOnDevice_detail

/// @{
/*!
 * \brief Call the pointwise function `Function::apply` at every grid point of
 * data that is already in device memory.
 *
 * `Function::apply` must be a `KOKKOS_FUNCTION` that takes
 * `gsl::not_null<Tensor<double>*>` results (the `ResultTags`) followed by the
 * arguments at a grid point (the `ArgsTags`, see `device_argument_at_index`).
 * The `results` are either
 * - a `BatchedVariables` of a batch of elements, in which case the function is
 *   called at every grid point of every element in a single kernel and the
 *   tensor arguments must be batches with the same number of grid points and
 *   elements, or
 * - a device `Variables` of one element (e.g. allocated with
 *   `Variables<tmpl::list<::Tags::MirrorView<Tags>...>>` or returned by
 *   `copy_to_device`), which is treated as a batch of one element.
 *
 * The results are written to the `ResultTags` of `results`. Other tags in
 * `results` are left unchanged. The `args` must already be in device memory
 * (see `DeviceArguments`).
 *
 * Nothing is copied and `exec` is not fenced: the kernel is only enqueued on
 * `exec`, so later kernels on the same instance can use the results directly.
 * Use this for data that stays on the device, and
 * `copy_and_apply_pointwise_on_device` for data on the host.
 *
 * If `Function::apply` returns a `bool` (e.g. whether it changed anything),
 * this returns whether it returned `true` at any grid point. Since the result
 * is needed on the host, this waits for the kernel to complete.
 */
template <typename Function, typename ExecSpace, typename BatchTags,
          typename... ResultTags, typename... ArgsTags>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto apply_pointwise_on_device(
    const ExecSpace& exec,
    const gsl::not_null<BatchedVariables<BatchTags>*> results,
    tmpl::list<ResultTags...> /*meta*/, tmpl::list<ArgsTags...> args_tags,
    const DeviceArguments<ArgsTags...>& args) {
  return ApplyPointwiseOnDevice_detail::apply_on_batch<Function>(
      exec,
      tuples::TaggedTuple<::Tags::BatchView<ResultTags>...>{
          get<::Tags::BatchView<ResultTags>>(*results)...},
      results->number_of_grid_points(), results->number_of_elements(),
      args_tags, args);
}

template <typename Function, typename ExecSpace, typename DeviceVarsTags,
          typename... ResultTags, typename... ArgsTags>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto apply_pointwise_on_device(
    const ExecSpace& exec,
    const gsl::not_null<Variables<DeviceVarsTags>*> results,
    tmpl::list<ResultTags...> /*meta*/, tmpl::list<ArgsTags...> args_tags,
    const DeviceArguments<ArgsTags...>& args) {
  return ApplyPointwiseOnDevice_detail::apply_on_batch<Function>(
      exec,
      tuples::TaggedTuple<::Tags::BatchView<ResultTags>...>{
          DeviceArguments_detail::as_batch_of_one<ResultTags>(
              get<::Tags::MirrorView<ResultTags>>(*results))...},
      results->number_of_grid_points(), 1, args_tags, args);
}
/// @}

/// @{
/*!
 * \brief Call the pointwise function `Function::apply` at every grid point of
 * host data in a Kokkos kernel.
 *
 * `Function::apply` must be a `KOKKOS_FUNCTION` that takes
 * `gsl::not_null<Tensor<double>*>` results (the `ResultTags`) followed by the
 * arguments at a grid point (the `ArgsTags`, see `device_argument_at_index`).
 * The `args` are copied to the device (see `make_device_arguments`), the kernel
 * is launched with `apply_pointwise_on_device`, and the results are copied
 * back into the `ResultTags` of `results`, which must already have the right
 * number of grid points. Other tags in `results` are left unchanged.
 *
 * All copies and the kernel are enqueued on the execution space instance
 * `exec` (the default instance if not given). The instance is fenced before
 * returning, so `results` are ready to use on the host.
 *
 * If `Function::apply` returns a `bool`, this returns whether it returned
 * `true` at any grid point (see `apply_pointwise_on_device`).
 *
 * \note The copies make this convenient for code that keeps its data on the
 * host, and for tests. Code that keeps its data on the device should call
 * `apply_pointwise_on_device` instead.
 */
template <typename Function, typename ExecSpace, typename VarsTags,
          typename... ResultTags, typename... ArgsTags>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto copy_and_apply_pointwise_on_device(
    const ExecSpace& exec, const gsl::not_null<Variables<VarsTags>*> results,
    tmpl::list<ResultTags...> result_tags, tmpl::list<ArgsTags...> args_tags,
    const typename ArgsTags::type&... args) {
  using DeviceResults =
      Variables<tmpl::list<::Tags::MirrorView<ResultTags>...>>;
  // Every grid point of the results is written by the kernel, so don't
  // initialize the memory
  DeviceResults device_results{typename DeviceResults::storage_type{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "results"),
      results->number_of_grid_points()}};
  const auto device_args = make_device_arguments<ArgsTags...>(exec, args...);
  const auto copy_results_and_fence = [&exec, &results, &device_results]() {
    (..., ApplyPointwiseOnDevice_detail::copy_result_to_host<ResultTags>(
              exec, make_not_null(&get<ResultTags>(*results)),
              get<::Tags::MirrorView<ResultTags>>(device_results)));
    // The host arguments, host results and device views must stay alive until
    // all work on `exec` is complete
    exec.fence("copy_and_apply_pointwise_on_device");
  };
  if constexpr (std::is_void_v<decltype(apply_pointwise_on_device<Function>(
                    exec, make_not_null(&device_results), result_tags,
                    args_tags, device_args))>) {
    apply_pointwise_on_device<Function>(exec, make_not_null(&device_results),
                                        result_tags, args_tags, device_args);
    copy_results_and_fence();
  } else {
    const bool result = apply_pointwise_on_device<Function>(
        exec, make_not_null(&device_results), result_tags, args_tags,
        device_args);
    copy_results_and_fence();
    return result;
  }
}

template <typename Function, typename VarsTags, typename... ResultTags,
          typename... ArgsTags>
auto copy_and_apply_pointwise_on_device(
    const gsl::not_null<Variables<VarsTags>*> results,
    tmpl::list<ResultTags...> result_tags, tmpl::list<ArgsTags...> args_tags,
    const typename ArgsTags::type&... args) {
  return copy_and_apply_pointwise_on_device<Function>(
      Kokkos::DefaultExecutionSpace{}, results, result_tags, args_tags,
      args...);
}
/// @}
#endif  // SPECTRE_KOKKOS
