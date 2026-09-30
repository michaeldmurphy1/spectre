// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

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
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/VariablesKokkos.hpp"

namespace ApplyPointwiseOnDevice_detail {
// Kokkos functor that calls `Function::apply` at grid point `i`. The members
// are exactly the data that is copied to the device.
template <typename Function, typename ResultTags, typename ArgsTags>
struct Functor;

template <typename Function, typename... ResultTags, typename... ArgsTags>
struct Functor<Function, tmpl::list<ResultTags...>, tmpl::list<ArgsTags...>> {
  KOKKOS_FUNCTION void operator()(const size_t i) const {
    // Results at this grid point, on the stack of this thread
    tuples::TaggedTuple<::Tags::AtIndex<ResultTags>...> results_i{};
    Function::apply(
        make_not_null(&get<::Tags::AtIndex<ResultTags>>(results_i))...,
        device_argument_at_index<ArgsTags>(args, i)...);
    (...,
     set_at_index(make_not_null(&get<::Tags::MirrorView<ResultTags>>(results)),
                  get<::Tags::AtIndex<ResultTags>>(results_i), i));
  }

  tuples::TaggedTuple<::Tags::MirrorView<ResultTags>...> results;
  DeviceArguments<ArgsTags...> args;
};

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

/*!
 * \brief Call the pointwise function `Function::apply` at every grid point of
 * data that is already in device memory.
 *
 * `Function::apply` must be a `KOKKOS_FUNCTION` that takes
 * `gsl::not_null<Tensor<double>*>` results (the `ResultTags`) followed by the
 * arguments at a grid point (the `ArgsTags`, see `device_argument_at_index`).
 * The results are written to the `::Tags::MirrorView<ResultTags>` of the
 * device `Variables` `results` (e.g. allocated with `Variables<tmpl::list<
 * ::Tags::MirrorView<Tags>...>>` or returned by `copy_to_device`). Other tags
 * in `results` are left unchanged. The `args` must already be in device memory
 * (see `DeviceArguments`).
 *
 * Nothing is copied and `exec` is not fenced: the kernel is only enqueued on
 * `exec`, so later kernels on the same instance can use the results directly.
 * Use this for data that stays on the device, and
 * `copy_and_apply_pointwise_on_device` for data on the host.
 */
template <typename Function, typename ExecSpace, typename DeviceVarsTags,
          typename... ResultTags, typename... ArgsTags>
  requires Kokkos::is_execution_space_v<ExecSpace>
void apply_pointwise_on_device(
    const ExecSpace& exec,
    const gsl::not_null<Variables<DeviceVarsTags>*> results,
    tmpl::list<ResultTags...> /*meta*/, tmpl::list<ArgsTags...> /*meta*/,
    const DeviceArguments<ArgsTags...>& args) {
  const size_t num_points = results->number_of_grid_points();
#ifdef SPECTRE_DEBUG
  tmpl::for_each<tmpl::list<ArgsTags...>>([&args, num_points](auto tag_v) {
    using Tag = tmpl::type_from<decltype(tag_v)>;
    if constexpr (DeviceArguments_detail::is_tensor_v<Tag>) {
      for (const auto& component : get<::Tags::MirrorView<Tag>>(args)) {
        ASSERT(component.extent(0) == num_points,
               "The argument " << pretty_type::get_name<Tag>() << " has "
                               << component.extent(0)
                               << " grid points, but the results have "
                               << num_points);
      }
    }
  });
#endif  // SPECTRE_DEBUG
  // The functor holds the views of the results (not copies of the data), so
  // the kernel writes directly into `results`
  const ApplyPointwiseOnDevice_detail::Functor<
      Function, tmpl::list<ResultTags...>, tmpl::list<ArgsTags...>>
      functor{{get<::Tags::MirrorView<ResultTags>>(*results)...}, args};
  Kokkos::parallel_for(
      "apply_pointwise_on_device",
      Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<size_t>>(exec, 0,
                                                                num_points),
      functor);
}

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
 * \note The copies make this convenient for code that keeps its data on the
 * host, and for tests. Code that keeps its data on the device should call
 * `apply_pointwise_on_device` instead.
 */
template <typename Function, typename ExecSpace, typename VarsTags,
          typename... ResultTags, typename... ArgsTags>
  requires Kokkos::is_execution_space_v<ExecSpace>
void copy_and_apply_pointwise_on_device(
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
  apply_pointwise_on_device<Function>(exec, make_not_null(&device_results),
                                      result_tags, args_tags, device_args);
  (..., ApplyPointwiseOnDevice_detail::copy_result_to_host<ResultTags>(
            exec, make_not_null(&get<ResultTags>(*results)),
            get<::Tags::MirrorView<ResultTags>>(device_results)));
  // The host arguments, host results and device views must stay alive until
  // all work on `exec` is complete
  exec.fence("copy_and_apply_pointwise_on_device");
}

template <typename Function, typename VarsTags, typename... ResultTags,
          typename... ArgsTags>
void copy_and_apply_pointwise_on_device(
    const gsl::not_null<Variables<VarsTags>*> results,
    tmpl::list<ResultTags...> result_tags, tmpl::list<ArgsTags...> args_tags,
    const typename ArgsTags::type&... args) {
  copy_and_apply_pointwise_on_device<Function>(Kokkos::DefaultExecutionSpace{},
                                               results, result_tags, args_tags,
                                               args...);
}
/// @}
#endif  // SPECTRE_KOKKOS
