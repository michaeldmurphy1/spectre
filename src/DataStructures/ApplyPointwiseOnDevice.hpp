// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DeviceArguments.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/Tags/MirrorView.hpp"

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

/// @{
/*!
 * \brief Call the pointwise function `Function::apply` at every grid point in
 * a Kokkos kernel.
 *
 * `Function::apply` must be a `KOKKOS_FUNCTION` that takes
 * `gsl::not_null<Tensor<double>*>` results (the `ResultTags`) followed by the
 * arguments at a grid point (the `ArgsTags`, see `device_argument_at_index`).
 * The `args` are copied to the device (see `make_device_arguments`), and the
 * results are copied back into the `ResultTags` of `results`, which must
 * already have the right number of grid points. Other tags in `results` are
 * left unchanged.
 *
 * All copies and the kernel are enqueued on the execution space instance
 * `exec` (the default instance if not given). The instance is fenced before
 * returning, so `results` are ready to use on the host.
 */
template <typename Function, typename ExecSpace, typename VarsTags,
          typename... ResultTags, typename... ArgsTags>
  requires Kokkos::is_execution_space_v<ExecSpace>
void apply_pointwise_on_device(
    const ExecSpace& exec, const gsl::not_null<Variables<VarsTags>*> results,
    tmpl::list<ResultTags...> /*meta*/, tmpl::list<ArgsTags...> /*meta*/,
    const typename ArgsTags::type&... args) {
  const size_t num_points = results->number_of_grid_points();
  // Every grid point of the results is written by the kernel, so don't
  // initialize the memory
  const ApplyPointwiseOnDevice_detail::Functor<
      Function, tmpl::list<ResultTags...>, tmpl::list<ArgsTags...>>
      functor{
          {typename ::Tags::MirrorView<ResultTags>::type{
              Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "result"),
              num_points}...},
          make_device_arguments<ArgsTags...>(exec, args...)};
  Kokkos::parallel_for(
      "apply_pointwise_on_device",
      Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<size_t>>(exec, 0,
                                                                num_points),
      functor);
  (..., ApplyPointwiseOnDevice_detail::copy_result_to_host<ResultTags>(
            exec, make_not_null(&get<ResultTags>(*results)),
            get<::Tags::MirrorView<ResultTags>>(functor.results)));
  // The host arguments, host results and device views must stay alive until
  // all work on `exec` is complete
  exec.fence("apply_pointwise_on_device");
}

template <typename Function, typename VarsTags, typename... ResultTags,
          typename... ArgsTags>
void apply_pointwise_on_device(
    const gsl::not_null<Variables<VarsTags>*> results,
    tmpl::list<ResultTags...> result_tags, tmpl::list<ArgsTags...> args_tags,
    const typename ArgsTags::type&... args) {
  apply_pointwise_on_device<Function>(Kokkos::DefaultExecutionSpace{}, results,
                                      result_tags, args_tags, args...);
}
/// @}
#endif  // SPECTRE_KOKKOS
