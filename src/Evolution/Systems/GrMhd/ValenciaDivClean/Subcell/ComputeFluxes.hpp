// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/BatchedVariables.hpp"
#include "DataStructures/DeviceArguments.hpp"
#endif  // SPECTRE_KOKKOS

namespace grmhd::ValenciaDivClean::subcell {
namespace detail {
template <typename TagsList, typename... ReturnTags, typename... ArgumentTags>
void compute_fluxes_impl(const gsl::not_null<Variables<TagsList>*> vars,
                         tmpl::list<ReturnTags...> return_tags,
                         tmpl::list<ArgumentTags...> argument_tags) {
#ifdef SPECTRE_KOKKOS
  copy_and_apply_pointwise_on_device<grmhd::ValenciaDivClean::ComputeFluxes>(
      vars, return_tags, argument_tags, get<ArgumentTags>(*vars)...);
#else
  (void)return_tags;
  (void)argument_tags;
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&get<ReturnTags>(*vars))..., get<ArgumentTags>(*vars)...);
#endif  // SPECTRE_KOKKOS
}

#ifdef SPECTRE_KOKKOS
// Copies the tensors `Tags` between two host batches with different tags
template <typename DestinationTags, typename SourceTags, typename... Tags>
void copy_tags(
    const gsl::not_null<BatchedVariables<DestinationTags, Kokkos::HostSpace>*>
        destination,
    const BatchedVariables<SourceTags, Kokkos::HostSpace>& source,
    tmpl::list<Tags...> /*meta*/) {
  (..., [&destination, &source]() {
    auto& destination_tensor =
        get<::Tags::BatchView<Tags, Kokkos::HostSpace>>(*destination);
    const auto& source_tensor =
        get<::Tags::BatchView<Tags, Kokkos::HostSpace>>(source);
    for (size_t i = 0; i < destination_tensor.size(); ++i) {
      Kokkos::deep_copy(destination_tensor[i], source_tensor[i]);
    }
  }());
}

template <typename ExecSpace, typename TagsList, typename... ReturnTags,
          typename... ArgumentTags>
void compute_fluxes_impl(
    const ExecSpace& exec,
    const gsl::not_null<BatchedVariables<TagsList, Kokkos::HostSpace>*>
        host_batch,
    tmpl::list<ReturnTags...> return_tags,
    tmpl::list<ArgumentTags...> argument_tags) {
  const size_t number_of_grid_points = host_batch->number_of_grid_points();
  const size_t number_of_elements = host_batch->number_of_elements();
  // Only the arguments are copied to the device and only the fluxes are
  // copied back, each in a single copy. They are packed into (and unpacked
  // from) contiguous host batches for this.
  BatchedVariables<tmpl::list<ArgumentTags...>, Kokkos::HostSpace> host_args{
      number_of_grid_points, number_of_elements};
  copy_tags(make_not_null(&host_args), *host_batch, argument_tags);
  BatchedVariables<tmpl::list<ArgumentTags...>> device_args{
      exec, number_of_grid_points, number_of_elements};
  deep_copy(exec, make_not_null(&device_args), host_args);
  BatchedVariables<tmpl::list<ReturnTags...>> device_fluxes{
      exec, number_of_grid_points, number_of_elements};
  apply_pointwise_on_device<grmhd::ValenciaDivClean::ComputeFluxes>(
      exec, make_not_null(&device_fluxes), return_tags, argument_tags,
      make_device_arguments<ArgumentTags...>(
          exec, get<::Tags::BatchView<ArgumentTags>>(device_args)...));
  BatchedVariables<tmpl::list<ReturnTags...>, Kokkos::HostSpace> host_fluxes{
      number_of_grid_points, number_of_elements};
  deep_copy(exec, make_not_null(&host_fluxes), device_fluxes);
  // The host and device batches must stay alive until all work on `exec` is
  // complete
  exec.fence("compute_fluxes");
  copy_tags(host_batch, host_fluxes, return_tags);
}
#endif  // SPECTRE_KOKKOS
}  // namespace detail

/*!
 * \brief Helper function that calls `ComputeFluxes` by retrieving the return
 * and argument tags from `vars`.
 *
 * With Kokkos, the fluxes are computed pointwise in a Kokkos kernel.
 */
template <typename TagsList>
void compute_fluxes(const gsl::not_null<Variables<TagsList>*> vars) {
  detail::compute_fluxes_impl(
      vars, typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags{},
      typename grmhd::ValenciaDivClean::ComputeFluxes::argument_tags{});
}

#ifdef SPECTRE_KOKKOS
/// @{
/*!
 * \brief Calls `ComputeFluxes` at every grid point of every element of the
 * host batch `vars` in a single Kokkos kernel.
 *
 * Only the arguments of `ComputeFluxes` are copied to the device and only the
 * fluxes are copied back, each in one copy. The copies and the kernel are
 * enqueued on the execution space instance `exec` (the default instance if not
 * given), and this function blocks until they are complete.
 */
template <typename ExecSpace, typename TagsList>
  requires Kokkos::is_execution_space_v<ExecSpace>
void compute_fluxes(
    const ExecSpace& exec,
    const gsl::not_null<BatchedVariables<TagsList, Kokkos::HostSpace>*> vars) {
  detail::compute_fluxes_impl(
      exec, vars,
      typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags{},
      typename grmhd::ValenciaDivClean::ComputeFluxes::argument_tags{});
}

template <typename TagsList>
void compute_fluxes(
    const gsl::not_null<BatchedVariables<TagsList, Kokkos::HostSpace>*> vars) {
  compute_fluxes(Kokkos::DefaultExecutionSpace{}, vars);
}
/// @}
#endif  // SPECTRE_KOKKOS
}  // namespace grmhd::ValenciaDivClean::subcell
