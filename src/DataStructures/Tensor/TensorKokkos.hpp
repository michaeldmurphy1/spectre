// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// @{
/*!
 * \brief Copy a `Tensor<DataVector>` to device memory.
 *
 * The allocation and copies are enqueued on the execution space instance
 * `exec`, so work that uses the result must run on the same instance (or fence
 * it first), and `tensor_host` must stay alive until `exec` is fenced. Without
 * `exec` the default instance is used and this function blocks until the copy
 * is complete, like `Kokkos::deep_copy`.
 */
template <typename ExecSpace, typename HostVectorType, typename... Properties>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto copy_to_device(const ExecSpace& exec,
                    const Tensor<HostVectorType, Properties...>& tensor_host) {
  using ValueType = typename HostVectorType::value_type;
  using DeviceVectorType = Kokkos::View<ValueType*>;
  // Every component is overwritten below, so don't initialize the memory
  Tensor<DeviceVectorType, Properties...> tensor_device{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "Tensor"),
      tensor_host.begin()->size()};
  // View that wraps the host data (DataVector) in a Kokkos::View without owning
  // it so we can deep-copy the data
  using HostUnmanaged = Kokkos::View<const ValueType*, Kokkos::HostSpace,
                                     Kokkos::MemoryUnmanaged>;
  for (size_t i = 0; i < tensor_host.size(); ++i) {
    HostUnmanaged unmanaged_host_view(tensor_host[i].data(),
                                      tensor_host[i].size());
    // Copy to device
    Kokkos::deep_copy(exec, tensor_device[i], unmanaged_host_view);
  }
  return tensor_device;
}

template <typename HostVectorType, typename... Properties>
auto copy_to_device(const Tensor<HostVectorType, Properties...>& tensor_host) {
  const Kokkos::DefaultExecutionSpace exec{};
  auto tensor_device = copy_to_device(exec, tensor_host);
  exec.fence("copy_to_device");
  return tensor_device;
}
/// @}

/// @{
/*!
 * \brief Copy a `Tensor<Kokkos::View>` to host memory.
 *
 * The copies are enqueued on the execution space instance `exec` (the default
 * instance if not given), which is fenced before returning so the host data is
 * ready to use.
 */
template <typename ExecSpace, typename DeviceVectorType, typename... Properties>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto copy_to_host(
    const ExecSpace& exec,
    const Tensor<DeviceVectorType, Properties...>& tensor_device) {
  using ValueType = typename DeviceVectorType::value_type;
  static_assert(
      std::is_same_v<ValueType, double>,
      "Only Tensor<Kokkos::View<double*>> is currently supported. Other "
      "datatypes can be added by mapping to their host representation (e.g. "
      "complex -> ComplexDataVector).");
  using HostVectorType = DataVector;
  Tensor<HostVectorType, Properties...> tensor_host{
      tensor_device.begin()->size()};
  // View that wraps the host data (DataVector) in a Kokkos::View without owning
  // it so we can deep-copy the data
  using HostUnmanaged =
      Kokkos::View<ValueType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  for (size_t i = 0; i < tensor_device.size(); ++i) {
    HostUnmanaged unmanaged_host_view(tensor_host[i].data(),
                                      tensor_host[i].size());
    // Copy to host
    Kokkos::deep_copy(exec, unmanaged_host_view, tensor_device[i]);
  }
  exec.fence("copy_to_host");
  return tensor_host;
}

template <typename DeviceVectorType, typename... Properties>
auto copy_to_host(
    const Tensor<DeviceVectorType, Properties...>& tensor_device) {
  return copy_to_host(Kokkos::DefaultExecutionSpace{}, tensor_device);
}
/// @}
#endif  // SPECTRE_KOKKOS
