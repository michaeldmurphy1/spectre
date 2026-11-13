// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/*!
 * \brief Parameters of a kernel that change between calls (e.g. in every
 * substep), passed to the kernel in device memory rather than as arguments.
 *
 * \details The parameters `T` are set on the host with `set()`, which writes
 * them into a pinned host buffer and enqueues copying them into a device buffer
 * that is allocated once. Kernels capture `device()`, whose address doesn't
 * change, and read the parameters from it, so their arguments are the same in
 * every call. This keeps the kernels of a step independent of the values of
 * the step, so the step could be recorded once and replayed (e.g. as a CUDA
 * graph), and the copy doesn't wait for the device.
 *
 * Because the copy is only enqueued, the host buffer must not change until
 * the copy is done: between two calls of `set()` the caller must have waited
 * for the device (e.g. once per substep). Use separate `DeviceParameters` for
 * parameters that are set several times between two waits.
 *
 * `T` must be trivially copyable. Device pointers in `T` are copied as
 * values, e.g. to point a kernel at data that changes between calls.
 */
template <typename T, typename MemorySpace =
                          typename Kokkos::DefaultExecutionSpace::memory_space>
class DeviceParameters {
  static_assert(std::is_trivially_copyable_v<T>,
                "Device parameters are copied byte by byte.");

 public:
  DeviceParameters() = default;
  explicit DeviceParameters(std::string label) : label_(std::move(label)) {}

  /// Write the `parameters` to the host buffer and enqueue copying them to
  /// the device in `exec`. Allocates the buffers in the first call.
  template <typename ExecutionSpace>
  void set(const ExecutionSpace& exec, const T& parameters) {
    if (not device_.is_allocated()) {
      device_ = Kokkos::View<T*, MemorySpace>(
          Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label_), 1);
      host_ = Kokkos::View<T*, Kokkos::SharedHostPinnedSpace>(
          Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                             label_ + " on the host"),
          1);
    }
    host_(0) = parameters;
    Kokkos::deep_copy(exec, device_, host_);
  }

  /// The parameters on the device, for kernels to capture. The address
  /// doesn't change after the first `set()`.
  Kokkos::View<const T*, MemorySpace> device() const { return device_; }

 private:
  std::string label_{"DeviceParameters"};
  Kokkos::View<T*, MemorySpace> device_{};
  Kokkos::View<T*, Kokkos::SharedHostPinnedSpace> host_{};
};
#endif  // SPECTRE_KOKKOS
