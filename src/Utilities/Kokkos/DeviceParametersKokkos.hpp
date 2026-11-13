// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"

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
 * parameters that are set several times between two waits. In debug builds
 * `set()` checks this with a flag that a kernel sets after the copy. The check
 * can only fail on backends that run asynchronously (e.g. GPUs).
 *
 * `DeviceParameters` are move-only, since copies would share the buffers.
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
  DeviceParameters(const DeviceParameters&) = delete;
  DeviceParameters& operator=(const DeviceParameters&) = delete;
  DeviceParameters(DeviceParameters&&) = default;
  DeviceParameters& operator=(DeviceParameters&&) = default;
  ~DeviceParameters() = default;

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
#ifdef SPECTRE_DEBUG
    if (copied_.is_allocated()) {
      ASSERT(copied_(0) == 1,
             "The DeviceParameters '"
                 << label_
                 << "' were set before the device finished copying the "
                    "parameters of the previous set(). Wait for the device "
                    "between two calls of set().");
    } else {
      copied_ = Kokkos::View<int*, Kokkos::SharedHostPinnedSpace>(
          Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                             label_ + " copied"),
          1);
    }
    copied_(0) = 0;
#endif  // SPECTRE_DEBUG
    host_(0) = parameters;
    Kokkos::deep_copy(exec, device_, host_);
#ifdef SPECTRE_DEBUG
    const auto copied = copied_;
    Kokkos::parallel_for(
        "DeviceParameters::copied",
        Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 1),
        KOKKOS_LAMBDA(const int /*i*/) { copied(0) = 1; });
#endif  // SPECTRE_DEBUG
  }

  /// The parameters on the device, for kernels to capture. The address
  /// doesn't change after the first `set()`.
  Kokkos::View<const T*, MemorySpace> device() const { return device_; }

  /// For parameters that are a `std::array`, its elements on the device, for
  /// kernels that take a view of them. The address doesn't change after the
  /// first `set()`.
  template <typename U = T>
    requires tt::is_std_array_v<U>
  Kokkos::View<const typename U::value_type*, MemorySpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  device_array() const {
    return Kokkos::View<const typename U::value_type*, MemorySpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        reinterpret_cast<const typename U::value_type*>(device_.data()),
        std::tuple_size_v<U>);
  }

 private:
  std::string label_{"DeviceParameters"};
  Kokkos::View<T*, MemorySpace> device_{};
  Kokkos::View<T*, Kokkos::SharedHostPinnedSpace> host_{};
#ifdef SPECTRE_DEBUG
  // Set by a kernel after the copy of the last `set()`
  Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> copied_{};
#endif  // SPECTRE_DEBUG
};
#endif  // SPECTRE_KOKKOS
