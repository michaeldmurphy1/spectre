// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace Parallel {
namespace ExecutionSpace_detail {
// The instances of all host threads.  They are destroyed by a finalize hook
// before Kokkos is finalized, and `generation` is incremented so that threads
// create new instances if Kokkos is initialized again, e.g., when restarting
// from a checkpoint.
struct Registry {
  std::mutex mutex{};
  std::vector<std::unique_ptr<Kokkos::DefaultExecutionSpace>> instances{};
  std::atomic<size_t> generation{1};
  bool finalize_hook_registered = false;
};

inline Registry& registry() {
  static Registry result{};
  return result;
}

struct ThreadInstance {
  size_t generation = 0;
  const Kokkos::DefaultExecutionSpace* instance = nullptr;
};

inline thread_local ThreadInstance thread_instance{};

inline const Kokkos::DefaultExecutionSpace& create_thread_instance() {
  Registry& registry = ExecutionSpace_detail::registry();
  const std::lock_guard lock(registry.mutex);
  if (not Kokkos::is_initialized()) {
    ERROR(
        "Kokkos must be initialized before getting a device execution space.");
  }
  if (not registry.finalize_hook_registered) {
    Kokkos::push_finalize_hook([]() {
      Registry& local_registry = ExecutionSpace_detail::registry();
      const std::lock_guard finalize_lock(local_registry.mutex);
      local_registry.instances.clear();
      ++local_registry.generation;
      local_registry.finalize_hook_registered = false;
    });
    registry.finalize_hook_registered = true;
  }
  auto [instance] =
      Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace{}, 1);
  registry.instances.push_back(
      std::make_unique<Kokkos::DefaultExecutionSpace>(std::move(instance)));
  thread_instance = {registry.generation.load(),
                     registry.instances.back().get()};
  return *thread_instance.instance;
}
}  // namespace ExecutionSpace_detail

/*!
 * \brief The Kokkos execution space instance of the calling host thread, i.e.,
 * of the calling Charm++ PE.
 *
 * Work enqueued on one instance runs in order, and work on different instances
 * can run concurrently.  Each PE gets its own instance (e.g., its own CUDA
 * stream) so that the work of different PEs can overlap on the device, and so
 * that a `fence()` on the instance only waits for the work of the calling PE.
 * The default instance, `Kokkos::DefaultExecutionSpace{}`, is shared by all
 * PEs.
 *
 * Actions and other entry points into device code get the instance here and
 * pass it to the device functions, which take the instance as their first
 * argument.  Device functions should not call this themselves, so that a
 * caller that manages its own instances (e.g., a driver for a batch of
 * elements) can pass those instead.
 *
 * The instance is created on first use and destroyed when Kokkos is
 * finalized.
 *
 * \warning Work enqueued on the instance of a PE must be finished before the
 * data it uses moves to another PE, e.g., when an element migrates.
 */
inline const Kokkos::DefaultExecutionSpace& device_execution_space() {
  const ExecutionSpace_detail::ThreadInstance& thread_instance =
      ExecutionSpace_detail::thread_instance;
  if (thread_instance.instance == nullptr or
      thread_instance.generation !=
          ExecutionSpace_detail::registry().generation.load()) {
    return ExecutionSpace_detail::create_thread_instance();
  }
  return *thread_instance.instance;
}
}  // namespace Parallel
#endif  // SPECTRE_KOKKOS
