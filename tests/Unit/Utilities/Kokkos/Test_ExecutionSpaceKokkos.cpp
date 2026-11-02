// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <thread>

#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// Fills `values` on `exec` and copies them to the host
double sum_on(const Kokkos::DefaultExecutionSpace& exec, const double value) {
  constexpr size_t size = 100;
  const Kokkos::View<double*> values(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "values"), size);
  Kokkos::parallel_for(
      "fill", Kokkos::RangePolicy<>(exec, 0, size),
      KOKKOS_LAMBDA(const size_t i) { values(i) = value; });
  double result = 0.0;
  Kokkos::parallel_reduce(
      "sum", Kokkos::RangePolicy<>(exec, 0, size),
      KOKKOS_LAMBDA(const size_t i, double& local_sum) {
        local_sum += values(i);
      },
      result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.ExecutionSpace", "[Unit][Utilities]") {
  // The same thread always gets the same instance
  const Kokkos::DefaultExecutionSpace& instance =
      Parallel::device_execution_space();
  CHECK(&Parallel::device_execution_space() == &instance);
  CHECK(sum_on(instance, 2.0) == 200.0);

  // Other threads get their own instances, which work concurrently
  std::array<const Kokkos::DefaultExecutionSpace*, 2> thread_instances{};
  std::array<double, 2> thread_sums{};
  // Catch's assertions aren't thread-safe, so the threads only record results
  std::array<bool, 2> thread_instance_is_reused{};
  {
    std::array<std::thread, 2> threads{};
    for (size_t i = 0; i < threads.size(); ++i) {
      gsl::at(threads, i) = std::thread(
          [i, &thread_instances, &thread_sums, &thread_instance_is_reused]() {
            const auto& thread_instance = Parallel::device_execution_space();
            gsl::at(thread_instances, i) = &thread_instance;
            gsl::at(thread_instance_is_reused, i) =
                &Parallel::device_execution_space() == &thread_instance;
            gsl::at(thread_sums, i) =
                sum_on(thread_instance, static_cast<double>(i + 1));
          });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  CHECK(thread_instances[0] != &instance);
  CHECK(thread_instances[1] != &instance);
  CHECK(thread_instances[0] != thread_instances[1]);
  CHECK(thread_instance_is_reused[0]);
  CHECK(thread_instance_is_reused[1]);
  CHECK(thread_sums[0] == 100.0);
  CHECK(thread_sums[1] == 200.0);
}
