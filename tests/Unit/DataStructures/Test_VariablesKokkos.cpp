// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/VariablesKokkos.hpp"
#include "Helpers/DataStructures/TestTags.hpp"

#ifdef SPECTRE_KOKKOS

SPECTRE_TEST_CASE("Unit.DataStructures.VariablesKokkos",
                  "[DataStructures][Unit]") {
  const size_t num_points = 5;
  using VectorTag = TestHelpers::Tags::Vector<DataVector>;
  using ScalarTag = TestHelpers::Tags::Scalar<DataVector>;
  using DeviceScalarTag = Tags::MirrorView<ScalarTag>;
  using HostMirrorScalarTag =
      Tags::MirrorView<DeviceScalarTag, Kokkos::HostSpace>;
  static_assert(
      std::is_same_v<DeviceScalarTag::type, Scalar<Kokkos::View<double*>>>);
  static_assert(std::is_same_v<HostMirrorScalarTag::type::type::memory_space,
                               Kokkos::HostSpace>);

  // Fill variables on host and copy to device
  Variables<tmpl::list<VectorTag, ScalarTag>> vars_host{num_points, 1.0};
  auto vars_device = copy_to_device(vars_host);
  CHECK(vars_device.number_of_grid_points() == num_points);
  CHECK(vars_device.size() == num_points * 4);

  // Compute on device
  Kokkos::parallel_for(
      "compute", num_points, KOKKOS_LAMBDA(const size_t i) {
        auto& [vec, scal] = vars_device;
        for (size_t d = 0; d < 3; ++d) {
          vec.get(d)[i] += static_cast<double>(d + i);
        }
        get(scal)[i] += static_cast<double>(i);
        // Test make_at_index and set_at_index
        auto vars_at_index = make_at_index(vars_device, i);
        auto& [vec_at_index, scal_at_index] = vars_at_index;
        get(scal_at_index) += 2.0;
        set_at_index(make_not_null(&vars_device), vars_at_index, i);
      });

  // Copy back to host and check
  copy_to_host(make_not_null(&vars_host), vars_device);
  for (size_t i = 0; i < num_points; ++i) {
    for (size_t d = 0; d < 3; ++d) {
      CHECK(get<VectorTag>(vars_host).get(d)[i] ==
            1. + static_cast<double>(d + i));
    }
    CHECK(get(get<ScalarTag>(vars_host))[i] == 3. + static_cast<double>(i));
  }

  // Zero tags
  const Variables<tmpl::list<>> empty_host{};
  Variables<tmpl::list<>> empty_result{};
  copy_to_host(make_not_null(&empty_result), copy_to_device(empty_host));
  CHECK(empty_result.size() == 0);

  // The overloads that take an execution space instance
  const Kokkos::DefaultExecutionSpace exec{};
  auto vars_device_exec = copy_to_device(exec, vars_host);
  Kokkos::parallel_for(
      "compute_exec",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                          Kokkos::IndexType<size_t>>(exec, 0, num_points),
      KOKKOS_LAMBDA(const size_t i) {
        get(get<DeviceScalarTag>(vars_device_exec))[i] *= 2.0;
      });
  Variables<tmpl::list<VectorTag, ScalarTag>> vars_host_exec{};
  copy_to_host(exec, make_not_null(&vars_host_exec), vars_device_exec);
  CHECK(get<VectorTag>(vars_host_exec) == get<VectorTag>(vars_host));
  CHECK(get(get<ScalarTag>(vars_host_exec)) ==
        2.0 * get(get<ScalarTag>(vars_host)));
  auto reused_device = allocate_on_device(exec, vars_host_exec);
  copy_to_device(exec, make_not_null(&reused_device), vars_host);
  copy_to_host(exec, make_not_null(&vars_host_exec), reused_device);
  CHECK(vars_host_exec == vars_host);

#ifdef SPECTRE_NAN_INIT
  // Newly allocated memory is filled with signaling NaNs
  for (const bool with_exec : {false, true}) {
    Variables<tmpl::list<::Tags::MirrorView<ScalarTag>>> uninitialized{};
    if (with_exec) {
      uninitialized.initialize(exec, num_points);
    } else {
      uninitialized.initialize(num_points);
    }
    Variables<tmpl::list<ScalarTag>> uninitialized_host{};
    copy_to_host(exec, make_not_null(&uninitialized_host), uninitialized);
    for (size_t i = 0; i < num_points; ++i) {
      CHECK(std::isnan(get(get<ScalarTag>(uninitialized_host))[i]));
    }
  }
#endif  // SPECTRE_NAN_INIT
}

#endif  // SPECTRE_KOKKOS
