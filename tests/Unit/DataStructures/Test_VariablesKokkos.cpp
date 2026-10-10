// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesKokkos.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_copy_and_compute() {
  const size_t num_points = 5;
  using VectorTag = TestHelpers::Tags::Vector<DataVector>;
  using ScalarTag = TestHelpers::Tags::Scalar<DataVector>;

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

  // Copying to a host `Variables` of a different size resizes it
  Variables<tmpl::list<VectorTag, ScalarTag>> resized_vars_host{};
  copy_to_host(make_not_null(&resized_vars_host), vars_device);
  CHECK(resized_vars_host == vars_host);
}

// Variables without tags have nothing to copy
void test_no_tags() {
  const Variables<tmpl::list<>> vars_host{};
  const auto vars_device = copy_to_device(vars_host);
  static_assert(std::is_same_v<std::decay_t<decltype(vars_device)>,
                               Variables<tmpl::list<>>>);
  CHECK(vars_device.number_of_grid_points() == 0);
  Variables<tmpl::list<>> copied_vars_host{};
  copy_to_host(make_not_null(&copied_vars_host), vars_device);
  CHECK(copied_vars_host.number_of_grid_points() == 0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.VariablesKokkos",
                  "[DataStructures][Unit]") {
  test_copy_and_compute();
  test_no_tags();
}
