// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Patches/HostCopyKokkos.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ScalarTag = TestHelpers::Tags::Scalar<DataVector>;
using VectorTag = TestHelpers::Tags::Vector<DataVector>;
using tags = tmpl::list<ScalarTag, VectorTag>;

constexpr size_t number_of_points = 5;

// A unique value for every grid point, component and element, shifted by
// `offset`
double value(const size_t i, const size_t component, const size_t e,
             const double offset = 0.0) {
  return static_cast<double>(i + 10 * component + 100 * e) + offset;
}

template <typename TagsList>
BatchedVariables<TagsList> make_batch(const size_t number_of_elements,
                                      const double offset) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host{number_of_points,
                                                     number_of_elements};
  const auto& view = host.view();
  for (size_t e = 0; e < view.extent(2); ++e) {
    for (size_t c = 0; c < view.extent(1); ++c) {
      for (size_t i = 0; i < view.extent(0); ++i) {
        view(i, c, e) = value(i, c, e, offset);
      }
    }
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<TagsList> result{exec, number_of_points, number_of_elements};
  deep_copy(exec, make_not_null(&result), host);
  exec.fence();
  return result;
}

// Whether the components of `batch` are those of the components of the
// batch from `make_batch` starting at `first_component`
template <typename View>
void check_values(const View& view, const size_t number_of_elements,
                  const size_t first_component, const double offset) {
  REQUIRE(view.extent(2) == number_of_elements);
  for (size_t e = 0; e < view.extent(2); ++e) {
    for (size_t c = 0; c < view.extent(1); ++c) {
      for (size_t i = 0; i < view.extent(0); ++i) {
        CHECK(view(i, c, e) == value(i, first_component + c, e, offset));
      }
    }
  }
}

void test_host_copy() {
  const Kokkos::DefaultExecutionSpace exec{};
  {
    INFO("All variables of the source");
    evolution::patches::HostCopy<tags> host_copy{};
    CHECK_FALSE(host_copy.is_allocated());
    const auto source = make_batch<tags>(3, 0.0);
    const auto& host = host_copy.enqueue_copy(exec, source);
    exec.fence();
    CHECK(host_copy.is_allocated());
    check_values(host.view(), 3, 0, 0.0);
  }
  {
    INFO("Some variables of the source, also after resizing");
    evolution::patches::HostCopy<tmpl::list<VectorTag>> host_copy{};
    const auto source = make_batch<tags>(3, 0.0);
    const auto* const first_copy = &host_copy.enqueue_copy(exec, source);
    exec.fence();
    // The vector follows the scalar in the source
    check_values(first_copy->view(), 3, 1, 0.0);
    const auto changed = make_batch<tags>(3, 0.5);
    const auto& second_copy = host_copy.enqueue_copy(exec, changed);
    exec.fence();
    CHECK(&second_copy == first_copy);
    check_values(second_copy.view(), 3, 1, 0.5);
    const auto more_elements = make_batch<tags>(4, 0.25);
    const auto& resized = host_copy.enqueue_copy(exec, more_elements);
    exec.fence();
    check_values(resized.view(), 4, 1, 0.25);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.HostCopyKokkos",
                  "[Unit][Evolution]") {
  test_host_copy();
}
