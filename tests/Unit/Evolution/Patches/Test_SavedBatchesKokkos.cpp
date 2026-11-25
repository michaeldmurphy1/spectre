// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Patches/SavedBatchesKokkos.hpp"
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

void test_saved_batches() {
  const Kokkos::DefaultExecutionSpace exec{};
  auto all = make_batch<tags>(3, 0.0);
  auto vector = make_batch<tmpl::list<VectorTag>>(2, 0.5);
  evolution::patches::SavedBatches<BatchedVariables<tags>,
                                   BatchedVariables<tmpl::list<VectorTag>>>
      saved{};
  saved.save(exec, all, vector);
  // The saved batches are copies, not references to the same memory
  Kokkos::deep_copy(exec, all.view(), 1.0);
  Kokkos::deep_copy(exec, vector.view(), 2.0);
  exec.fence();
  saved.restore(exec, make_not_null(&all), make_not_null(&vector));
  exec.fence();
  check_values(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, all.view()), 3,
      0, 0.0);
  check_values(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vector.view()),
      2, 0, 0.5);
  // They can be restored again
  Kokkos::deep_copy(exec, all.view(), 1.0);
  saved.restore(exec, make_not_null(&all), make_not_null(&vector));
  exec.fence();
  check_values(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, all.view()), 3,
      0, 0.0);
  saved.discard();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.SavedBatchesKokkos",
                  "[Unit][Evolution]") {
  test_saved_batches();
}
