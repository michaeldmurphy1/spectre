// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/BatchedVariables.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesKokkos.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ScalarTag = TestHelpers::Tags::Scalar<DataVector>;
using VectorTag = TestHelpers::Tags::Vector<DataVector>;
using tags = tmpl::list<ScalarTag, VectorTag>;

constexpr size_t num_points = 5;
constexpr size_t num_elements = 3;

// A unique value for every grid point, tensor component and element
double value(const size_t i, const size_t component, const size_t e) {
  return static_cast<double>(i + 10 * component + 100 * e);
}

// Fill the batch on the device through the batch views of the tensors
void fill_on_device(const BatchedVariables<tags>& batch) {
  const auto& scalar = get<Tags::BatchView<ScalarTag>>(batch);
  const auto& vector = get<Tags::BatchView<VectorTag>>(batch);
  Kokkos::parallel_for(
      "fill_batch", Kokkos::RangePolicy<>(0, num_points * num_elements),
      KOKKOS_LAMBDA(const size_t k) {
        const size_t i = k % num_points;
        const size_t e = k / num_points;
        get(scalar)(i, e) = static_cast<double>(i + 100 * e);
        for (size_t d = 0; d < 3; ++d) {
          vector.get(d)(i, e) = static_cast<double>(i + 10 * (d + 1) + 100 * e);
        }
      });
  Kokkos::fence();
}

void test_layout_and_elements() {
  const BatchedVariables<tags> batch{num_points, num_elements};
  CHECK(batch.number_of_grid_points() == num_points);
  CHECK(batch.number_of_elements() == num_elements);
  fill_on_device(batch);

  for (size_t e = 0; e < num_elements; ++e) {
    CAPTURE(e);
    const auto element = batch.element(e);
    // The element shares the memory of the batch
    CHECK(element.view().data() ==
          Kokkos::subview(batch.view(), 0, 0, e).data());
    CHECK(element.number_of_grid_points() == num_points);
    Variables<tags> host_element{};
    copy_to_host(make_not_null(&host_element), element);
    // Each element has the layout of a `Variables`
    for (size_t i = 0; i < num_points; ++i) {
      CHECK(get(get<ScalarTag>(host_element))[i] == value(i, 0, e));
      for (size_t d = 0; d < 3; ++d) {
        CHECK(get<VectorTag>(host_element).get(d)[i] == value(i, d + 1, e));
      }
    }
  }
}

void test_host_batch() {
  BatchedVariables<tags, Kokkos::HostSpace> host_batch{num_points,
                                                       num_elements};
  for (size_t e = 0; e < num_elements; ++e) {
    Variables<tags> host_element{};
    set_data_ref_to_element(make_not_null(&host_element), host_batch, e);
    CHECK(host_element.number_of_grid_points() == num_points);
    // The element as a `Variables` of Kokkos views in host memory shares the
    // memory of the batch
    CHECK(host_batch.element(e).view().data() == host_element.data());
    for (size_t i = 0; i < num_points; ++i) {
      get(get<ScalarTag>(host_element))[i] = value(i, 0, e);
      for (size_t d = 0; d < 3; ++d) {
        get<VectorTag>(host_element).get(d)[i] = value(i, d + 1, e);
      }
    }
  }

  BatchedVariables<tags> device_batch{num_points, num_elements};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&device_batch), host_batch);
  exec.fence();

  BatchedVariables<tags> expected{num_points, num_elements};
  fill_on_device(expected);
  const auto device_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, device_batch.view());
  const auto expected_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected.view());
  for (size_t e = 0; e < num_elements; ++e) {
    for (size_t component = 0; component < 4; ++component) {
      for (size_t i = 0; i < num_points; ++i) {
        CHECK(device_values(i, component, e) ==
              expected_values(i, component, e));
      }
    }
  }
}

// Newly allocated data is filled with signaling NaNs, with and without an
// execution space instance
void test_nan_init() {
#ifdef SPECTRE_NAN_INIT
  const Kokkos::DefaultExecutionSpace exec{};
  const BatchedVariables<tags> with_exec{exec, num_points, num_elements};
  const BatchedVariables<tags> without_exec{num_points, num_elements};
  for (const auto* batch : {&with_exec, &without_exec}) {
    const auto values =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, batch->view());
    for (size_t e = 0; e < num_elements; ++e) {
      for (size_t component = 0; component < 4; ++component) {
        for (size_t i = 0; i < num_points; ++i) {
          CHECK(std::isnan(values(i, component, e)));
        }
      }
    }
  }
#endif  // SPECTRE_NAN_INIT
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.BatchedVariables",
                  "[DataStructures][Unit]") {
  test_layout_and_elements();
  test_host_batch();
  test_nan_init();
}
