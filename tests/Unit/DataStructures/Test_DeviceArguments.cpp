// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/BatchedVariables.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArguments.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorKokkos.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
struct Factor : db::SimpleTag {
  using type = double;
};

using VectorTag = TestHelpers::Tags::Vector<DataVector>;

static_assert(std::is_same_v<device_argument_tag<VectorTag>,
                             ::Tags::BatchView<VectorTag>>);
static_assert(std::is_same_v<device_argument_tag<Factor>, Factor>);
static_assert(std::is_same_v<device_argument_tag<Tags::PerElement<Factor>>,
                             DeviceArguments_detail::PerElementView<Factor>>);
// `to_device_argument` accepts a tensor in device memory as the
// `Tags::MirrorView` tensor, so it must be the type `copy_to_device` returns.
static_assert(
    std::is_same_v<decltype(copy_to_device(
                       std::declval<const typename VectorTag::type&>())),
                   typename ::Tags::MirrorView<VectorTag>::type>);

// Arguments for a batch of elements: a batch of tensors that is already on the
// device and a factor for each element
void test_batch() {
  const size_t num_points = 4;
  const size_t num_elements = 3;
  const BatchedVariables<tmpl::list<VectorTag>> vectors{num_points,
                                                        num_elements};
  const auto& vector_batch = get<::Tags::BatchView<VectorTag>>(vectors);
  Kokkos::parallel_for(
      "fill", num_points * num_elements, KOKKOS_LAMBDA(const size_t k) {
        for (size_t d = 0; d < 3; ++d) {
          vector_batch.get(d)(k % num_points, k / num_points) =
              static_cast<double>(d + 10 * k);
        }
      });
  const std::vector<double> factors{1.0, 2.0, 3.0};
  const auto device_args =
      make_device_arguments<VectorTag, ::Tags::PerElement<Factor>>(vector_batch,
                                                                   factors);

  BatchedVariables<tmpl::list<VectorTag>> results{num_points, num_elements};
  const auto& result_batch = get<::Tags::BatchView<VectorTag>>(results);
  Kokkos::parallel_for(
      "compute_batch", num_points * num_elements,
      KOKKOS_LAMBDA(const size_t k) {
        const size_t i = k % num_points;
        const size_t e = k / num_points;
        const tnsr::I<double, 3> vector_i =
            device_argument_at_index<VectorTag>(device_args, i, e);
        const double factor_i =
            device_argument_at_index<::Tags::PerElement<Factor>>(device_args, i,
                                                                 e);
        tnsr::I<double, 3> result_i{};
        for (size_t d = 0; d < 3; ++d) {
          result_i.get(d) = factor_i * vector_i.get(d);
        }
        set_at_index(make_not_null(&result_batch), result_i, i, e);
      });
  const auto host_results =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results.view());
  for (size_t e = 0; e < num_elements; ++e) {
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < num_points; ++i) {
        CHECK(host_results(i, d, e) ==
              factors[e] * static_cast<double>(d + 10 * (i + num_points * e)));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DeviceArguments",
                  "[DataStructures][Unit]") {
  const size_t num_points = 5;
  tnsr::I<DataVector, 3> vector_host{num_points};
  for (size_t d = 0; d < 3; ++d) {
    for (size_t i = 0; i < num_points; ++i) {
      vector_host.get(d)[i] = static_cast<double>(d + i);
    }
  }
  const double factor = 2.0;

  const auto device_args =
      make_device_arguments<VectorTag, Factor>(vector_host, factor);
  tnsr::I<Kokkos::View<double*>, 3> result{"result", num_points};
  Kokkos::parallel_for(
      "compute", num_points, KOKKOS_LAMBDA(const size_t i) {
        const tnsr::I<double, 3> vector_i =
            device_argument_at_index<VectorTag>(device_args, i);
        const double factor_i =
            device_argument_at_index<Factor>(device_args, i);
        tnsr::I<double, 3> result_i{};
        for (size_t d = 0; d < 3; ++d) {
          result_i.get(d) = factor_i * vector_i.get(d);
        }
        set_at_index(make_not_null(&result), result_i, i);
      });

  const auto result_host = copy_to_host(result);
  for (size_t d = 0; d < 3; ++d) {
    for (size_t i = 0; i < num_points; ++i) {
      CHECK(result_host.get(d)[i] == factor * static_cast<double>(d + i));
    }
  }

  // Same on a non-default execution space instance (e.g. a separate CUDA
  // stream). Nothing is fenced between the copies and the kernel, so this
  // relies on all of them being ordered on `exec`.
  const auto exec = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace{}, 1)[0];
  const auto device_args_exec =
      make_device_arguments<VectorTag, Factor>(exec, vector_host, factor);
  tnsr::I<Kokkos::View<double*>, 3> result_exec{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "result"),
      num_points};
  Kokkos::parallel_for(
      "compute_on_instance",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                          Kokkos::IndexType<size_t>>(exec, 0, num_points),
      KOKKOS_LAMBDA(const size_t i) {
        const tnsr::I<double, 3> vector_i =
            device_argument_at_index<VectorTag>(device_args_exec, i);
        const double factor_i =
            device_argument_at_index<Factor>(device_args_exec, i);
        tnsr::I<double, 3> result_i{};
        for (size_t d = 0; d < 3; ++d) {
          result_i.get(d) = factor_i * vector_i.get(d);
        }
        set_at_index(make_not_null(&result_exec), result_i, i);
      });
  const auto result_exec_host = copy_to_host(exec, result_exec);
  CHECK(result_exec_host == result_host);

  test_batch();
}
