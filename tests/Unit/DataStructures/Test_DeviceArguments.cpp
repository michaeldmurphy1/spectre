// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArguments.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
struct Factor : db::SimpleTag {
  using type = double;
};

using VectorTag = TestHelpers::Tags::Vector<DataVector>;

static_assert(std::is_same_v<device_argument_tag<VectorTag>,
                             ::Tags::MirrorView<VectorTag>>);
static_assert(std::is_same_v<device_argument_tag<Factor>, Factor>);
// `DeviceArguments` stores the result of `copy_to_device` under the
// `Tags::MirrorView` tag, so their types must agree.
static_assert(
    std::is_same_v<decltype(copy_to_device(
                       std::declval<const typename VectorTag::type&>())),
                   typename ::Tags::MirrorView<VectorTag>::type>);
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
}
