// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/BatchedVariables.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArguments.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesKokkos.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDevice.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
struct Factor : db::SimpleTag {
  using type = double;
};
struct ScaledVector : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
struct SquaredMagnitude : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Untouched : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Value : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Clamped : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// A pointwise function in the form that `apply_pointwise_on_device` expects
struct ScaleAndSquare {
  using return_tags = tmpl::list<ScaledVector, SquaredMagnitude>;
  using argument_tags = tmpl::list<Vector, Factor>;

  // Also instantiated for `DataVector`, which is only used on the host
  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <typename DataType>
  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<tnsr::I<DataType, 3>*> scaled_vector,
      const gsl::not_null<Scalar<DataType>*> squared_magnitude,
      const tnsr::I<DataType, 3>& vector, const double factor) {
    get(*squared_magnitude) = get<0>(vector) * get<0>(vector);
    for (size_t d = 0; d < 3; ++d) {
      scaled_vector->get(d) = factor * vector.get(d);
      if (d > 0) {
        get(*squared_magnitude) += vector.get(d) * vector.get(d);
      }
    }
  }
};

// Same as `ScaleAndSquare`, but with a different factor for each element
struct ScaleAndSquarePerElement {
  using return_tags = tmpl::list<ScaledVector, SquaredMagnitude>;
  using argument_tags = tmpl::list<Vector, Tags::PerElement<Factor>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<tnsr::I<double, 3>*> scaled_vector,
      const gsl::not_null<Scalar<double>*> squared_magnitude,
      const tnsr::I<double, 3>& vector, const double factor) {
    ScaleAndSquare::apply(scaled_vector, squared_magnitude, vector, factor);
  }
};

// A pointwise function that reports whether it changed anything
struct ClampToZero {
  using return_tags = tmpl::list<Clamped>;
  using argument_tags = tmpl::list<Value>;

  KOKKOS_FUNCTION static bool apply(
      const gsl::not_null<Scalar<double>*> clamped,
      const Scalar<double>& value) {
    const bool is_negative = get(value) < 0.0;
    get(*clamped) = is_negative ? 0.0 : get(value);
    return is_negative;
  }
};

tnsr::I<DataVector, 3> make_vector(const size_t num_points) {
  tnsr::I<DataVector, 3> vector{num_points};
  for (size_t d = 0; d < 3; ++d) {
    for (size_t i = 0; i < num_points; ++i) {
      vector.get(d)[i] = static_cast<double>(d + 3 * i);
    }
  }
  return vector;
}

void test_against_host() {
  TestHelpers::check_pointwise_on_device<ScaleAndSquare>(
      &ScaleAndSquare::apply<DataVector>, ScaleAndSquare::return_tags{},
      ScaleAndSquare::argument_tags{}, 10, -1.0, 1.0);
}

// Only the `ResultTags` of the `Variables` are written. Also runs the kernel
// on a non-default execution space instance (e.g. a separate CUDA stream).
void test_other_tags_and_instance() {
  const size_t num_points = 4;
  const auto vector = make_vector(num_points);
  const double factor = 2.0;

  Variables<tmpl::list<ScaledVector, Untouched, SquaredMagnitude>> results{
      num_points, -1.0};
  const auto exec = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace{}, 1)[0];
  copy_and_apply_pointwise_on_device<ScaleAndSquare>(
      exec, make_not_null(&results), ScaleAndSquare::return_tags{},
      ScaleAndSquare::argument_tags{}, vector, factor);

  CHECK(get(get<Untouched>(results)) == DataVector(num_points, -1.0));
  // All values are small integers, so the results are exact
  for (size_t i = 0; i < num_points; ++i) {
    double squared_magnitude = 0.0;
    for (size_t d = 0; d < 3; ++d) {
      CHECK(get<ScaledVector>(results).get(d)[i] == factor * vector.get(d)[i]);
      squared_magnitude += vector.get(d)[i] * vector.get(d)[i];
    }
    CHECK(get(get<SquaredMagnitude>(results))[i] == squared_magnitude);
  }
}

// Data that stays on the device: the results of the first call are the
// arguments of the second, without copying them to the host in between.
void test_device_data() {
  const size_t num_points = 4;
  const auto vector = make_vector(num_points);
  const double factor = 2.0;
  const Kokkos::DefaultExecutionSpace exec{};

  const auto first_args =
      make_device_arguments<Vector, Factor>(exec, vector, factor);
  Variables<
      tmpl::list<Tags::MirrorView<ScaledVector>, Tags::MirrorView<Untouched>,
                 Tags::MirrorView<SquaredMagnitude>>>
      first_results{num_points};
  Kokkos::deep_copy(exec, first_results.view(), -1.0);
  apply_pointwise_on_device<ScaleAndSquare>(
      exec, make_not_null(&first_results), ScaleAndSquare::return_tags{},
      ScaleAndSquare::argument_tags{}, first_args);

  const auto second_args = make_device_arguments<Vector, Factor>(
      exec, get<Tags::MirrorView<ScaledVector>>(first_results), factor);
  Variables<tmpl::list<Tags::MirrorView<ScaledVector>,
                       Tags::MirrorView<SquaredMagnitude>>>
      second_results{num_points};
  apply_pointwise_on_device<ScaleAndSquare>(
      exec, make_not_null(&second_results), ScaleAndSquare::return_tags{},
      ScaleAndSquare::argument_tags{}, second_args);

  exec.fence();
  Variables<tmpl::list<ScaledVector, Untouched, SquaredMagnitude>>
      host_first_results{num_points};
  copy_to_host(make_not_null(&host_first_results), first_results);
  Variables<tmpl::list<ScaledVector, SquaredMagnitude>> host_second_results{
      num_points};
  copy_to_host(make_not_null(&host_second_results), second_results);

  CHECK(get(get<Untouched>(host_first_results)) ==
        DataVector(num_points, -1.0));
  // All values are small integers, so the results are exact
  for (size_t i = 0; i < num_points; ++i) {
    double squared_magnitude = 0.0;
    for (size_t d = 0; d < 3; ++d) {
      CHECK(get<ScaledVector>(host_first_results).get(d)[i] ==
            factor * vector.get(d)[i]);
      CHECK(get<ScaledVector>(host_second_results).get(d)[i] ==
            square(factor) * vector.get(d)[i]);
      squared_magnitude += square(factor * vector.get(d)[i]);
    }
    CHECK(get(get<SquaredMagnitude>(host_second_results))[i] ==
          squared_magnitude);
  }
}

// A batch of elements is computed in one kernel and gives the same results as
// computing each element on its own
void test_batch() {
  const size_t num_points = 4;
  const size_t num_elements = 3;
  const std::vector<double> factors{1.0, 2.0, 3.0};
  const Kokkos::DefaultExecutionSpace exec{};

  // The vectors of all elements, set on the host
  BatchedVariables<tmpl::list<Vector>, Kokkos::HostSpace> host_vectors{
      num_points, num_elements};
  std::vector<tnsr::I<DataVector, 3>> vectors{};
  for (size_t e = 0; e < num_elements; ++e) {
    Variables<tmpl::list<Vector>> host_element{};
    set_data_ref_to_element(make_not_null(&host_element), host_vectors, e);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < num_points; ++i) {
        get<Vector>(host_element).get(d)[i] =
            static_cast<double>(d + 3 * i + 20 * e);
      }
    }
    vectors.push_back(get<Vector>(host_element));
  }
  BatchedVariables<tmpl::list<Vector>> device_vectors{num_points, num_elements};
  deep_copy(exec, make_not_null(&device_vectors), host_vectors);

  BatchedVariables<tmpl::list<ScaledVector, Untouched, SquaredMagnitude>>
      results{num_points, num_elements};
  Kokkos::deep_copy(exec, results.view(), -1.0);
  apply_pointwise_on_device<ScaleAndSquarePerElement>(
      exec, make_not_null(&results), ScaleAndSquarePerElement::return_tags{},
      ScaleAndSquarePerElement::argument_tags{},
      make_device_arguments<Vector, Tags::PerElement<Factor>>(
          exec, get<Tags::BatchView<Vector>>(device_vectors), factors));
  exec.fence();

  for (size_t e = 0; e < num_elements; ++e) {
    CAPTURE(e);
    Variables<tmpl::list<ScaledVector, Untouched, SquaredMagnitude>>
        batch_results{};
    copy_to_host(make_not_null(&batch_results), results.element(e));
    Variables<tmpl::list<ScaledVector, SquaredMagnitude>> expected{num_points};
    copy_and_apply_pointwise_on_device<ScaleAndSquare>(
        make_not_null(&expected), ScaleAndSquare::return_tags{},
        ScaleAndSquare::argument_tags{}, vectors[e], factors[e]);
    CHECK(get(get<Untouched>(batch_results)) == DataVector(num_points, -1.0));
    CHECK(get<ScaledVector>(batch_results) == get<ScaledVector>(expected));
    CHECK(get<SquaredMagnitude>(batch_results) ==
          get<SquaredMagnitude>(expected));
  }
}

// Functions that return a `bool` report whether they returned `true` at any
// grid point
void test_reduction() {
  Variables<tmpl::list<Clamped>> results{4};
  Scalar<DataVector> value{DataVector{1.0, 2.0, 3.0, 4.0}};
  CHECK_FALSE(copy_and_apply_pointwise_on_device<ClampToZero>(
      make_not_null(&results), ClampToZero::return_tags{},
      ClampToZero::argument_tags{}, value));
  CHECK(get(get<Clamped>(results)) == get(value));

  get(value)[2] = -3.0;
  CHECK(copy_and_apply_pointwise_on_device<ClampToZero>(
      make_not_null(&results), ClampToZero::return_tags{},
      ClampToZero::argument_tags{}, value));
  CHECK(get(get<Clamped>(results)) == DataVector{1.0, 2.0, 0.0, 4.0});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.ApplyPointwiseOnDevice",
                  "[DataStructures][Unit]") {
  test_against_host();
  test_other_tags_and_instance();
  test_device_data();
  test_batch();
  test_reduction();
}
