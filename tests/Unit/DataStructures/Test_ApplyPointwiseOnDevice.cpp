// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDevice.hpp"
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

// A pointwise function in the form that `apply_pointwise_on_device` expects
struct ScaleAndSquare {
  using return_tags = tmpl::list<ScaledVector, SquaredMagnitude>;
  using argument_tags = tmpl::list<Vector, Factor>;

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

void test_against_host() {
  TestHelpers::check_pointwise_on_device<ScaleAndSquare>(
      &ScaleAndSquare::apply<DataVector>, ScaleAndSquare::return_tags{},
      ScaleAndSquare::argument_tags{}, 10, -1.0, 1.0);
}

// Only the `ResultTags` of the `Variables` are written. Also runs the kernel
// on a non-default execution space instance (e.g. a separate CUDA stream).
void test_other_tags_and_instance() {
  const size_t num_points = 4;
  tnsr::I<DataVector, 3> vector{num_points};
  for (size_t d = 0; d < 3; ++d) {
    for (size_t i = 0; i < num_points; ++i) {
      vector.get(d)[i] = static_cast<double>(d + 3 * i);
    }
  }
  const double factor = 2.0;

  Variables<tmpl::list<ScaledVector, Untouched, SquaredMagnitude>> results{
      num_points, -1.0};
  const auto exec = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace{}, 1)[0];
  apply_pointwise_on_device<ScaleAndSquare>(
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
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.ApplyPointwiseOnDevice",
                  "[DataStructures][Unit]") {
  test_against_host();
  test_other_tags_and_instance();
}
