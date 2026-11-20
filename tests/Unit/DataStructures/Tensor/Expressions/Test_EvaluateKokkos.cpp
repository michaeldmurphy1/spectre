// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AtPointKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t dim = 3;
constexpr size_t num_points = 7;
constexpr size_t num_elements = 3;

struct Lapse : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Shift : db::SimpleTag {
  using type = tnsr::I<DataVector, dim>;
};
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, dim>;
};
struct Pi : db::SimpleTag {
  using type = tnsr::aa<DataVector, dim>;
};
struct Phi : db::SimpleTag {
  using type = tnsr::iaa<DataVector, dim>;
};
struct DerivMetric : db::SimpleTag {
  using type = tnsr::iaa<DataVector, dim>;
};
using input_tags =
    tmpl::list<Lapse, Shift, InverseSpatialMetric, Pi, Phi, DerivMetric>;

struct ResultAa : db::SimpleTag {
  using type = tnsr::aa<DataVector, dim>;
};
struct ResultI : db::SimpleTag {
  using type = tnsr::i<DataVector, dim>;
};
struct ResultScalar : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct ResultTimeComponents : db::SimpleTag {
  using type = tnsr::aa<DataVector, dim>;
};
struct ResultSpatialComponents : db::SimpleTag {
  using type = tnsr::aa<DataVector, dim>;
};
struct ResultIaa : db::SimpleTag {
  using type = tnsr::iaa<DataVector, dim>;
};
struct ResultiJ : db::SimpleTag {
  using type = tnsr::iJ<DataVector, dim>;
};
using result_tags =
    tmpl::list<ResultAa, ResultI, ResultScalar, ResultTimeComponents,
               ResultSpatialComponents, ResultIaa, ResultiJ>;

template <typename Tag, typename Batch>
const auto& batch_view(const Batch& batch) {
  return get<Tags::BatchView<Tag, typename Batch::memory_space>>(batch);
}

// The tensor `Tag` of element `e` of a host batch
template <typename Tag, typename Batch>
typename Tag::type element_tensor(const Batch& host_batch, const size_t e) {
  typename Tag::type result{num_points};
  const auto& batch_tensor = batch_view<Tag>(host_batch);
  for (size_t c = 0; c < result.size(); ++c) {
    for (size_t i = 0; i < num_points; ++i) {
      result[c][i] = batch_tensor[c](i, e);
    }
  }
  return result;
}

// The expressions, evaluated on the host with DataVectors for comparison
template <typename Lapse_, typename Shift_, typename InvMetric, typename Pi_,
          typename Phi_, typename DMetric>
auto host_results(const Lapse_& lapse, const Shift_& shift,
                  const InvMetric& inverse_spatial_metric, const Pi_& pi,
                  const Phi_& phi, const DMetric& d_metric) {
  tnsr::aa<DataVector, dim> result_aa{num_points};
  tenex::evaluate<ti::a, ti::b>(
      make_not_null(&result_aa),
      -lapse() * pi(ti::a, ti::b) + shift(ti::I) * phi(ti::i, ti::a, ti::b) +
          2.0 * shift(ti::I) *
              (d_metric(ti::i, ti::a, ti::b) - phi(ti::i, ti::a, ti::b)));
  tnsr::i<DataVector, dim> result_i{num_points};
  tenex::evaluate<ti::i>(
      make_not_null(&result_i),
      2.0 * pi(ti::t, ti::i) - inverse_spatial_metric(ti::J, ti::K) *
                                   phi(ti::j, ti::k, ti::i) / lapse());
  Scalar<DataVector> result_scalar{num_points};
  tenex::evaluate(
      make_not_null(&result_scalar),
      sqrt(lapse() * lapse() +
           shift(ti::I) * pi(ti::t, ti::i) * pi(ti::t, ti::j) * shift(ti::J)));
  tnsr::aa<DataVector, dim> result_time{num_points, 0.0};
  tenex::evaluate<ti::t, ti::a>(make_not_null(&result_time),
                                -pi(ti::a, ti::t) * 3.0);
  tnsr::aa<DataVector, dim> result_spatial{num_points, 0.0};
  tenex::evaluate<ti::i, ti::j>(
      make_not_null(&result_spatial),
      pi(ti::j, ti::i) + 2.0 * phi(ti::i, ti::t, ti::j));
  tnsr::iaa<DataVector, dim> result_iaa{num_points};
  tenex::evaluate<ti::i, ti::b, ti::a>(
      make_not_null(&result_iaa),
      phi(ti::i, ti::a, ti::b) - d_metric(ti::i, ti::b, ti::a));
  tnsr::iJ<DataVector, dim> result_ij{num_points};
  tenex::evaluate<ti::i, ti::J>(make_not_null(&result_ij),
                                inverse_spatial_metric(ti::J, ti::K) *
                                    phi(ti::i, ti::k, ti::t) / lapse());
  return std::tuple{result_aa,      result_i,   result_scalar, result_time,
                    result_spatial, result_iaa, result_ij};
}

template <typename Tag>
using Batch = typename Tags::BatchView<Tag>::type;

// [at_point_example]
struct Kernel {
  Batch<Lapse> lapse_batch;
  Batch<Shift> shift_batch;
  Batch<InverseSpatialMetric> inverse_spatial_metric_batch;
  Batch<Pi> pi_batch;
  Batch<Phi> phi_batch;
  Batch<DerivMetric> d_metric_batch;
  Batch<ResultAa> result_aa_batch;
  Batch<ResultI> result_i_batch;
  Batch<ResultScalar> result_scalar_batch;
  Batch<ResultTimeComponents> result_time_batch;
  Batch<ResultSpatialComponents> result_spatial_batch;
  Batch<ResultIaa> result_iaa_batch;
  Batch<ResultiJ> result_ij_batch;

  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const size_t i = k % num_points;
    const size_t e = k / num_points;
    const auto lapse = tenex::at_point(lapse_batch, i, e);
    const auto shift = tenex::at_point(shift_batch, i, e);
    const auto inverse_spatial_metric =
        tenex::at_point(inverse_spatial_metric_batch, i, e);
    const auto pi = tenex::at_point(pi_batch, i, e);
    const auto phi = tenex::at_point(phi_batch, i, e);
    const auto d_metric = tenex::at_point(d_metric_batch, i, e);
    tenex::evaluate<ti::a, ti::b>(
        tenex::at_point(result_aa_batch, i, e),
        -lapse() * pi(ti::a, ti::b) + shift(ti::I) * phi(ti::i, ti::a, ti::b) +
            2.0 * shift(ti::I) *
                (d_metric(ti::i, ti::a, ti::b) - phi(ti::i, ti::a, ti::b)));
    // [at_point_example]

    // Into a `Tensor<double>`, through `update` and the returning `evaluate`
    tnsr::i<double, dim> result_i{};
    tenex::evaluate<ti::i>(make_not_null(&result_i), 2.0 * pi(ti::t, ti::i));
    tenex::update<ti::i>(
        make_not_null(&result_i),
        result_i(ti::i) - inverse_spatial_metric(ti::J, ti::K) *
                              phi(ti::j, ti::k, ti::i) / lapse());
    const auto copy_of_result_i = tenex::evaluate<ti::i>(result_i(ti::i));
    tenex::evaluate<ti::i>(tenex::at_point(result_i_batch, i, e),
                           copy_of_result_i(ti::i));

    tenex::evaluate(
        tenex::at_point(result_scalar_batch, i, e),
        sqrt(lapse() * lapse() + shift(ti::I) * pi(ti::t, ti::i) *
                                     pi(ti::t, ti::j) * shift(ti::J)));
    // Only the time components
    tenex::evaluate<ti::t, ti::a>(tenex::at_point(result_time_batch, i, e),
                                  -pi(ti::a, ti::t) * 3.0);
    // Only the spatial components, a spatial index of a spacetime tensor
    tenex::evaluate<ti::i, ti::j>(
        tenex::at_point(result_spatial_batch, i, e),
        pi(ti::j, ti::i) + 2.0 * phi(ti::i, ti::t, ti::j));
    // The indices of the result in a different order than in the expression
    tenex::evaluate<ti::i, ti::b, ti::a>(
        tenex::at_point(result_iaa_batch, i, e),
        phi(ti::i, ti::a, ti::b) - d_metric(ti::i, ti::b, ti::a));
    tenex::evaluate<ti::i, ti::J>(tenex::at_point(result_ij_batch, i, e),
                                  inverse_spatial_metric(ti::J, ti::K) *
                                      phi(ti::i, ti::k, ti::t) / lapse());
  }
};

void test_at_point() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist{0.5, 1.5};
  const BatchedVariables<input_tags, Kokkos::HostSpace> host_inputs{
      num_points, num_elements};
  for (size_t c = 0; c < host_inputs.view().extent(1); ++c) {
    for (size_t e = 0; e < num_elements; ++e) {
      for (size_t i = 0; i < num_points; ++i) {
        host_inputs.view()(i, c, e) = dist(gen);
      }
    }
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<input_tags> inputs{exec, num_points, num_elements};
  deep_copy(exec, make_not_null(&inputs), host_inputs);
  BatchedVariables<result_tags> results{exec, num_points, num_elements};
  // Zero, because only some components of `ResultTimeComponents` and
  // `ResultSpatialComponents` are set
  Kokkos::deep_copy(exec, results.view(), 0.0);
  Kokkos::parallel_for(
      "test_at_point",
      Kokkos::RangePolicy<>(exec, 0, num_points * num_elements),
      Kernel{
          .lapse_batch = batch_view<Lapse>(inputs),
          .shift_batch = batch_view<Shift>(inputs),
          .inverse_spatial_metric_batch =
              batch_view<InverseSpatialMetric>(inputs),
          .pi_batch = batch_view<Pi>(inputs),
          .phi_batch = batch_view<Phi>(inputs),
          .d_metric_batch = batch_view<DerivMetric>(inputs),
          .result_aa_batch = batch_view<ResultAa>(results),
          .result_i_batch = batch_view<ResultI>(results),
          .result_scalar_batch = batch_view<ResultScalar>(results),
          .result_time_batch = batch_view<ResultTimeComponents>(results),
          .result_spatial_batch = batch_view<ResultSpatialComponents>(results),
          .result_iaa_batch = batch_view<ResultIaa>(results),
          .result_ij_batch = batch_view<ResultiJ>(results)});
  BatchedVariables<result_tags, Kokkos::HostSpace> host_results_batch{
      num_points, num_elements};
  deep_copy(exec, make_not_null(&host_results_batch), results);
  exec.fence();

  for (size_t e = 0; e < num_elements; ++e) {
    CAPTURE(e);
    const auto [expected_aa, expected_i, expected_scalar, expected_time,
                expected_spatial, expected_iaa, expected_ij] =
        host_results(element_tensor<Lapse>(host_inputs, e),
                     element_tensor<Shift>(host_inputs, e),
                     element_tensor<InverseSpatialMetric>(host_inputs, e),
                     element_tensor<Pi>(host_inputs, e),
                     element_tensor<Phi>(host_inputs, e),
                     element_tensor<DerivMetric>(host_inputs, e));
    CHECK_ITERABLE_APPROX(element_tensor<ResultAa>(host_results_batch, e),
                          expected_aa);
    CHECK_ITERABLE_APPROX(element_tensor<ResultI>(host_results_batch, e),
                          expected_i);
    CHECK_ITERABLE_APPROX(element_tensor<ResultScalar>(host_results_batch, e),
                          expected_scalar);
    CHECK_ITERABLE_APPROX(
        element_tensor<ResultTimeComponents>(host_results_batch, e),
        expected_time);
    CHECK_ITERABLE_APPROX(
        element_tensor<ResultSpatialComponents>(host_results_batch, e),
        expected_spatial);
    CHECK_ITERABLE_APPROX(element_tensor<ResultIaa>(host_results_batch, e),
                          expected_iaa);
    CHECK_ITERABLE_APPROX(element_tensor<ResultiJ>(host_results_batch, e),
                          expected_ij);
  }
}

#ifdef SPECTRE_DEBUG
void test_out_of_bounds() {
  const BatchedVariables<tmpl::list<Lapse>, Kokkos::HostSpace> batch{
      num_points, num_elements};
  CHECK_THROWS_WITH(tenex::at_point(batch_view<Lapse>(batch), num_points, 0),
                    Catch::Matchers::ContainsSubstring("is out of bounds"));
}
#endif  // SPECTRE_DEBUG
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateKokkos",
                  "[DataStructures][Unit]") {
  test_at_point();
#ifdef SPECTRE_DEBUG
  test_out_of_bounds();
#endif  // SPECTRE_DEBUG
}
