// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/SpacetimeOnFacesKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using lapse_tag = gr::Tags::Lapse<DataVector>;
using shift_tag = gr::Tags::Shift<DataVector, 3>;
using spatial_metric_tag = gr::Tags::SpatialMetric<DataVector, 3>;
using sqrt_det_tag = gr::Tags::SqrtDetSpatialMetric<DataVector>;
using inverse_tag = gr::Tags::InverseSpatialMetric<DataVector, 3>;
using spacetime_metric_tag = gr::Tags::SpacetimeMetric<DataVector, 3>;
// Like the face variables, the batch holds other tags too
using face_tags =
    tmpl::list<lapse_tag, ::Tags::TempScalar<0>, shift_tag, spatial_metric_tag,
               sqrt_det_tag, inverse_tag, spacetime_metric_tag>;

constexpr size_t number_of_elements = 3;
constexpr size_t number_of_points = 20;

template <typename Tag, typename Batch>
const auto& batch_view(const Batch& batch) {
  return get<::Tags::BatchView<Tag>>(batch);
}

void test(const gsl::not_null<std::mt19937*> generator) {
  const DataVector used_for_size(number_of_points);
  std::vector<tnsr::aa<DataVector, 3>> spacetime_metrics{};
  BatchedVariables<face_tags, Kokkos::HostSpace> host_batch{number_of_points,
                                                            number_of_elements};
  // The other tags must be left unchanged
  const Variables<face_tags> initial_values{number_of_points, -1.0};
  for (size_t e = 0; e < number_of_elements; ++e) {
    spacetime_metrics.push_back(gr::spacetime_metric(
        TestHelpers::gr::random_lapse(generator, used_for_size),
        TestHelpers::gr::random_shift<3>(generator, used_for_size),
        TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size)));
    Variables<face_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    // Copy (not move) into the batch
    element = initial_values;
    get<spacetime_metric_tag>(element) = spacetime_metrics.back();
  }
  BatchedVariables<face_tags> batch{number_of_points, number_of_elements};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&batch), host_batch);
  grmhd::GhValenciaDivClean::fd::spacetime_on_faces_on_device(
      exec, batch_view<lapse_tag>(batch), batch_view<shift_tag>(batch),
      batch_view<spatial_metric_tag>(batch), batch_view<sqrt_det_tag>(batch),
      batch_view<inverse_tag>(batch), batch_view<spacetime_metric_tag>(batch));
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    Variables<face_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    const auto& spacetime_metric = spacetime_metrics[e];
    CHECK(get<spacetime_metric_tag>(element) == spacetime_metric);
    CHECK(get(get<::Tags::TempScalar<0>>(element)) ==
          DataVector(number_of_points, -1.0));
    // The host computes the same operations in the same order, so the results
    // agree exactly on the CPU (but not where the compiler fuses operations)
    const auto spatial_metric = gr::spatial_metric(spacetime_metric);
    const auto [det, inverse] = determinant_and_inverse(spatial_metric);
    const auto shift = gr::shift(spacetime_metric, inverse);
    const auto lapse = gr::lapse(shift, spacetime_metric);
    CHECK_ITERABLE_APPROX(get<spatial_metric_tag>(element), spatial_metric);
    CHECK_ITERABLE_APPROX(get<inverse_tag>(element), inverse);
    CHECK_ITERABLE_APPROX(get(get<sqrt_det_tag>(element)), sqrt(get(det)));
    CHECK_ITERABLE_APPROX(get<shift_tag>(element), shift);
    CHECK_ITERABLE_APPROX(get<lapse_tag>(element), lapse);
    // The 3+1 variables give back the spacetime metric
    CHECK_ITERABLE_APPROX(
        gr::spacetime_metric(get<lapse_tag>(element), get<shift_tag>(element),
                             get<spatial_metric_tag>(element)),
        spacetime_metric);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.SpacetimeOnFaces."
    "Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
}
