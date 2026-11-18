// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Tag>
using deriv = ::Tags::deriv<Tag, tmpl::size_t<3>, Frame::Inertial>;
using spacetime_metric_tag = gr::Tags::SpacetimeMetric<DataVector, 3>;
using pi_tag = gh::Tags::Pi<DataVector, 3>;
using phi_tag = gh::Tags::Phi<DataVector, 3>;
using dt_pi_tag = ::Tags::dt<pi_tag>;
using lapse_tag = gr::Tags::Lapse<DataVector>;
using shift_tag = gr::Tags::Shift<DataVector, 3>;
using inverse_tag = gr::Tags::InverseSpatialMetric<DataVector, 3>;
using primitive_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::SpecificInternalEnergy<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::LorentzFactor<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, 3>,
               hydro::Tags::MagneticField<DataVector, 3>>;
using input_tags =
    tmpl::append<tmpl::list<spacetime_metric_tag, pi_tag, phi_tag, dt_pi_tag,
                            lapse_tag, shift_tag, inverse_tag>,
                 primitive_tags>;
using cell_spacetime_tags =
    grmhd::ValenciaDivClean::subcell::cell_spacetime_tags;

constexpr size_t number_of_elements = 3;
constexpr size_t number_of_points = 20;

template <typename Tag, typename Batch>
auto batch_view(const Batch& batch) {
  return get<::Tags::BatchView<Tag>>(batch);
}

template <typename TagsList>
BatchedVariables<TagsList> to_device(
    const std::vector<Variables<TagsList>>& elements) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{number_of_points,
                                                           number_of_elements};
  for (size_t e = 0; e < number_of_elements; ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    element = elements[e];
  }
  BatchedVariables<TagsList> result{number_of_points, number_of_elements};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

template <typename TagsList>
std::vector<Variables<TagsList>> to_host(
    const BatchedVariables<TagsList>& batch) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{number_of_points,
                                                           number_of_elements};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();
  std::vector<Variables<TagsList>> result{};
  for (size_t e = 0; e < number_of_elements; ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    result.push_back(element);
  }
  return result;
}

// Random GH variables, with the lapse, shift and inverse spatial metric
// computed from the spacetime metric as `gh::TimeDerivative` does, and random
// primitives
Variables<input_tags> random_inputs(
    const gsl::not_null<std::mt19937*> generator) {
  const DataVector used_for_size(number_of_points);
  Variables<input_tags> result{number_of_points};
  auto& spacetime_metric = get<spacetime_metric_tag>(result);
  spacetime_metric = gr::spacetime_metric(
      TestHelpers::gr::random_lapse(generator, used_for_size),
      TestHelpers::gr::random_shift<3>(generator, used_for_size),
      TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size));
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  get<pi_tag>(result) = make_with_random_values<tnsr::aa<DataVector, 3>>(
      generator, make_not_null(&dist), used_for_size);
  get<phi_tag>(result) = make_with_random_values<tnsr::iaa<DataVector, 3>>(
      generator, make_not_null(&dist), used_for_size);
  get<dt_pi_tag>(result) = make_with_random_values<tnsr::aa<DataVector, 3>>(
      generator, make_not_null(&dist), used_for_size);
  const auto [det, inverse] =
      determinant_and_inverse(gr::spatial_metric(spacetime_metric));
  get<inverse_tag>(result) = inverse;
  get<shift_tag>(result) = gr::shift(spacetime_metric, inverse);
  get<lapse_tag>(result) = gr::lapse(get<shift_tag>(result), spacetime_metric);

  std::uniform_real_distribution<> positive(0.1, 1.0);
  std::uniform_real_distribution<> small(-0.2, 0.2);
  get<hydro::Tags::RestMassDensity<DataVector>>(result) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&positive), used_for_size);
  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(result) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&positive), used_for_size);
  get<hydro::Tags::Pressure<DataVector>>(result) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&positive), used_for_size);
  get(get<hydro::Tags::LorentzFactor<DataVector>>(result)) =
      1.0 + get(make_with_random_values<Scalar<DataVector>>(
                generator, make_not_null(&positive), used_for_size));
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result) =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          generator, make_not_null(&small), used_for_size);
  get<hydro::Tags::MagneticField<DataVector, 3>>(result) =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          generator, make_not_null(&dist), used_for_size);
  return result;
}

// Compare to the GH functions that compute the same quantities, which agree
// to roundoff
void test_grmhd_spacetime(const std::vector<Variables<input_tags>>& inputs) {
  const auto device_inputs = to_device(inputs);
  // The inertial coordinates must be left unchanged
  std::vector<Variables<cell_spacetime_tags>> initial_values(
      number_of_elements,
      Variables<cell_spacetime_tags>{number_of_points, -1.0});
  auto cell_spacetime_variables = to_device(initial_values);
  grmhd::GhValenciaDivClean::subcell::grmhd_spacetime_on_device(
      Kokkos::DefaultExecutionSpace{}, make_not_null(&cell_spacetime_variables),
      batch_view<spacetime_metric_tag>(device_inputs),
      batch_view<pi_tag>(device_inputs), batch_view<phi_tag>(device_inputs),
      batch_view<lapse_tag>(device_inputs),
      batch_view<shift_tag>(device_inputs),
      batch_view<inverse_tag>(device_inputs));
  const auto results = to_host(cell_spacetime_variables);

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const auto& input = inputs[e];
    const auto& result = results[e];
    const auto& spacetime_metric = get<spacetime_metric_tag>(input);
    const auto& phi = get<phi_tag>(input);
    const auto& lapse = get<lapse_tag>(input);
    const auto& shift = get<shift_tag>(input);
    const auto normal = gr::spacetime_normal_vector(lapse, shift);
    const auto inverse_spacetime_metric =
        gr::inverse_spacetime_metric(lapse, shift, get<inverse_tag>(input));
    const auto spatial_metric = gr::spatial_metric(spacetime_metric);
    const auto [det, inverse] = determinant_and_inverse(spatial_metric);

    CHECK(get<lapse_tag>(result) == lapse);
    CHECK(get<shift_tag>(result) == shift);
    CHECK(get<inverse_tag>(result) == get<inverse_tag>(input));
    CHECK(get<gr::Tags::SpatialMetric<DataVector, 3>>(result) ==
          spatial_metric);
    CHECK_ITERABLE_APPROX(
        get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(result)),
        sqrt(get(det)));
    CHECK_ITERABLE_APPROX(get<deriv<lapse_tag>>(result),
                          gh::spatial_deriv_of_lapse(lapse, normal, phi));
    // The derivative of the shift contracts the inverse spacetime metric of
    // the random metric in another order than the GH function, which can
    // differ by more than the default tolerance
    const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<deriv<shift_tag>>(result),
        gh::spatial_deriv_of_shift(lapse, inverse_spacetime_metric, normal,
                                   phi),
        custom_approx);
    const auto& extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result);
    CHECK_ITERABLE_APPROX(
        extrinsic_curvature,
        gh::extrinsic_curvature(normal, get<pi_tag>(input), phi));
    const auto& d_spatial_metric =
        get<deriv<gr::Tags::SpatialMetric<DataVector, 3>>>(result);
    for (size_t k = 0; k < 3; ++k) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          CHECK(d_spatial_metric.get(k, i, j) == phi.get(k, i + 1, j + 1));
        }
      }
    }
    CHECK(get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
              result) ==
          get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
              initial_values[e]));
  }
}

// Compare to the host, which computes the same operations in the same order
// (up to fused multiply-adds)
void test_stress_energy(const std::vector<Variables<input_tags>>& inputs) {
  auto device_inputs = to_device(inputs);
  grmhd::GhValenciaDivClean::subcell::add_stress_energy_to_dt_pi_on_device(
      Kokkos::DefaultExecutionSpace{}, batch_view<dt_pi_tag>(device_inputs),
      batch_view<hydro::Tags::RestMassDensity<DataVector>>(device_inputs),
      batch_view<hydro::Tags::SpecificInternalEnergy<DataVector>>(
          device_inputs),
      batch_view<hydro::Tags::Pressure<DataVector>>(device_inputs),
      batch_view<hydro::Tags::LorentzFactor<DataVector>>(device_inputs),
      batch_view<hydro::Tags::SpatialVelocity<DataVector, 3>>(device_inputs),
      batch_view<hydro::Tags::MagneticField<DataVector, 3>>(device_inputs),
      batch_view<spacetime_metric_tag>(device_inputs),
      batch_view<lapse_tag>(device_inputs),
      batch_view<shift_tag>(device_inputs));
  const auto results = to_host(device_inputs);

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const auto& input = inputs[e];
    const auto& spacetime_metric = get<spacetime_metric_tag>(input);
    const auto& velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(input);
    const auto& field = get<hydro::Tags::MagneticField<DataVector, 3>>(input);
    const auto& lorentz_factor =
        get<hydro::Tags::LorentzFactor<DataVector>>(input);
    // As `subcell::TimeDerivative`
    const auto spatial_metric = gr::spatial_metric(spacetime_metric);
    const auto velocity_one_form =
        tenex::evaluate<ti::i>(velocity(ti::J) * spatial_metric(ti::i, ti::j));
    const auto field_one_form =
        tenex::evaluate<ti::i>(field(ti::J) * spatial_metric(ti::i, ti::j));
    const auto field_squared =
        tenex::evaluate(field(ti::J) * field_one_form(ti::j));
    const auto field_dot_velocity =
        tenex::evaluate(velocity(ti::J) * field_one_form(ti::j));
    const auto one_over_w_squared =
        tenex::evaluate(1.0 / square(lorentz_factor()));
    tnsr::aa<DataVector, 3> stress_energy{number_of_points};
    tnsr::a<DataVector, 3> four_velocity_buffer{number_of_points};
    tnsr::a<DataVector, 3> comoving_field_buffer{number_of_points};
    grmhd::GhValenciaDivClean::trace_reversed_stress_energy(
        make_not_null(&stress_energy), make_not_null(&four_velocity_buffer),
        make_not_null(&comoving_field_buffer),
        get<hydro::Tags::RestMassDensity<DataVector>>(input), velocity_one_form,
        field_one_form, field_squared, field_dot_velocity, lorentz_factor,
        one_over_w_squared, get<hydro::Tags::Pressure<DataVector>>(input),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(input),
        spacetime_metric, get<shift_tag>(input), get<lapse_tag>(input));
    auto expected = get<dt_pi_tag>(input);
    grmhd::GhValenciaDivClean::add_stress_energy_term_to_dt_pi(
        make_not_null(&expected), stress_energy, get<lapse_tag>(input));
    // Only agrees to roundoff: the compiler fuses the multiplications and
    // additions of the contractions into FMAs differently in the vectorized
    // tensor expressions on the host (even differently for different grid
    // points) than in the kernel. Where the stress-energy term cancels the
    // random time derivative, this can exceed the default tolerance.
    const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(get<dt_pi_tag>(results[e]), expected,
                                 custom_approx);
    // The other variables are unchanged
    const auto unchanged = [&](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      if constexpr (not std::is_same_v<tag, dt_pi_tag>) {
        CHECK(get<tag>(results[e]) == get<tag>(input));
      }
    };
    tmpl::for_each<input_tags>(unchanged);
  }
}
// The spatial metric, its inverse and the square root of its determinant, as
// the compute tags of an element on the host give them
void test_fixing_metric(const std::vector<Variables<input_tags>>& inputs) {
  using metric_tags = grmhd::GhValenciaDivClean::subcell::fixing_metric_tags;
  const auto device_inputs = to_device(inputs);
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<metric_tags> metric{exec, number_of_points,
                                       number_of_elements};
  grmhd::GhValenciaDivClean::subcell::fixing_metric_on_device(
      exec, make_not_null(&metric),
      batch_view<spacetime_metric_tag>(device_inputs));
  const auto results = to_host(metric);
  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const auto spatial_metric =
        gr::spatial_metric(get<spacetime_metric_tag>(inputs[e]));
    const auto [det, inverse] = determinant_and_inverse(spatial_metric);
    // The same operations in the same order as on the host, so they agree
    // to roundoff: the device and host compilers may contract them into fused
    // multiply-adds differently (bitwise on the Serial backend with clang)
    CHECK(get<gr::Tags::SpatialMetric<DataVector, 3>>(results[e]) ==
          spatial_metric);
    const auto& device_inverse =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(results[e]);
    CHECK_ITERABLE_APPROX(device_inverse, inverse);
    const auto& device_sqrt_det =
        get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(results[e]));
    CHECK_ITERABLE_APPROX(device_sqrt_det, sqrt(get(det)));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Subcell.Coupling.Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  std::vector<Variables<input_tags>> inputs{};
  inputs.reserve(number_of_elements);
  for (size_t e = 0; e < number_of_elements; ++e) {
    inputs.push_back(random_inputs(make_not_null(&generator)));
  }
  test_grmhd_spacetime(inputs);
  test_stress_energy(inputs);
  test_fixing_metric(inputs);
}
