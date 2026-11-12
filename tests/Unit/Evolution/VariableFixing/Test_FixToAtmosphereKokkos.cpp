// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.tpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Fixer = VariableFixing::FixToAtmosphere<3>;
using Vlo = Fixer::VelocityLimitingOptions;
using Klo = Fixer::KappaLimitingOptions;
using primitive_tags = hydro::grmhd_tags<DataVector>;
using metric_tags = tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>>;

template <typename TagsList>
BatchedVariables<TagsList> to_device(
    const std::vector<Variables<TagsList>>& elements) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{
      elements.front().number_of_grid_points(), elements.size()};
  for (size_t e = 0; e < elements.size(); ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    element = elements[e];
  }
  BatchedVariables<TagsList> result{elements.front().number_of_grid_points(),
                                    elements.size()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

std::vector<Variables<primitive_tags>> to_host(
    const BatchedVariables<primitive_tags>& batch) {
  BatchedVariables<primitive_tags, Kokkos::HostSpace> host_batch{
      batch.number_of_grid_points(), batch.number_of_elements()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();
  std::vector<Variables<primitive_tags>> result{};
  for (size_t e = 0; e < batch.number_of_elements(); ++e) {
    Variables<primitive_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    result.push_back(element);
  }
  return result;
}

PointwiseStatusSummary fix_on_device(
    const gsl::not_null<BatchedVariables<primitive_tags>*> primitives,
    const BatchedVariables<metric_tags>& metric,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const Fixer::DeviceView& fixer) {
  return VariableFixing::fix_to_atmosphere_on_device(
      Kokkos::DefaultExecutionSpace{}, primitives,
      get<Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(metric), eos,
      fixer);
}

// Random states around the density cutoff, the velocity limiting densities and
// the kappa limiting densities of the fixers below, so all branches are taken
void random_state(const gsl::not_null<Variables<primitive_tags>*> prims,
                  const gsl::not_null<Variables<metric_tags>*> metric,
                  const gsl::not_null<std::mt19937*> generator) {
  const size_t num_points = prims->number_of_grid_points();
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_log_density(-13.0, -8.0);
  std::uniform_real_distribution<> dist_velocity(-0.3, 0.3);
  std::uniform_real_distribution<> dist_temperature(0.0, 2.e-3);
  std::uniform_real_distribution<> dist_positive(0.1, 1.0);
  std::uniform_real_distribution<> dist_metric(0.9, 1.1);
  std::uniform_real_distribution<> dist_off_diagonal(-0.05, 0.05);
  auto& rest_mass_density =
      get(get<hydro::Tags::RestMassDensity<DataVector>>(*prims));
  rest_mass_density = make_with_random_values<DataVector>(
      generator, make_not_null(&dist_log_density), used_for_size);
  for (double& density : rest_mass_density) {
    density = std::pow(10.0, density);
  }
  // A few points far above the atmosphere
  rest_mass_density[0] = 1.e-3;
  rest_mass_density[1] = 0.5;
  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(*prims) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_positive), used_for_size);
  get<hydro::Tags::SpatialVelocity<DataVector, 3>>(*prims) =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_velocity), used_for_size);
  get<hydro::Tags::LorentzFactor<DataVector>>(*prims) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_positive), used_for_size);
  get<hydro::Tags::Pressure<DataVector>>(*prims) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_positive), used_for_size);
  get<hydro::Tags::Temperature<DataVector>>(*prims) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_temperature), used_for_size);
  get<hydro::Tags::ElectronFraction<DataVector>>(*prims) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_positive), used_for_size);
  get(get<hydro::Tags::ElectronFraction<DataVector>>(*prims)) *= 0.5;
  for (auto& component :
       get<hydro::Tags::MagneticField<DataVector, 3>>(*prims)) {
    component = 0.0;
  }
  get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(*prims)) = 0.0;
  auto& spatial_metric = get<gr::Tags::SpatialMetric<DataVector, 3>>(*metric);
  spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_off_diagonal), used_for_size);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist_metric), used_for_size);
  }
}

// The batched version fixes the variables of each element like the
// `DataVector` version
void test_random_batch(const gsl::not_null<std::mt19937*> generator,
                       const Fixer& fixer,
                       const EquationsOfState::EquationOfState<true, 3>& eos) {
  const size_t num_points = 200;
  const size_t num_elements = 3;
  std::vector<Variables<primitive_tags>> prims(
      num_elements, Variables<primitive_tags>{num_points});
  std::vector<Variables<metric_tags>> metric(
      num_elements, Variables<metric_tags>{num_points});
  for (size_t e = 0; e < num_elements; ++e) {
    random_state(make_not_null(&prims[e]), make_not_null(&metric[e]),
                 generator);
  }
  auto device_prims = to_device(prims);
  const auto status =
      fix_on_device(make_not_null(&device_prims), to_device(metric), eos,
                    fixer.device_view());
  CHECK(status.number_of_failures == 0);
  const auto fixed = to_host(device_prims);

  for (size_t e = 0; e < num_elements; ++e) {
    CAPTURE(e);
    auto expected = prims[e];
    fixer(
        make_not_null(&get<hydro::Tags::RestMassDensity<DataVector>>(expected)),
        make_not_null(
            &get<hydro::Tags::SpecificInternalEnergy<DataVector>>(expected)),
        make_not_null(
            &get<hydro::Tags::SpatialVelocity<DataVector, 3>>(expected)),
        make_not_null(&get<hydro::Tags::LorentzFactor<DataVector>>(expected)),
        make_not_null(&get<hydro::Tags::Pressure<DataVector>>(expected)),
        make_not_null(&get<hydro::Tags::Temperature<DataVector>>(expected)),
        get<hydro::Tags::ElectronFraction<DataVector>>(expected),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(metric[e]), eos);
    // The device code may be compiled differently (e.g. with fused
    // multiply-adds)
    const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<hydro::Tags::RestMassDensity<DataVector>>(fixed[e]),
        get<hydro::Tags::RestMassDensity<DataVector>>(expected), custom_approx);
    const auto& fixed_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(fixed[e]);
    const auto& expected_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(expected);
    CHECK_ITERABLE_CUSTOM_APPROX(fixed_velocity, expected_velocity,
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<hydro::Tags::LorentzFactor<DataVector>>(fixed[e]),
        get<hydro::Tags::LorentzFactor<DataVector>>(expected), custom_approx);
    // The temperature of the kappa limiting is found with a root finder to a
    // relative tolerance of 1e-13
    const Approx root_finder_approx =
        Approx::custom().epsilon(1.e-10).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<hydro::Tags::Temperature<DataVector>>(fixed[e]),
        get<hydro::Tags::Temperature<DataVector>>(expected),
        root_finder_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(fixed[e]),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(expected),
        root_finder_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<hydro::Tags::Pressure<DataVector>>(fixed[e]),
        get<hydro::Tags::Pressure<DataVector>>(expected), root_finder_approx);
    // The other variables are unchanged
    CHECK(get<hydro::Tags::ElectronFraction<DataVector>>(fixed[e]) ==
          get<hydro::Tags::ElectronFraction<DataVector>>(prims[e]));
    CHECK(get<hydro::Tags::MagneticField<DataVector, 3>>(fixed[e]) ==
          get<hydro::Tags::MagneticField<DataVector, 3>>(prims[e]));
  }
}

// The options don't allow a negative minimum temperature, but with one the
// pressure at the minimum temperature is negative. Then the maximum pressure of
// the kappa limiting, which is a factor larger than one times that, is smaller
// and doesn't bound the root, so the kappa limiting fails. The failure is
// reported for its element and grid point, the variables there are left
// unfixed, and the error shows them.
void test_kappa_limiting_failure() {
  auto fixer_view = Fixer{
      1.e-12, 3.e-12, std::nullopt,
      Fixer::KappaLimitingOptions{
          3.e-12, 1.e-3, 3.e-11, 0.01, std::nullopt,
          false}}.device_view();
  fixer_view.kappa_limiting.min_temperature_from_eos = false;
  fixer_view.kappa_limiting.min_temperature = -1.0;
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>> eos{
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}};

  // Far above the kappa limiting densities, so not limited
  Variables<primitive_tags> prims{2, 0.0};
  get(get<hydro::Tags::RestMassDensity<DataVector>>(prims)) = 0.5;
  get(get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prims)) = 1.0;
  get(get<hydro::Tags::LorentzFactor<DataVector>>(prims)) = 1.0;
  get(get<hydro::Tags::Pressure<DataVector>>(prims)) = 1.0;
  get(get<hydro::Tags::Temperature<DataVector>>(prims)) = 1.0;
  get(get<hydro::Tags::ElectronFraction<DataVector>>(prims)) = 0.1;
  auto failing_prims = prims;
  get(get<hydro::Tags::RestMassDensity<DataVector>>(failing_prims))[1] = 1.e-11;
  Variables<metric_tags> metric{2, 0.0};
  for (size_t d = 0; d < 3; ++d) {
    get<gr::Tags::SpatialMetric<DataVector, 3>>(metric).get(d, d) = 1.0;
  }

  auto device_prims = to_device(std::vector{prims, failing_prims});
  const auto status =
      fix_on_device(make_not_null(&device_prims),
                    to_device(std::vector{metric, metric}), eos, fixer_view);
  CHECK(status.number_of_failures == 1);
  CHECK(status.first_failed_element == 1);
  CHECK(status.first_failed_grid_point == 1);
  CHECK(to_host(device_prims)[1] == failing_prims);
  CHECK_THROWS_WITH(
      VariableFixing::error_fixing_to_atmosphere_on_device(
          Kokkos::DefaultExecutionSpace{}, status, device_prims,
          "the element [B0,(L0I0)]"),
      Catch::Matchers::ContainsSubstring(
          "The root find of the kappa limiting in FixToAtmosphere failed on "
          "the device at 1 of 4 grid points. At the first one, grid point 1 "
          "of the element [B0,(L0I0)], before fixing:") and
          Catch::Matchers::ContainsSubstring(
              "temperature = T()=1.000000000000000000e+00"));

  // With a deferred status, the failure is recorded on the device, with the
  // variables before fixing
  const Kokkos::DefaultExecutionSpace exec{};
  auto deferred_prims = to_device(std::vector{prims, failing_prims});
  const auto device_metric = to_device(std::vector{metric, metric});
  DeferredPointwiseStatus<VariableFixing::fix_to_atmosphere_snapshot_tags>
      deferred_status{exec};
  deferred_status.reset(exec);
  VariableFixing::fix_to_atmosphere_on_device(
      exec, make_not_null(&deferred_prims), make_not_null(&deferred_status),
      get<Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
          device_metric),
      eos, fixer_view);
  deferred_status.copy_to_host(exec);
  exec.fence();
  REQUIRE(deferred_status.failed());
  // There is only one failure, so it is the first
  const PointwiseStatusSummary deferred_summary = deferred_status.summary();
  CHECK(deferred_summary.number_of_failures == 1);
  CHECK(deferred_summary.first_failed_element == 1);
  CHECK(deferred_summary.first_failed_grid_point == 1);
  CHECK(to_host(deferred_prims)[1] == failing_prims);
  CHECK(get(get<Tags::AtIndex<hydro::Tags::RestMassDensity<DataVector>>>(
            deferred_status.snapshot())) == 1.e-11);
  CHECK_THROWS_WITH(
      VariableFixing::error_fixing_to_atmosphere_on_device(
          deferred_status, 4, "the element [B0,(L0I0)]"),
      Catch::Matchers::ContainsSubstring(
          "The root find of the kappa limiting in FixToAtmosphere failed on "
          "the device at 1 of 4 grid points. At one of them, grid point 1 "
          "of the element [B0,(L0I0)], before fixing:") and
          Catch::Matchers::ContainsSubstring(
              "temperature = T()=1.000000000000000000e+00"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.FixToAtmosphere.Kokkos",
                  "[VariableFixing][Unit]") {
  MAKE_GENERATOR(generator);
  const Fixer no_limiting{1.e-12, 3.e-12, std::nullopt, std::nullopt};
  const Fixer limiting{1.e-12, 3.e-12, Vlo{0.0, 1.e-4, 3.e-12, 1.e-11},
                       Klo{3.e-12, 1.e-3, 3.e-11, 0.01, std::nullopt, false}};
  const Fixer limiting_everywhere{
      1.e-12, 3.e-12, Vlo{1.e-5, 1.e-4, 3.e-12, 1.e-11},
      Klo{3.e-12, 1.e-3, 3.e-11, 0.01, 1.e-3, true}};
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
      ideal_fluid{EquationsOfState::IdealFluid<true>{5.0 / 3.0}};
  const EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>
      polytrope{EquationsOfState::PolytropicFluid<true>{100.0, 2.0}};
  for (const auto* const fixer :
       {&no_limiting, &limiting, &limiting_everywhere}) {
    test_random_batch(make_not_null(&generator), *fixer, ideal_fluid);
    test_random_batch(make_not_null(&generator), *fixer, polytrope);
  }
  test_kappa_limiting_failure();
}
