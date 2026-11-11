// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
// The definitions, so that the device kernel inlines them
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.tpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t dim = 3;
using Device = gh::TimeDerivativeHarmonicGauge<dim>;
using Host = gh::TimeDerivative<gh::Solutions::all_solutions<dim>, dim>;

// Random arguments with a spacetime metric close to Minkowski, so that the
// lapse is real
Variables<Device::argument_tags> make_arguments(
    const gsl::not_null<std::mt19937*> generator, const size_t num_points) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  Variables<Device::argument_tags> args{num_points};
  fill_with_random_values(make_not_null(&args), generator,
                          make_not_null(&distribution));
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, dim>>(args);
  for (size_t a = 0; a < dim + 1; ++a) {
    for (size_t b = a; b < dim + 1; ++b) {
      spacetime_metric.get(a, b) *= 0.1;
    }
    spacetime_metric.get(a, a) += a == 0 ? -1.0 : 1.0;
  }
  for (auto* gamma : {&get(get<gh::Tags::ConstraintGamma0>(args)),
                      &get(get<gh::Tags::ConstraintGamma1>(args)),
                      &get(get<gh::Tags::ConstraintGamma2>(args))}) {
    *gamma = abs(*gamma);
  }
  return args;
}

template <typename... ArgsTags>
Variables<Device::return_tags> apply_on_host(
    const Variables<Device::argument_tags>& args,
    tmpl::list<ArgsTags...> /*meta*/) {
  Variables<Device::return_tags> result{args.number_of_grid_points()};
  Device::apply(make_not_null(&get<tmpl::at_c<Device::return_tags, 0>>(result)),
                make_not_null(&get<tmpl::at_c<Device::return_tags, 1>>(result)),
                make_not_null(&get<tmpl::at_c<Device::return_tags, 2>>(result)),
                get<ArgsTags>(args)...);
  return result;
}

// `TimeDerivativeHarmonicGauge` gives exactly what `TimeDerivative` gives with
// the harmonic gauge
void test_same_as_time_derivative(
    const Variables<Device::argument_tags>& args) {
  const size_t num_points = args.number_of_grid_points();
  Variables<Device::return_tags> expected{num_points};
  Variables<Host::temporary_tags> temps{num_points};
  const Mesh<dim> mesh{2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const tnsr::I<DataVector, dim> inertial_coords{num_points, 0.0};
  const InverseJacobian<DataVector, dim, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{num_points, 0.0};
  [&]<typename... TempTags>(tmpl::list<TempTags...> /*meta*/) {
    Host::apply(
        make_not_null(&get<tmpl::at_c<Device::return_tags, 0>>(expected)),
        make_not_null(&get<tmpl::at_c<Device::return_tags, 1>>(expected)),
        make_not_null(&get<tmpl::at_c<Device::return_tags, 2>>(expected)),
        make_not_null(&get<TempTags>(temps))...,
        get<tmpl::at_c<Device::argument_tags, 0>>(args),
        get<tmpl::at_c<Device::argument_tags, 1>>(args),
        get<tmpl::at_c<Device::argument_tags, 2>>(args),
        get<gr::Tags::SpacetimeMetric<DataVector, dim>>(args),
        get<gh::Tags::Pi<DataVector, dim>>(args),
        get<gh::Tags::Phi<DataVector, dim>>(args),
        get<gh::Tags::ConstraintGamma0>(args),
        get<gh::Tags::ConstraintGamma1>(args),
        get<gh::Tags::ConstraintGamma2>(args), gh::gauges::Harmonic{}, mesh,
        0.0, inertial_coords, inverse_jacobian,
        std::optional{get<Device::MeshVelocity>(args)});
  }(Host::temporary_tags{});
  // The same operations, so bitwise identical
  CHECK(apply_on_host(args, Device::argument_tags{}) == expected);
}

void test_on_device(const Variables<Device::argument_tags>& args) {
  const auto host_result = apply_on_host(args, Device::argument_tags{});
  Variables<Device::return_tags> device_result{args.number_of_grid_points()};
  [&]<typename... ArgsTags>(tmpl::list<ArgsTags...> /*meta*/) {
    copy_and_apply_pointwise_on_device<Device>(
        make_not_null(&device_result), Device::return_tags{},
        Device::argument_tags{}, get<ArgsTags>(args)...);
  }(Device::argument_tags{});
  // The device code may use fused multiply-adds
  Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_VARIABLES_CUSTOM_APPROX(device_result, host_result, custom_approx);
}

// Benchmarks of the time derivative on the device for the FD elements of the
// BNS runs: 504 elements of 17^3 cells. They only run when the test is run
// directly (ctest skips benchmarks), e.g.
//   Test_GeneralizedHarmonic "Unit.Evolution.Systems.Gh.TimeDerivative.Kokkos"
//     --benchmark-samples 20
// "copy" reads the same arguments and writes the same results with almost no
// arithmetic, which is the lower bound set by the memory bandwidth.
struct BenchmarkData {
  static constexpr size_t number_of_elements = 504;
  static constexpr size_t number_of_points = 17 * 17 * 17;
  BatchedVariables<Device::argument_tags> arguments{};
  BatchedVariables<Device::return_tags> results{};
};

BenchmarkData make_benchmark_data() {
  return []() {
    const Kokkos::DefaultExecutionSpace exec{};
    BenchmarkData result{};
    // The same arguments in all elements
    MAKE_GENERATOR(generator);
    const auto element_arguments = make_arguments(
        make_not_null(&generator), BenchmarkData::number_of_points);
    BatchedVariables<Device::argument_tags, Kokkos::HostSpace> host_element{
        BenchmarkData::number_of_points, 1};
    Variables<Device::argument_tags> host_view{};
    set_data_ref_to_element(make_not_null(&host_view), host_element, 0);
    host_view = element_arguments;
    BatchedVariables<Device::argument_tags> device_element{
        exec, BenchmarkData::number_of_points, 1};
    deep_copy(exec, make_not_null(&device_element), host_element);
    result.arguments = BatchedVariables<Device::argument_tags>{
        exec, BenchmarkData::number_of_points,
        BenchmarkData::number_of_elements};
    const auto source = device_element.view();
    const auto destination = result.arguments.view();
    Kokkos::parallel_for(
        "copy to all elements",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            exec, {0, 0, 0},
            {destination.extent(0), destination.extent(1),
             destination.extent(2)}),
        KOKKOS_LAMBDA(const size_t point, const size_t component,
                      const size_t element) {
          destination(point, component, element) = source(point, component, 0);
        });
    result.results = BatchedVariables<Device::return_tags>{
        exec, BenchmarkData::number_of_points,
        BenchmarkData::number_of_elements};
    exec.fence();
    return result;
  }();
}

void benchmark_on_device() {
  // Catch's statistics of the samples can raise floating point exceptions
  const ScopedFpeState disable_fpes{false};
  // Made by the first benchmark that runs (none run in ctest), and freed
  // before Kokkos is finalized
  std::optional<BenchmarkData> benchmark_data_storage{};
  const auto benchmark_data = [&benchmark_data_storage]() -> const auto& {
    if (not benchmark_data_storage.has_value()) {
      benchmark_data_storage = make_benchmark_data();
    }
    return *benchmark_data_storage;
  };
  BENCHMARK_ADVANCED("GH time derivative, 504 elements of 17^3 cells")
  (Catch::Benchmark::Chronometer meter) {
    const Kokkos::DefaultExecutionSpace exec{};
    const auto& data = benchmark_data();
    auto results = data.results;
    const auto arguments =
        [&]<typename... ArgsTags>(tmpl::list<ArgsTags...> /*meta*/) {
          return make_device_arguments<ArgsTags...>(
              exec, get<::Tags::BatchView<ArgsTags>>(data.arguments)...);
        }(Device::argument_tags{});
    meter.measure([&]() {
      apply_pointwise_on_device<Device>(exec, make_not_null(&results),
                                        Device::return_tags{},
                                        Device::argument_tags{}, arguments);
      exec.fence();
    });
  };
  BENCHMARK_ADVANCED("copy, 504 elements of 17^3 cells")
  (Catch::Benchmark::Chronometer meter) {
    const Kokkos::DefaultExecutionSpace exec{};
    const auto& data = benchmark_data();
    const auto source = data.arguments.view();
    const auto destination = data.results.view();
    const size_t number_of_points = source.extent(0);
    const size_t number_of_elements = source.extent(2);
    meter.measure([&]() {
      Kokkos::parallel_for(
          "copy",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
              exec, {0, 0}, {number_of_points, number_of_elements}),
          KOKKOS_LAMBDA(const size_t point, const size_t element) {
            double sum = 0.0;
            for (size_t i = 0; i < source.extent(1); ++i) {
              sum += source(point, i, element);
            }
            for (size_t i = 0; i < destination.extent(1); ++i) {
              destination(point, i, element) = sum;
            }
          });
      exec.fence();
    });
  };
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Gh.TimeDerivative.Kokkos",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  const auto args = make_arguments(make_not_null(&generator), 20);
  test_same_as_time_derivative(args);
  test_on_device(args);
  benchmark_on_device();
}
