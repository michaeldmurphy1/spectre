// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivativeKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t dim = 3;
using Device = gh::TimeDerivativeOnDevice;
using Host = gh::TimeDerivative<gh::Solutions::all_solutions<dim>, dim>;
template <typename Tag>
using Batch = Device::Batch<Tag>;

using return_tags =
    tmpl::list<::Tags::dt<Device::spacetime_metric_tag>,
               ::Tags::dt<Device::pi_tag>, ::Tags::dt<Device::phi_tag>>;
using argument_tags =
    tmpl::list<Device::deriv<Device::spacetime_metric_tag>,
               Device::deriv<Device::pi_tag>, Device::deriv<Device::phi_tag>,
               Device::spacetime_metric_tag, Device::pi_tag, Device::phi_tag,
               gh::Tags::ConstraintGamma0, gh::Tags::ConstraintGamma1,
               gh::Tags::ConstraintGamma2, Device::MeshVelocity>;

// Random arguments with a spacetime metric close to Minkowski, so that the
// lapse is real
Variables<argument_tags> make_arguments(
    const gsl::not_null<std::mt19937*> generator, const size_t num_points) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  Variables<argument_tags> args{num_points};
  fill_with_random_values(make_not_null(&args), generator,
                          make_not_null(&distribution));
  auto& spacetime_metric = get<Device::spacetime_metric_tag>(args);
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

Variables<return_tags> apply_on_host(const Variables<argument_tags>& args,
                                     const bool moving_mesh) {
  const size_t num_points = args.number_of_grid_points();
  Variables<return_tags> result{num_points};
  Variables<Host::temporary_tags> temps{num_points};
  const Mesh<dim> mesh{2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const tnsr::I<DataVector, dim> inertial_coords{num_points, 0.0};
  const InverseJacobian<DataVector, dim, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{num_points, 0.0};
  [&]<typename... TempTags>(tmpl::list<TempTags...> /*meta*/) {
    Host::apply(make_not_null(&get<tmpl::at_c<return_tags, 0>>(result)),
                make_not_null(&get<tmpl::at_c<return_tags, 1>>(result)),
                make_not_null(&get<tmpl::at_c<return_tags, 2>>(result)),
                make_not_null(&get<TempTags>(temps))...,
                get<tmpl::at_c<argument_tags, 0>>(args),
                get<tmpl::at_c<argument_tags, 1>>(args),
                get<tmpl::at_c<argument_tags, 2>>(args),
                get<Device::spacetime_metric_tag>(args),
                get<Device::pi_tag>(args), get<Device::phi_tag>(args),
                get<gh::Tags::ConstraintGamma0>(args),
                get<gh::Tags::ConstraintGamma1>(args),
                get<gh::Tags::ConstraintGamma2>(args), gh::gauges::Harmonic{},
                mesh, 0.0, inertial_coords, inverse_jacobian,
                moving_mesh ? std::optional{get<Device::MeshVelocity>(args)}
                            : std::nullopt);
  }(Host::temporary_tags{});
  if (moving_mesh) {
    // The advection term that the evolution loop adds on the host
    const auto& mesh_velocity = get<Device::MeshVelocity>(args);
    tmpl::for_each<return_tags>([&](auto dt_tag_v) {
      using DtTag = tmpl::type_from<decltype(dt_tag_v)>;
      using deriv_tag = Device::deriv<typename DtTag::tag>;
      auto& dt_var = get<DtTag>(result);
      const auto& d_var = get<deriv_tag>(args);
      for (size_t i = 0; i < dt_var.size(); ++i) {
        const auto tensor_index = dt_var.get_tensor_index(i);
        for (size_t j = 0; j < dim; ++j) {
          dt_var[i] +=
              mesh_velocity.get(j) * d_var.get(prepend(tensor_index, j));
        }
      }
    });
  }
  return result;
}

void apply_on_device(const gsl::not_null<Device*> time_derivative,
                     const Kokkos::DefaultExecutionSpace& exec,
                     const BatchedVariables<return_tags>& results,
                     const BatchedVariables<argument_tags>& arguments,
                     const bool moving_mesh) {
  const auto arg = [&arguments]<typename Tag>(Tag /*meta*/) -> const auto& {
    return get<::Tags::BatchView<Tag>>(arguments);
  };
  time_derivative->apply(
      exec, get<::Tags::BatchView<tmpl::at_c<return_tags, 0>>>(results),
      get<::Tags::BatchView<tmpl::at_c<return_tags, 1>>>(results),
      get<::Tags::BatchView<tmpl::at_c<return_tags, 2>>>(results),
      arg(tmpl::at_c<argument_tags, 0>{}), arg(tmpl::at_c<argument_tags, 1>{}),
      arg(tmpl::at_c<argument_tags, 2>{}), arg(Device::spacetime_metric_tag{}),
      arg(Device::pi_tag{}), arg(Device::phi_tag{}),
      arg(gh::Tags::ConstraintGamma0{}), arg(gh::Tags::ConstraintGamma1{}),
      arg(gh::Tags::ConstraintGamma2{}),
      moving_mesh ? std::optional{arg(Device::MeshVelocity{})} : std::nullopt);
}

// The device time derivative agrees with `gh::TimeDerivative` with the
// harmonic gauge, for several elements with different data
void test_same_as_time_derivative(const gsl::not_null<std::mt19937*> generator,
                                  const bool moving_mesh) {
  const Kokkos::DefaultExecutionSpace exec{};
  const size_t number_of_points = 20;
  const size_t number_of_elements = 3;
  std::vector<Variables<argument_tags>> element_arguments{};
  BatchedVariables<argument_tags, Kokkos::HostSpace> host_arguments{
      number_of_points, number_of_elements};
  for (size_t e = 0; e < number_of_elements; ++e) {
    element_arguments.push_back(make_arguments(generator, number_of_points));
    Variables<argument_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_arguments, e);
    element = element_arguments.back();
  }
  BatchedVariables<argument_tags> arguments{exec, number_of_points,
                                            number_of_elements};
  deep_copy(exec, make_not_null(&arguments), host_arguments);
  const BatchedVariables<return_tags> results{exec, number_of_points,
                                              number_of_elements};
  Device time_derivative{};
  // Twice, to check that the cache is reused
  for (size_t i = 0; i < 2; ++i) {
    apply_on_device(make_not_null(&time_derivative), exec, results, arguments,
                    moving_mesh);
  }
  BatchedVariables<return_tags, Kokkos::HostSpace> host_results{
      number_of_points, number_of_elements};
  deep_copy(exec, make_not_null(&host_results), results);
  exec.fence();

  // The device code may use fused multiply-adds
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t e = 0; e < number_of_elements; ++e) {
    Variables<return_tags> device_result{};
    set_data_ref_to_element(make_not_null(&device_result), host_results, e);
    const auto expected = apply_on_host(element_arguments[e], moving_mesh);
    CHECK_VARIABLES_CUSTOM_APPROX(device_result, expected, custom_approx);
  }
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
  static constexpr size_t number_of_points = size_t{17} * 17 * 17;
  BatchedVariables<argument_tags> arguments{};
  BatchedVariables<return_tags> results{};
  Device time_derivative{};
};

BenchmarkData make_benchmark_data() {
  const Kokkos::DefaultExecutionSpace exec{};
  BenchmarkData result{};
  // The same arguments in all elements
  MAKE_GENERATOR(generator);
  const auto element_arguments = make_arguments(
      make_not_null(&generator), BenchmarkData::number_of_points);
  BatchedVariables<argument_tags, Kokkos::HostSpace> host_element{
      BenchmarkData::number_of_points, 1};
  Variables<argument_tags> host_view{};
  set_data_ref_to_element(make_not_null(&host_view), host_element, 0);
  host_view = element_arguments;
  BatchedVariables<argument_tags> device_element{
      exec, BenchmarkData::number_of_points, 1};
  deep_copy(exec, make_not_null(&device_element), host_element);
  result.arguments = BatchedVariables<argument_tags>{
      exec, BenchmarkData::number_of_points, BenchmarkData::number_of_elements};
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
  result.results = BatchedVariables<return_tags>{
      exec, BenchmarkData::number_of_points, BenchmarkData::number_of_elements};
  exec.fence();
  return result;
}

void benchmark_on_device() {
  // Catch's statistics of the samples can raise floating point exceptions
  const ScopedFpeState disable_fpes{false};
  // Made by the first benchmark that runs (none run in ctest), and freed
  // before Kokkos is finalized
  std::optional<BenchmarkData> benchmark_data_storage{};
  const auto benchmark_data = [&benchmark_data_storage]() -> BenchmarkData& {
    if (not benchmark_data_storage.has_value()) {
      benchmark_data_storage = make_benchmark_data();
    }
    return *benchmark_data_storage;
  };
  for (const bool moving_mesh : {false, true}) {
    BENCHMARK_ADVANCED(moving_mesh
                           ? "GH time derivative, moving mesh, 504 elements of "
                             "17^3 cells"
                           : "GH time derivative, 504 elements of 17^3 cells")
    (Catch::Benchmark::Chronometer meter) {
      const Kokkos::DefaultExecutionSpace exec{};
      auto& data = benchmark_data();
      // Allocate the cache before measuring
      apply_on_device(make_not_null(&data.time_derivative), exec, data.results,
                      data.arguments, moving_mesh);
      exec.fence();
      meter.measure([&]() {
        apply_on_device(make_not_null(&data.time_derivative), exec,
                        data.results, data.arguments, moving_mesh);
        exec.fence();
      });
    };
  }
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
  test_same_as_time_derivative(make_not_null(&generator), false);
  test_same_as_time_derivative(make_not_null(&generator), true);
  benchmark_on_device();
}
