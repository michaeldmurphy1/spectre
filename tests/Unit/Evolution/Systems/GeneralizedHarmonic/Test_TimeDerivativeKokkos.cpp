// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/DataVector.hpp"
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Gh.TimeDerivative.Kokkos",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  const auto args = make_arguments(make_not_null(&generator), 20);
  test_same_as_time_derivative(args);
  test_on_device(args);
}
