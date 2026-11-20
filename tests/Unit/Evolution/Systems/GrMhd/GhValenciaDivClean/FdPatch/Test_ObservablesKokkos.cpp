// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/Patches/DeviceObservablesKokkos.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/ObservablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/PatchKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace fd_patch = grmhd::GhValenciaDivClean::fd_patch;
using evolved_tags = grmhd::GhValenciaDivClean::subcell::evolved_tags_on_device;
using primitive_tags =
    typename fd_patch::System::primitive_variables_tag::tags_list;
using geometry_tags = evolution::dg::subcell::fd_patch::cell_geometry_tags;
using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
using Pi = gh::Tags::Pi<DataVector, 3>;
using Phi = gh::Tags::Phi<DataVector, 3>;
using InverseJacobian =
    domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>;
using DetInvJacobian =
    domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>;

constexpr size_t number_of_elements = 2;
const Mesh<3> mesh{5, Spectral::Basis::FiniteDifference,
                   Spectral::Quadrature::CellCentered};

template <typename TagsList>
std::vector<Variables<TagsList>> host_elements(const size_t number_of_points) {
  return std::vector<Variables<TagsList>>(
      number_of_elements, Variables<TagsList>{number_of_points});
}

template <typename TagsList>
BatchedVariables<TagsList> to_device(
    const std::vector<Variables<TagsList>>& elements) {
  const size_t number_of_points = elements[0].number_of_grid_points();
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{number_of_points,
                                                           number_of_elements};
  for (size_t e = 0; e < number_of_elements; ++e) {
    const Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace,
                       Kokkos::MemoryUnmanaged>
        element(elements[e].data(), number_of_points,
                elements[e].number_of_independent_components);
    for (size_t c = 0; c < elements[e].number_of_independent_components; ++c) {
      for (size_t i = 0; i < number_of_points; ++i) {
        host_batch.view()(i, c, e) = element(i, c);
      }
    }
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<TagsList> result{exec, number_of_points, number_of_elements};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

// The values of a component of the observables at the grid points of an
// element
DataVector observable(const Kokkos::View<double***, Kokkos::LayoutLeft,
                                         Kokkos::HostSpace>& values,
                      const size_t component, const size_t e) {
  DataVector result(values.extent(0));
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = values(i, component, e);
  }
  return result;
}

template <typename TensorType>
void check_observable(const Kokkos::View<double***, Kokkos::LayoutLeft,
                                         Kokkos::HostSpace>& values,
                      const evolution::patches::DeviceObservables& observables,
                      const std::string& name, const TensorType& expected,
                      const size_t e) {
  CAPTURE(name);
  // The constraints are sums of terms that cancel, in a different order than
  // on the host
  const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  for (const auto& tensor : observables.tensors) {
    if (tensor.name == name) {
      CHECK(tensor.component_names == expected.get_vector_of_data().first);
      for (size_t c = 0; c < expected.size(); ++c) {
        CHECK_ITERABLE_CUSTOM_APPROX(
            observable(values, tensor.first_component + c, e), expected[c],
            custom_approx);
      }
      return;
    }
  }
  CHECK(false);
}

void test_device_observables() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist{-0.1, 0.1};
  const size_t number_of_points = mesh.number_of_grid_points();
  const DataVector used_for_size(number_of_points);

  auto evolved = host_elements<evolved_tags>(number_of_points);
  auto primitives = host_elements<primitive_tags>(number_of_points);
  auto geometry = host_elements<geometry_tags>(number_of_points);
  for (size_t e = 0; e < number_of_elements; ++e) {
    fill_with_random_values(make_not_null(&evolved[e]), make_not_null(&gen),
                            make_not_null(&dist));
    fill_with_random_values(make_not_null(&primitives[e]), make_not_null(&gen),
                            make_not_null(&dist));
    // A physical spacetime metric, with a smooth variation so that the
    // derivatives are meaningful
    const auto spatial_metric = TestHelpers::gr::random_spatial_metric<3>(
        make_not_null(&gen), used_for_size);
    get<SpacetimeMetric>(evolved[e]) = gr::spacetime_metric(
        TestHelpers::gr::random_lapse(make_not_null(&gen), used_for_size),
        TestHelpers::gr::random_shift<3>(make_not_null(&gen), used_for_size),
        spatial_metric);
    // An inverse Jacobian that varies between the grid points, so that the
    // integrals weight the points differently
    auto& inverse_jacobian = get<InverseJacobian>(geometry[e]);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        inverse_jacobian.get(i, j) =
            (i == j ? 2.0 + static_cast<double>(e) : 0.0) +
            make_with_random_values<DataVector>(
                make_not_null(&gen), make_not_null(&dist), used_for_size);
      }
    }
    get(get<DetInvJacobian>(geometry[e])) = get(determinant(inverse_jacobian));
  }

  const Kokkos::DefaultExecutionSpace exec{};
  const auto device_evolved = to_device(evolved);
  const auto device_primitives = to_device(primitives);
  const auto device_geometry = to_device(geometry);
  BatchedVariables<grmhd::GhValenciaDivClean::subcell::fixing_metric_tags>
      fixing_metric{exec, number_of_points, number_of_elements};
  grmhd::GhValenciaDivClean::subcell::fixing_metric_on_device(
      exec, make_not_null(&fixing_metric),
      get<::Tags::BatchView<SpacetimeMetric>>(device_evolved));

  const std::vector<std::string> tensor_names{
      "PointwiseL2Norm(ThreeIndexConstraint)",
      "RestMassDensity",
      "GaugeConstraint",
      "Phi",
      "ThreeIndexConstraint",
      "PointwiseL2Norm(GaugeConstraint)",
      "GaugeConstraint"};
  for (const auto& name : tensor_names) {
    CHECK(fd_patch::is_device_observable<primitive_tags>(name));
  }
  CHECK_FALSE(fd_patch::is_device_observable<primitive_tags>("Lapse"));
  evolution::patches::DeviceObservables observables{};
  fd_patch::DeviceObservablesBuffers buffers{};
  fd_patch::device_observables(
      exec, make_not_null(&observables), make_not_null(&buffers), tensor_names,
      device_evolved, device_primitives, fixing_metric, device_geometry, mesh);
  // Each tensor once
  CHECK(observables.tensors.size() == tensor_names.size() - 1);
  const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          observables.values);
  const auto statistics =
      evolution::patches::norm_statistics_on_device(exec, observables);
  CHECK(statistics.extent(0) == number_of_elements);
  CHECK(statistics.extent(1) == observables.values.extent(1) + 1);

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const auto& spacetime_metric = get<SpacetimeMetric>(evolved[e]);
    const auto& pi = get<Pi>(evolved[e]);
    const auto& phi = get<Phi>(evolved[e]);
    // As the compute tags of an element with the harmonic gauge
    const auto spatial_metric = gr::spatial_metric(spacetime_metric);
    const auto inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
    const auto lapse = gr::lapse(shift, spacetime_metric);
    const auto gauge_constraint = gh::gauge_constraint(
        make_with_value<tnsr::a<DataVector, 3>>(used_for_size, 0.0),
        gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse),
        gr::spacetime_normal_vector(lapse, shift), inverse_spatial_metric,
        gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric), pi,
        phi);
    tnsr::iaa<DataVector, 3> d_spacetime_metric{number_of_points};
    partial_derivative(make_not_null(&d_spacetime_metric), spacetime_metric,
                       mesh, get<InverseJacobian>(geometry[e]));
    const auto three_index_constraint =
        gh::three_index_constraint(d_spacetime_metric, phi);

    check_observable(
        values, observables, "RestMassDensity",
        get<hydro::Tags::RestMassDensity<DataVector>>(primitives[e]), e);
    check_observable(values, observables, "Phi", phi, e);
    check_observable(values, observables, "GaugeConstraint", gauge_constraint,
                     e);
    check_observable(values, observables, "PointwiseL2Norm(GaugeConstraint)",
                     pointwise_l2_norm(gauge_constraint), e);
    check_observable(values, observables, "ThreeIndexConstraint",
                     three_index_constraint, e);
    check_observable(values, observables,
                     "PointwiseL2Norm(ThreeIndexConstraint)",
                     pointwise_l2_norm(three_index_constraint), e);

    // The norm statistics, as `ObserveNorms` computes them on the host
    const DataVector det_jacobian = 1.0 / get(get<DetInvJacobian>(geometry[e]));
    const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
    for (size_t c = 0; c < observables.values.extent(1); ++c) {
      CAPTURE(c);
      const DataVector component = observable(values, c, e);
      const Events::NormStatistics& result = statistics(e, c);
      CHECK(result.max == max(component));
      CHECK(result.min == min(component));
      CHECK(result.sum_abs ==
            custom_approx(alg::accumulate(abs(component), 0.0)));
      CHECK(result.sum_square ==
            custom_approx(alg::accumulate(square(component), 0.0)));
      CHECK(result.integral_abs == custom_approx(definite_integral(
                                       abs(component) * det_jacobian, mesh)));
      CHECK(result.integral_square ==
            custom_approx(
                definite_integral(square(component) * det_jacobian, mesh)));
      CHECK(result.integral ==
            custom_approx(definite_integral(component * det_jacobian, mesh)));
    }
    // The volume
    CHECK(statistics(e, observables.values.extent(1)).integral ==
          custom_approx(definite_integral(det_jacobian, mesh)));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.FdPatch.ObservablesKokkos",
    "[Unit][Evolution]") {
  test_device_observables();
}
