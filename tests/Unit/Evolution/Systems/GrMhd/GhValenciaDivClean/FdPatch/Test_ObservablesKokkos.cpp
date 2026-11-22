// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/Patches/DeviceComputeTagsKokkos.hpp"
#include "Evolution/Patches/DeviceComputeTagsKokkos.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/ObservablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/PatchKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/ObservablesHostValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.tpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.tpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.tpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.tpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.tpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.hpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.tpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/DeviceParametersKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace fd_patch = grmhd::GhValenciaDivClean::fd_patch;
namespace device_compute = evolution::patches::device_compute;
using evolved_tags = grmhd::GhValenciaDivClean::subcell::evolved_tags_on_device;
using primitive_tags =
    typename fd_patch::System::primitive_variables_tag::tags_list;
using geometry_tags = evolution::dg::subcell::fd_patch::cell_geometry_tags;
using fixing_metric_tags =
    grmhd::GhValenciaDivClean::subcell::fixing_metric_tags;
using Sources = fd_patch::DeviceObservableSources<primitive_tags>;

using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
using Pi = gh::Tags::Pi<DataVector, 3>;
using Phi = gh::Tags::Phi<DataVector, 3>;
using InverseJacobian =
    domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>;
using DetInvJacobian =
    domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>;
using GridCoordinates = domain::Tags::Coordinates<3, Frame::Grid>;

// The compute tags of the observable tensors, as in the ObserveNorms of
// EvolveGhValenciaDivClean
using observed_compute_tags = tmpl::list<
    gr::Tags::SpacetimeNormalOneFormCompute<DataVector, 3, Frame::Inertial>,
    gr::Tags::SpacetimeNormalVectorCompute<DataVector, 3, Frame::Inertial>,
    gr::Tags::InverseSpacetimeMetricCompute<DataVector, 3, Frame::Inertial>,
    gh::Tags::GaugeConstraintCompute<3, Frame::Inertial>,
    gh::Tags::TwoIndexConstraintCompute<3, Frame::Inertial>,
    gh::Tags::ThreeIndexConstraintCompute<3, Frame::Inertial>,
    gh::Tags::FourIndexConstraintCompute<3, Frame::Inertial>,
    gh::Tags::FConstraintCompute<3, Frame::Inertial>,
    gh::Tags::ConstraintEnergyCompute<3, Frame::Inertial>,
    ::Tags::PointwiseL2NormCompute<gh::Tags::GaugeConstraint<DataVector, 3>>,
    ::Tags::PointwiseL2NormCompute<gh::Tags::ConstraintEnergy<DataVector, 3>>,
    hydro::Tags::MassWeightedInternalEnergyCompute<DataVector>,
    hydro::Tags::MassWeightedKineticEnergyCompute<DataVector>,
    hydro::Tags::TildeDUnboundUtCriterionCompute<DataVector, 3,
                                                 Frame::Inertial>>;
using Universe =
    tmpl::append<fd_patch::element_box_compute_tags, observed_compute_tags>;

// An executable with one ObserveNorms that can observe the tensors
struct Metavariables {
  struct factory_creation {
    using factory_classes = tmpl::map<tmpl::pair<
        Event, tmpl::list<Events::ObserveNorms<tmpl::push_front<
                   observed_compute_tags, gr::Tags::Lapse<DataVector>,
                   gr::Tags::DetSpatialMetric<DataVector>,
                   hydro::Tags::RestMassDensity<DataVector>, Phi>>>>>;
  };
};

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

using HostValues = TestHelpers::GhValenciaDivClean::fd_patch::HostValues;

HostValues host_values(const Variables<evolved_tags>& evolved,
                       const Variables<primitive_tags>& primitives,
                       const Variables<geometry_tags>& geometry,
                       const tnsr::I<DataVector, 3, Frame::Grid>& grid_coords,
                       const ConstraintDamping::GaussianSum<3>& gamma2) {
  return TestHelpers::GhValenciaDivClean::fd_patch::host_values(
      mesh, get<SpacetimeMetric>(evolved), get<Pi>(evolved), get<Phi>(evolved),
      get<grmhd::ValenciaDivClean::Tags::TildeD>(evolved),
      get<hydro::Tags::RestMassDensity<DataVector>>(primitives),
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(primitives),
      get<hydro::Tags::LorentzFactor<DataVector>>(primitives),
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(primitives),
      get<InverseJacobian>(geometry), grid_coords, gamma2);
}

// The tensor `Tag` of the elements of `range` from the device
template <typename TensorType>
TensorType from_device(const device_compute::View& view, const size_t e) {
  // The views of base tensors are strided subviews of the batches, which
  // can't be copied between memory spaces directly
  const Kokkos::View<double***, Kokkos::LayoutRight> contiguous{
      "contiguous", view.extent(0), view.extent(1), view.extent(2)};
  Kokkos::deep_copy(contiguous, view);
  const auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, contiguous);
  TensorType result{view.extent(0)};
  for (size_t c = 0; c < result.size(); ++c) {
    for (size_t i = 0; i < view.extent(0); ++i) {
      result[c][i] = host_view(i, c, e);
    }
  }
  return result;
}

struct Setup {
  std::vector<Variables<evolved_tags>> evolved;
  std::vector<Variables<primitive_tags>> primitives;
  std::vector<Variables<geometry_tags>> geometry;
  std::vector<tnsr::I<DataVector, 3, Frame::Grid>> grid_coordinates;
  ConstraintDamping::GaussianSum<3> gamma2;
};

Setup make_setup(const gsl::not_null<std::mt19937*> gen) {
  std::uniform_real_distribution<double> dist{-0.1, 0.1};
  std::uniform_real_distribution<double> positive{0.5, 2.0};
  const size_t number_of_points = mesh.number_of_grid_points();
  const DataVector used_for_size(number_of_points);
  Setup setup{.evolved = host_elements<evolved_tags>(number_of_points),
              .primitives = host_elements<primitive_tags>(number_of_points),
              .geometry = host_elements<geometry_tags>(number_of_points),
              .grid_coordinates = {},
              .gamma2 = {}};
  for (size_t e = 0; e < number_of_elements; ++e) {
    fill_with_random_values(make_not_null(&setup.evolved[e]), gen,
                            make_not_null(&dist));
    fill_with_random_values(make_not_null(&setup.primitives[e]), gen,
                            make_not_null(&positive));
    // A physical spacetime metric
    const auto spatial_metric =
        TestHelpers::gr::random_spatial_metric<3>(gen, used_for_size);
    get<SpacetimeMetric>(setup.evolved[e]) = gr::spacetime_metric(
        TestHelpers::gr::random_lapse(gen, used_for_size),
        TestHelpers::gr::random_shift<3>(gen, used_for_size), spatial_metric);
    // An inverse Jacobian that varies between the grid points
    auto& inverse_jacobian = get<InverseJacobian>(setup.geometry[e]);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        inverse_jacobian.get(i, j) =
            (i == j ? 2.0 + static_cast<double>(e) : 0.0) +
            make_with_random_values<DataVector>(gen, make_not_null(&dist),
                                                used_for_size);
      }
    }
    get(get<DetInvJacobian>(setup.geometry[e])) =
        get(determinant(inverse_jacobian));
    setup.grid_coordinates.push_back(
        make_with_random_values<tnsr::I<DataVector, 3, Frame::Grid>>(
            gen, make_not_null(&dist), used_for_size));
  }
  setup.gamma2.constant = 0.3;
  setup.gamma2.number_of_gaussians = 1;
  setup.gamma2.gaussians[0] = {
      .amplitude = 2.0, .exponent_factor = 4.0, .center = {0.01, 0.0, -0.02}};
  return setup;
}

// The device batches and the sources of an observation
struct Device {
  explicit Device(const Setup& setup)
      : evolved(to_device(setup.evolved)),
        primitives(to_device(setup.primitives)),
        geometry(to_device(setup.geometry)) {
    const Kokkos::DefaultExecutionSpace exec{};
    fixing_metric = BatchedVariables<fixing_metric_tags>{
        exec, mesh.number_of_grid_points(), number_of_elements};
    grmhd::GhValenciaDivClean::subcell::fixing_metric_on_device(
        exec, make_not_null(&fixing_metric),
        get<::Tags::BatchView<SpacetimeMetric>>(evolved));
    std::vector<Variables<tmpl::list<GridCoordinates>>> coordinates{};
    for (const auto& coords : setup.grid_coordinates) {
      coordinates.emplace_back(coords.begin()->size());
      get<GridCoordinates>(coordinates.back()) = coords;
    }
    grid_coordinates = to_device(coordinates);
    damping_functions.set(exec,
                          std::array{setup.gamma2, setup.gamma2, setup.gamma2});
    exec.fence();
  }

  Sources sources() {
    return Sources{.evolved_variables = &evolved,
                   .primitives = &primitives,
                   .fixing_metric = &fixing_metric,
                   .cell_geometry = &geometry,
                   .grid_coordinates = grid_coordinates.view(),
                   .damping_functions = damping_functions.device_array(),
                   .mesh = mesh,
                   // The quadrature weight of all cell centers, (2 / 5)^3
                   .weight = cube(2.0 / 5.0),
                   .buffers = &scratch};
  }

  BatchedVariables<evolved_tags> evolved;
  BatchedVariables<primitive_tags> primitives;
  BatchedVariables<geometry_tags> geometry;
  BatchedVariables<fixing_metric_tags> fixing_metric{};
  BatchedVariables<tmpl::list<GridCoordinates>> grid_coordinates{};
  DeviceParameters<std::array<ConstraintDamping::GaussianSum<3>, 3>>
      damping_functions{"damping functions"};
  device_compute::Scratch scratch{};
};

void test_evaluation(const Setup& setup, const device_compute::Mode mode) {
  CAPTURE(static_cast<int>(mode));
  Device device{setup};
  const Sources sources = device.sources();
  const Kokkos::DefaultExecutionSpace exec{};
  // The tensors evaluate the same on all elements and on a range of them
  std::vector<device_compute::ElementRange> ranges{
      {.first = 1, .size = number_of_elements - 1}};
  if (mode == device_compute::Mode::Generic) {
    ranges.push_back({.first = 0, .size = number_of_elements});
  }
  for (const auto& range : ranges) {
    device_compute::detail::Evaluator<Sources, Universe> evaluator{
        exec, sources, range, mode};
    for (size_t e = 0; e < range.size; ++e) {
      const size_t element = range.first + e;
      CAPTURE(element);
      const auto expected =
          host_values(setup.evolved[element], setup.primitives[element],
                      setup.geometry[element], setup.grid_coordinates[element],
                      setup.gamma2);
      // The constraints are sums of terms that cancel, so their roundoff is
      // relative to the terms
      const Approx custom_approx = Approx::custom().epsilon(1.e-11).scale(1.0);
      tmpl::for_each<typename decltype(expected.values)::tags_list>(
          [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
            CAPTURE(db::tag_name<Tag>());
            CHECK(device_compute::is_provided<Tag, Sources, Universe>);
            const device_compute::View view = evaluator.template get<Tag>();
            CHECK_ITERABLE_CUSTOM_APPROX(
                from_device<typename Tag::type>(view, e),
                get<Tag>(expected.values), custom_approx);
          });
    }
  }
}

void test_norm_statistics(const Setup& setup) {
  Device device{setup};
  const Sources sources = device.sources();
  const Kokkos::DefaultExecutionSpace exec{};
  const std::vector<std::string> tensor_names{
      "PointwiseL2Norm(ConstraintEnergy)",
      "RestMassDensity",
      "GaugeConstraint",
      "Lapse",
      "TildeDUnboundUtCriterion",
      "GaugeConstraint",
      "Phi"};
  for (const auto& name : tensor_names) {
    CHECK(device_compute::is_device_observable<Metavariables, Sources>(name));
  }
  CHECK_FALSE(device_compute::is_device_observable<Metavariables, Sources>(
      "SpatialRicciScalar"));
  const auto result = device_compute::device_norm_statistics<Metavariables>(
      exec, sources, tensor_names);
  // Each tensor once
  REQUIRE(result.tensors.size() == tensor_names.size() - 1);
  REQUIRE(result.statistics.extent(0) == number_of_elements);

  const Approx custom_approx = Approx::custom().epsilon(1.e-11).scale(1.0);
  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const auto expected =
        host_values(setup.evolved[e], setup.primitives[e], setup.geometry[e],
                    setup.grid_coordinates[e], setup.gamma2);
    const DataVector det_jacobian =
        1.0 / get(get<DetInvJacobian>(setup.geometry[e]));
    size_t column = 0;
    for (const auto& tensor : result.tensors) {
      CAPTURE(tensor.name);
      CHECK(tensor.first_component == column);
      tmpl::for_each<typename decltype(expected.values)::tags_list>(
          [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
            if (db::tag_name<Tag>() != tensor.name) {
              return;
            }
            const auto& value = get<Tag>(expected.values);
            CHECK(tensor.component_names == value.get_vector_of_data().first);
            for (size_t c = 0; c < value.size(); ++c) {
              const DataVector& component = value[c];
              const Events::NormStatistics& statistics =
                  result.statistics(e, column + c);
              CHECK(statistics.max == custom_approx(max(component)));
              CHECK(statistics.min == custom_approx(min(component)));
              CHECK(statistics.sum_abs ==
                    custom_approx(alg::accumulate(abs(component), 0.0)));
              CHECK(statistics.sum_square ==
                    custom_approx(alg::accumulate(square(component), 0.0)));
              CHECK(statistics.integral_abs ==
                    custom_approx(definite_integral(
                        abs(component) * det_jacobian, mesh)));
              CHECK(statistics.integral_square ==
                    custom_approx(definite_integral(
                        square(component) * det_jacobian, mesh)));
              CHECK(statistics.integral ==
                    custom_approx(
                        definite_integral(component * det_jacobian, mesh)));
            }
          });
      column += tensor.component_names.size();
    }
    // The volume
    CHECK(result.statistics(e, column).integral ==
          custom_approx(definite_integral(det_jacobian, mesh)));
  }
}
}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.FdPatch.ObservablesKokkos",
    "[Unit][Evolution]") {
  // One element at a time, so that the elements are evaluated in chunks. No
  // other thread reads the environment here.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  setenv("SPECTRE_DEVICE_OBSERVABLES_CHUNK", "1", 1);
  MAKE_GENERATOR(gen);
  const Setup setup = make_setup(make_not_null(&gen));
  test_evaluation(setup, device_compute::Mode::Generic);
  test_evaluation(setup, device_compute::Mode::GenericOutOfLine);
  test_norm_statistics(setup);
}
