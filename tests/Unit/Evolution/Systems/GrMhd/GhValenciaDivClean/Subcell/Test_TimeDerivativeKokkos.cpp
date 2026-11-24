// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Patches/GridToInertial.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FilterOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianPlusConstant.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean {
namespace {
namespace fd_patch = evolution::dg::subcell::fd_patch;
using TheSystem =
    GhValenciaDivClean::System<RadiationTransport::NoNeutrinos::System>;
using prim_tags = typename TheSystem::primitive_variables_tag::tags_list;
using evolved_tags = typename TheSystem::variables_tag::tags_list;
using gh_tags = typename TheSystem::gh_system::variables_tag::tags_list;
using ghost_tags = Tags::primitive_grmhd_and_spacetime_reconstruction_tags;

// Needed by `BoundaryConditionGhostData` in the host time derivative
struct DummyAnalyticSolutionTag : db::SimpleTag {
  using type =
      gh::Solutions::WrappedGr<::RelativisticEuler::Solutions::TovStar>;
};

struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = true;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            BoundaryConditions::BoundaryCondition,
            tmpl::push_back<
                BoundaryConditions::standard_boundary_conditions<TheSystem>,
                BoundaryConditions::DirichletAnalytic<TheSystem>>>,
        tmpl::pair<evolution::BoundaryCorrection,
                   BoundaryCorrections::standard_boundary_corrections>,
        tmpl::pair<evolution::initial_data::InitialData,
                   ghmhd::GhValenciaDivClean::InitialData::
                       analytic_solutions_and_data_list>>;
  };
};

template <typename Tags, typename HostVars>
BatchedVariables<Tags> to_device_batch(
    const Kokkos::DefaultExecutionSpace& exec, const HostVars& host_vars) {
  BatchedVariables<Tags, Kokkos::HostSpace> host_batch{
      host_vars.number_of_grid_points(), 1};
  Variables<Tags> element{};
  set_data_ref_to_element(make_not_null(&element), host_batch, 0);
  element.assign_subset(host_vars);
  BatchedVariables<Tags> result{exec, host_batch.number_of_grid_points(), 1};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

template <typename Tags>
Variables<Tags> to_host(const Kokkos::DefaultExecutionSpace& exec,
                        const BatchedVariables<Tags>& batch) {
  BatchedVariables<Tags, Kokkos::HostSpace> host_batch{
      batch.number_of_grid_points(), batch.number_of_elements()};
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();
  Variables<Tags> element{};
  set_data_ref_to_element(make_not_null(&element), host_batch, 0);
  // A copy that owns its data, since the batch is freed on return
  Variables<Tags> result = element;
  return result;
}

// The primitive and GH variables of the TOV star at the `coords`, with random
// perturbations so that the time derivatives don't nearly vanish, and with a
// velocity, a magnetic field and a temperature so that all terms contribute
template <typename Solution>
std::pair<Variables<prim_tags>, Variables<gh_tags>> perturbed_variables(
    const gsl::not_null<std::mt19937*> generator, const Solution& soln,
    const tnsr::I<DataVector, 3>& coords) {
  const size_t number_of_points = get<0>(coords).size();
  Variables<prim_tags> prims{number_of_points};
  prims.assign_subset(soln.variables(coords, 0.0, prim_tags{}));
  Variables<gh_tags> gh_vars{number_of_points};
  gh_vars.assign_subset(soln.variables(coords, 0.0, gh_tags{}));
  std::uniform_real_distribution<> unit(-1.0, 1.0);
  const auto random = [&generator, &unit, number_of_points]() {
    return make_with_random_values<DataVector>(generator, make_not_null(&unit),
                                               DataVector{number_of_points});
  };
  get(get<hydro::Tags::RestMassDensity<DataVector>>(prims)) *=
      1.0 + 0.05 * random();
  get(get<hydro::Tags::Temperature<DataVector>>(prims)) =
      0.01 + 0.005 * random();
  get(get<hydro::Tags::ElectronFraction<DataVector>>(prims)) =
      0.15 + 0.05 * random();
  get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(prims)) =
      1.0e-3 * random();
  auto& velocity = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prims);
  auto& magnetic_field = get<hydro::Tags::MagneticField<DataVector, 3>>(prims);
  for (size_t i = 0; i < 3; ++i) {
    velocity.get(i) = 0.05 * random();
    magnetic_field.get(i) = 1.0e-3 * random();
  }
  for (auto& component :
       get<gr::Tags::SpacetimeMetric<DataVector, 3>>(gh_vars)) {
    component += 1.0e-4 * random();
  }
  for (auto& component : get<gh::Tags::Pi<DataVector, 3>>(gh_vars)) {
    component += 1.0e-3 * random();
  }
  for (auto& component : get<gh::Tags::Phi<DataVector, 3>>(gh_vars)) {
    component += 1.0e-3 * random();
  }
  // The Lorentz factor of the velocity in the perturbed metric
  const auto spatial_metric = gr::spatial_metric(
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(gh_vars));
  DataVector v_squared{number_of_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_squared += spatial_metric.get(i, j) * velocity.get(i) * velocity.get(j);
    }
  }
  get(get<hydro::Tags::LorentzFactor<DataVector>>(prims)) =
      1.0 / sqrt(1.0 - v_squared);
  return {std::move(prims), std::move(gh_vars)};
}

// The ghost variables of the primitive and GH variables
Variables<ghost_tags> ghost_variables(const Variables<prim_tags>& prims,
                                      const Variables<gh_tags>& gh_vars) {
  Variables<ghost_tags> result{prims.number_of_grid_points()};
  result.assign_subset(prims);
  result.assign_subset(gh_vars);
  auto& lorentz_factor_times_velocity =
      get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
          result);
  for (size_t i = 0; i < 3; ++i) {
    lorentz_factor_times_velocity.get(i) =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prims).get(i) *
        get(get<hydro::Tags::LorentzFactor<DataVector>>(prims));
  }
  return result;
}

// The FD time derivative of one element of a perturbed TOV star on the device
// is the same as on the host. With `div_per_cell`, the divergence of the mesh
// velocity varies in the element and the device gets it at each cell, as the
// host projects it.
template <typename Reconstructor>
void test(const gsl::not_null<std::mt19937*> generator,
          const Reconstructor& recons,
          const std::optional<double>& expansion_velocity,
          const bool div_per_cell = false) {
  CAPTURE(pretty_type::short_name<Reconstructor>());
  CAPTURE(recons.ghost_zone_size());
  CAPTURE(expansion_velocity);
  CAPTURE(div_per_cell);
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using BoundaryCorrection = BoundaryCorrections::ProductOfCorrections<
      gh::BoundaryCorrections::UpwindPenalty<3>,
      ValenciaDivClean::BoundaryCorrections::Hll>;
  using DampingFunction = ConstraintDamping::DampingFunction<3, Frame::Grid>;
  using variables_tag = typename TheSystem::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  const gh::Solutions::WrappedGr<::RelativisticEuler::Solutions::TovStar> soln{
      1.28e-3,
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0)
          ->get_clone(),
      RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};
  const Affine affine_map{-1.0, 1.0, -4.0, 4.0};
  const Affine3D coordinate_map{affine_map, affine_map, affine_map};
  std::vector<Block<3>> blocks;
  blocks.emplace_back(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          coordinate_map),
      0, DirectionMap<3, BlockNeighbors<3>>{});
  // An element with neighbors in all directions
  const ElementId<3> element_id{
      0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}};
  const auto element = domain::create_initial_element(
      element_id, blocks,
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});
  const ElementMap<3, Frame::Grid> element_map{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          coordinate_map)};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const double time = 0.0;
  const domain::FunctionsOfTimeMap functions_of_time{};

  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_cells = subcell_mesh.number_of_grid_points();
  const size_t ghost_width = recons.ghost_zone_size();
  const auto cell_coords =
      (*grid_to_inertial_map)(element_map(logical_coordinates(subcell_mesh)));
  const auto dg_coords =
      (*grid_to_inertial_map)(element_map(logical_coordinates(dg_mesh)));

  // The variables in the element and the ghost data from the neighbors
  const auto [cell_prims, cell_gh_vars] =
      perturbed_variables(generator, soln, cell_coords);
  evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      neighbor_data{};
  DirectionMap<3, DataVector> ghost_data_in_directions{};
  for (const Direction<3>& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    const auto [neighbor_prims, neighbor_gh_vars] = perturbed_variables(
        generator, soln,
        (*grid_to_inertial_map)(element_map(neighbor_logical_coords)));
    const auto neighbor_ghost_vars =
        ghost_variables(neighbor_prims, neighbor_gh_vars);
    ghost_data_in_directions[direction] =
        evolution::dg::subcell::slice_data(
            neighbor_ghost_vars, subcell_mesh.extents(), ghost_width,
            std::unordered_set{direction.opposite()}, 0, {})
            .at(direction.opposite());
    const DirectionalId<3> key{direction,
                               *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        ghost_data_in_directions.at(direction);
  }
  typename variables_tag::type cell_vars{number_of_cells};
  cell_vars.assign_subset(cell_gh_vars);

  // A mesh velocity that is linear in the coordinates, as in the test of
  // `subcell::TimeDerivative`
  std::optional<tnsr::I<DataVector, 3>> dg_mesh_velocity{};
  std::optional<Scalar<DataVector>> div_dg_mesh_velocity{};
  if (expansion_velocity.has_value()) {
    dg_mesh_velocity = tnsr::I<DataVector, 3>{dg_mesh.number_of_grid_points()};
    for (size_t i = 0; i < 3; ++i) {
      dg_mesh_velocity->get(i) = dg_coords.get(i) * *expansion_velocity;
    }
    div_dg_mesh_velocity = Scalar<DataVector>{dg_mesh.number_of_grid_points(),
                                              3.0 * *expansion_velocity};
    if (div_per_cell) {
      get(*div_dg_mesh_velocity) += 0.2 * get<0>(dg_coords);
    }
  }

  const auto damping_function = [](const double constant,
                                   const double amplitude, const double width) {
    return std::unique_ptr<DampingFunction>(
        std::make_unique<
            ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
            constant, amplitude, width, std::array{0.0, 0.0, 0.0}));
  };
  const std::array<std::unique_ptr<DampingFunction>, 3> damping_functions{
      damping_function(0.01, 0.09 / 1.4, 5.5 * 1.4),
      damping_function(-0.999, 0.999, 100.0),
      damping_function(0.01, 1.35 / 1.4, 5.5 * 1.4)};
  const double spacetime_dissipation = 0.1;
  // The density of the star in the element is about 1.2e-3, so the cutoff
  // fixes a part of the reconstructed states to the atmosphere if the
  // reconstructor does
  const ::VariableFixing::FixToAtmosphere<3> fix_to_atmosphere{
      1.1e-3, 1.2e-3, std::nullopt, std::nullopt};

  const auto inv_jacobian_dg =
      element_map.inv_jacobian(logical_coordinates(dg_mesh));
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      dg_inv_jacobian{dg_mesh.number_of_grid_points()};
  for (size_t i = 0; i < dg_inv_jacobian.size(); ++i) {
    dg_inv_jacobian[i] = inv_jacobian_dg[i];
  }
  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, evolution::dg::subcell::Tags::Mesh<3>,
          domain::Tags::Mesh<3>, fd::Tags::Reconstructor<TheSystem>,
          evolution::Tags::BoundaryCorrection,
          hydro::Tags::GrmhdEquationOfState,
          typename TheSystem::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
          evolution::dg::subcell::Tags::ReconstructionOrder<3>,
          ValenciaDivClean::Tags::ConstraintDampingParameter,
          evolution::dg::Tags::MortarData<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::Domain<3>, domain::Tags::ExternalBoundaryConditions<3>,
          domain::Tags::MeshVelocity<3, Frame::Inertial>,
          domain::Tags::DivMeshVelocity,
          domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                        Frame::Inertial>,
          evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
          domain::Tags::FunctionsOfTimeInitialize, DummyAnalyticSolutionTag,
          Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars>,
          gh::Tags::DampingFunctionGamma0<3, Frame::Grid>,
          gh::Tags::DampingFunctionGamma1<3, Frame::Grid>,
          gh::Tags::DampingFunctionGamma2<3, Frame::Grid>,
          ::gh::gauges::Tags::GaugeCondition,
          grmhd::GhValenciaDivClean::fd::Tags::FilterOptions,
          ::Tags::VariableFixer<::VariableFixing::FixToAtmosphere<3>>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<3,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          evolution::dg::subcell::Tags::InertialCoordinatesCompute<
              ::domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                            Frame::Inertial>>,
          evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGridCompute<
              ::domain::Tags::ElementMap<3, Frame::Grid>, 3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToGridCompute<3>,
          evolution::dg::subcell::fd::Tags::
              InverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      3, Frame::Grid, Frame::Inertial>,
                  3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      3, Frame::Grid, Frame::Inertial>,
                  3>,
          domain::Tags::DetInvJacobianCompute<3, Frame::ElementLogical,
                                              Frame::Inertial>,
          gr::Tags::SpatialMetricCompute<DataVector, 3, Frame::Inertial>,
          gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3,
                                                      Frame::Inertial>,
          gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3,
                                                Frame::Inertial>>>(
      element, subcell_mesh, dg_mesh,
      std::unique_ptr<fd::Reconstructor<TheSystem>>{
          std::make_unique<Reconstructor>(recons)},
      std::unique_ptr<evolution::BoundaryCorrection>{
          std::make_unique<BoundaryCorrection>(
              gh::BoundaryCorrections::UpwindPenalty<3>{},
              ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8})},
      soln.equation_of_state().promote_to_3d_eos(), cell_prims,
      Variables<typename dt_variables_tag::tags_list>{number_of_cells},
      cell_vars, neighbor_data,
      typename evolution::dg::subcell::Tags::ReconstructionOrder<3>::type{},
      1.0, evolution::dg::Tags::MortarData<3>::type{},
      ElementMap<3, Frame::Grid>{element_id,
                                 element_map.block_map().get_clone()},
      grid_to_inertial_map->get_clone(), Domain<3>{std::move(blocks)},
      std::vector<DirectionMap<
          3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>(
          1),
      dg_mesh_velocity, div_dg_mesh_velocity, dg_inv_jacobian,
      typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type{}, time,
      clone_unique_ptrs(functions_of_time), soln, DummyEvolutionMetaVars{},
      damping_functions[0]->get_clone(), damping_functions[1]->get_clone(),
      damping_functions[2]->get_clone(),
      std::unique_ptr<gh::gauges::GaugeCondition>(
          std::make_unique<gh::gauges::Harmonic>()),
      grmhd::GhValenciaDivClean::fd::FilterOptions{spacetime_dissipation},
      fix_to_atmosphere);
  db::mutate_apply<ValenciaDivClean::ConservativeFromPrimitive>(
      make_not_null(&box));
  // The evolved variables before the host filters them
  const auto evolved_before = db::get<variables_tag>(box);
  subcell::TimeDerivative<TheSystem>::apply(make_not_null(&box));
  const auto& expected_dt_vars = db::get<dt_variables_tag>(box);
  const auto& expected_filtered_vars = db::get<variables_tag>(box);

  // The same on the device
  const Kokkos::DefaultExecutionSpace exec{};
  auto evolved =
      to_device_batch<subcell::evolved_tags_on_device>(exec, evolved_before);
  const auto primitives =
      to_device_batch<hydro::grmhd_tags<DataVector>>(exec, cell_prims);
  const auto padded = ::fd::device::make_volume_view(
      exec, "padded", 1,
      Variables<ghost_tags>::number_of_independent_components,
      subcell_mesh.extents(), ghost_width);
  {
    const auto host_padded = Kokkos::create_mirror_view(padded);
    Variables<gh_tags> volume_gh_vars{number_of_cells};
    volume_gh_vars.assign_subset(evolved_before);
    const auto volume_ghost_vars = ghost_variables(cell_prims, volume_gh_vars);
    DirectionMap<3, gsl::span<const double>> ghost_spans{};
    for (const auto& [direction, data] : ghost_data_in_directions) {
      ghost_spans[direction] = gsl::make_span(data.data(), data.size());
    }
    ::fd::device::pack_with_ghost_zones(
        host_padded, 0,
        gsl::make_span(volume_ghost_vars.data(), volume_ghost_vars.size()),
        ghost_spans, subcell_mesh.extents());
    Kokkos::deep_copy(exec, padded, host_padded);
    exec.fence();
  }
  // As in FD patches, the hydro ghost variables of the cells of the elements
  // are read from their own batch, not from the interior of the padded view,
  // which is overwritten here to check that. The filter reads the GH variables
  // from the interior of the padded view.
  constexpr size_t number_of_hydro_ghost_variables =
      Variables<Tags::primitive_grmhd_reconstruction_tags>::
          number_of_independent_components;
  const auto hydro_ghost_variables =
      TestHelpers::fd::device::split(
          exec,
          Kokkos::subview(
              padded, Kokkos::ALL(),
              Kokkos::make_pair(0_st, number_of_hydro_ghost_variables),
              Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
          subcell_mesh.extents())
          .interior;
  {
    const auto interior_cells = [&](const size_t d) {
      return Kokkos::make_pair(ghost_width,
                               ghost_width + subcell_mesh.extents(d));
    };
    Kokkos::deep_copy(
        exec,
        Kokkos::subview(
            padded, Kokkos::ALL(),
            Kokkos::make_pair(0_st, number_of_hydro_ghost_variables),
            interior_cells(2), interior_cells(1), interior_cells(0)),
        1.0e10);
  }
  using inverse_jacobian_tag =
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>;
  Variables<tmpl::list<
      inverse_jacobian_tag, domain::Tags::Coordinates<3, Frame::Grid>,
      subcell::TimeDerivativeOnDevice<
          ValenciaDivClean::BoundaryCorrections::Hll>::MeshVelocity>>
      cell_geometry_vars{number_of_cells};
  get<inverse_jacobian_tag>(cell_geometry_vars) = db::get<
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>(
      box);
  get<domain::Tags::Coordinates<3, Frame::Grid>>(cell_geometry_vars) =
      db::get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Grid>>(box);
  for (size_t i = 0; i < 3; ++i) {
    get<subcell::TimeDerivativeOnDevice<
        ValenciaDivClean::BoundaryCorrections::Hll>::MeshVelocity>(
        cell_geometry_vars)
        .get(i) = cell_coords.get(i) * expansion_velocity.value_or(0.0);
  }
  const auto cell_geometry_batch =
      to_device_batch<typename decltype(cell_geometry_vars)::tags_list>(
          exec, cell_geometry_vars);

  std::vector<ElementMap<3, Frame::Grid>> element_maps{};
  element_maps.emplace_back(element_id, element_map.block_map().get_clone());
  const auto logical_to_grid =
      fd_patch::logical_to_grid(exec, element_maps, dg_mesh);
  const auto grid_to_inertial = evolution::patches::affine_grid_to_inertial(
      *grid_to_inertial_map, time, {}, logical_to_grid.corners);
  BatchedVariables<fd_patch::inertial_face_tags> face_geometry{
      exec, logical_to_grid.faces.number_of_grid_points(), 3};
  BatchedVariables<fd_patch::inertial_cell_tags> cell_geometry{
      exec, number_of_cells, 1};
  const Kokkos::View<evolution::patches::AffineGridToInertial*> maps{"maps", 3};
  Kokkos::deep_copy(exec, maps, grid_to_inertial);
  fd_patch::inertial_geometry_on_device(exec, make_not_null(&face_geometry),
                                        make_not_null(&cell_geometry),
                                        logical_to_grid, maps);
  const Kokkos::View<double*> div_mesh_velocity{"div_mesh_velocity", 1};
  BatchedVariables<tmpl::list<fd_patch::Tags::CellDivMeshVelocity>>
      cell_div_mesh_velocity{exec, number_of_cells, 1};
  if (expansion_velocity.has_value()) {
    // The same mesh velocity on the faces as on the host
    BatchedVariables<fd_patch::inertial_face_tags, Kokkos::HostSpace>
        host_face_geometry{face_geometry.number_of_grid_points(), 3};
    deep_copy(exec, make_not_null(&host_face_geometry), face_geometry);
    BatchedVariables<fd_patch::logical_to_grid_face_tags, Kokkos::HostSpace>
        host_logical_to_grid{face_geometry.number_of_grid_points(), 3};
    deep_copy(exec, make_not_null(&host_logical_to_grid),
              logical_to_grid.faces);
    exec.fence();
    for (size_t face_set = 0; face_set < 3; ++face_set) {
      Variables<fd_patch::inertial_face_tags> faces{};
      set_data_ref_to_element(make_not_null(&faces), host_face_geometry,
                              face_set);
      Variables<fd_patch::logical_to_grid_face_tags> grid_faces{};
      set_data_ref_to_element(make_not_null(&grid_faces), host_logical_to_grid,
                              face_set);
      for (size_t i = 0; i < 3; ++i) {
        get<fd_patch::Tags::FaceMeshVelocity>(faces).get(i) =
            get<fd_patch::Tags::FaceGridCoordinates>(grid_faces).get(i) *
            *expansion_velocity;
      }
    }
    deep_copy(exec, make_not_null(&face_geometry), host_face_geometry);
    Kokkos::deep_copy(exec, div_mesh_velocity, 3.0 * *expansion_velocity);
    // At the cells, projected from the DG grid as on the host
    const BatchedVariables<tmpl::list<fd_patch::Tags::CellDivMeshVelocity>,
                           Kokkos::HostSpace>
        host_cell_div{number_of_cells, 1};
    const DataVector projected_div = evolution::dg::subcell::fd::project(
        get(*div_dg_mesh_velocity), dg_mesh, subcell_mesh.extents(),
        Spectral::Parity::Even);
    for (size_t i = 0; i < number_of_cells; ++i) {
      get(get<::Tags::BatchView<fd_patch::Tags::CellDivMeshVelocity,
                                Kokkos::HostSpace>>(host_cell_div))(i, 0) =
          projected_div[i];
    }
    deep_copy(exec, make_not_null(&cell_div_mesh_velocity), host_cell_div);
  }

  const Kokkos::View<ConstraintDamping::GaussianSum<3>*> gaussian_sums{
      "gaussian sums", 3};
  const auto host_gaussian_sums = Kokkos::create_mirror_view(gaussian_sums);
  for (size_t i = 0; i < 3; ++i) {
    host_gaussian_sums(i) =
        gsl::at(damping_functions, i)->as_gaussian_sum(time, functions_of_time);
  }
  Kokkos::deep_copy(exec, gaussian_sums, host_gaussian_sums);

  using Hll = ValenciaDivClean::BoundaryCorrections::Hll;
  using DeviceTimeDerivative = subcell::TimeDerivativeOnDevice<Hll>;
  const auto fixer = fix_to_atmosphere.device_view();
  DeviceTimeDerivative time_derivative{
      exec,
      1,
      dg_mesh,
      subcell::hydro_reconstructor_on_device<TheSystem>(recons),
      subcell::spacetime_reconstruction_degree<TheSystem>(recons),
      spacetime_dissipation};
  BatchedVariables<subcell::dt_tags_on_device> dt_vars{exec, number_of_cells,
                                                       1};
  DeferredPointwiseStatus<ValenciaDivClean::fd::face_atmosphere_snapshot_tags>
      face_status{exec};
  face_status.reset(exec);
  const auto eos = soln.equation_of_state().promote_to_3d_eos();
  std::optional<DeviceTimeDerivative::Batch<DeviceTimeDerivative::MeshVelocity>>
      mesh_velocity{};
  if (expansion_velocity.has_value()) {
    mesh_velocity = get<::Tags::BatchView<DeviceTimeDerivative::MeshVelocity>>(
        cell_geometry_batch);
  }
  time_derivative.apply(
      exec, make_not_null(&dt_vars), make_not_null(&evolved),
      make_not_null(&face_status), padded, hydro_ghost_variables, primitives,
      get<::Tags::BatchView<inverse_jacobian_tag>>(cell_geometry_batch),
      get<::Tags::BatchView<domain::Tags::Coordinates<3, Frame::Grid>>>(
          cell_geometry_batch),
      mesh_velocity, face_geometry, cell_geometry,
      div_per_cell
          ? ValenciaDivClean::subcell::DivergenceOfMeshVelocity{get<
                ::Tags::BatchView<fd_patch::Tags::CellDivMeshVelocity>>(
                cell_div_mesh_velocity)}
          : ValenciaDivClean::subcell::DivergenceOfMeshVelocity{Kokkos::View<
                const double*>{div_mesh_velocity}},
      gaussian_sums, Hll{1.0e-30, 1.0e-8}, *eos,
      subcell::fix_reconstructed_states_to_atmosphere<TheSystem>(recons)
          ? &fixer
          : nullptr,
      1.0, fd_patch::PackagedDataFromDg<3>{}, Kokkos::View<const double*>{});
  face_status.copy_to_host(exec);
  exec.fence();
  CHECK_FALSE(face_status.failed());

  const auto dt_on_device = to_host(exec, dt_vars);
  const auto filtered_on_device = to_host(exec, evolved);
  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);
  tmpl::for_each<evolved_tags>([&](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CAPTURE(pretty_type::short_name<tag>());
    CHECK_ITERABLE_CUSTOM_APPROX(get<::Tags::dt<tag>>(dt_on_device),
                                 get<::Tags::dt<tag>>(expected_dt_vars),
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(get<tag>(filtered_on_device),
                                 get<tag>(expected_filtered_vars),
                                 custom_approx);
  });
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Subcell.TimeDerivative."
    "Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  using ::fd::reconstruction::FallbackReconstructorType;
  using ::VariableFixing::FixReconstructedStateToAtmosphere;
  const fd::MonotonisedCentralPrim<TheSystem> monotonised_central{
      FixReconstructedStateToAtmosphere::Never, true};
  const fd::PositivityPreservingAdaptiveOrderPrim<TheSystem> adaptive_order{
      3.8,
      std::nullopt,
      std::nullopt,
      FallbackReconstructorType::MonotonisedCentral,
      FixReconstructedStateToAtmosphere::Never,
      true};
  for (const std::optional<double> expansion_velocity :
       {std::optional<double>{}, std::optional<double>{0.5}}) {
    test(make_not_null(&generator), monotonised_central, expansion_velocity);
    test(make_not_null(&generator), adaptive_order, expansion_velocity);
  }
  // A divergence of the mesh velocity that varies in the element, e.g. for
  // grid to inertial maps that aren't affine
  test(make_not_null(&generator), monotonised_central, 0.5, true);
  // Fixing the reconstructed states to the atmosphere, as the BNS runs do, and
  // reconstructing T
  const fd::MonotonisedCentralPrim<TheSystem> fixed_monotonised_central{
      FixReconstructedStateToAtmosphere::OnFdOnly, false};
  const fd::PositivityPreservingAdaptiveOrderPrim<TheSystem>
      fixed_adaptive_order{3.8,
                           std::nullopt,
                           std::nullopt,
                           FallbackReconstructorType::MonotonisedCentral,
                           FixReconstructedStateToAtmosphere::OnFdOnly,
                           false};
  test(make_not_null(&generator), fixed_monotonised_central, {0.5});
  test(make_not_null(&generator), fixed_adaptive_order, {});
}
}  // namespace grmhd::GhValenciaDivClean
