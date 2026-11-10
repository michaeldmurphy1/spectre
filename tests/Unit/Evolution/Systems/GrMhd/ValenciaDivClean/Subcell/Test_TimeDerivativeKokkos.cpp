// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/GhostZoneInverseJacobian.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/GhostZoneInverseJacobian.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/TagsDomain.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
namespace fd_patch = evolution::dg::subcell::fd_patch;

// Needed by `BoundaryConditionGhostData` in the host time derivative
struct DummyAnalyticSolutionTag : db::SimpleTag {
  using type = Solutions::SmoothFlow;
};

struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = false;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<BoundaryConditions::BoundaryCondition,
                   BoundaryConditions::standard_boundary_conditions>,
        tmpl::pair<evolution::BoundaryCorrection,
                   BoundaryCorrections::standard_boundary_corrections>>;
  };
};

template <typename Tags, typename HostVars>
BatchedVariables<Tags> to_device_batch(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::vector<HostVars>& host_vars) {
  BatchedVariables<Tags, Kokkos::HostSpace> host_batch{
      host_vars.front().number_of_grid_points(), host_vars.size()};
  for (size_t e = 0; e < host_vars.size(); ++e) {
    Variables<Tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    element.assign_subset(host_vars[e]);
  }
  BatchedVariables<Tags> result{exec, host_batch.number_of_grid_points(),
                                host_batch.number_of_elements()};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

// The FD time derivative of one element of Bondi-Michel on the device is the
// same as on the host, as in the test of `subcell::TimeDerivative`.
template <typename Correction>
void test(const Correction& correction) {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const Affine affine_map{-1.0, 1.0, 1.0, 15.0};
  const Affine3D coordinate_map{affine_map, affine_map, affine_map};
  std::vector<Block<3>> blocks;
  blocks.emplace_back(Block<3>(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          coordinate_map),
      0, {}));
  const auto element = domain::create_initial_element(
      ElementId<3>{0, {SegmentId{3, 2}, SegmentId{3, 2}, SegmentId{3, 2}}},
      blocks,
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});
  const ElementMap<3, Frame::Grid> element_map{
      element.id(),
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          coordinate_map)};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  const fd::MonotonisedCentralPrim recons{false};
  const Solutions::BondiMichel soln{1.0, 5.0, 0.05, 1.4, 2.0};
  const double time = 0.5;
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_cells = subcell_mesh.number_of_grid_points();
  const auto cell_coords =
      (*grid_to_inertial_map)(element_map(logical_coordinates(subcell_mesh)));

  typename evolution::dg::subcell::Tags::GhostZoneInverseJacobian<3>::type
      ghost_zone_inv_jac{};
  evolution::dg::subcell::GhostZoneInverseJacobian<
      3, fd::MonotonisedCentralPrim>::apply(make_not_null(&ghost_zone_inv_jac),
                                            subcell_mesh, element_map, recons);

  Variables<typename System::spacetime_variables_tag::tags_list>
      cell_spacetime_vars{number_of_cells};
  cell_spacetime_vars.assign_subset(
      soln.variables(cell_coords, time,
                     typename System::spacetime_variables_tag::tags_list{}));
  Variables<typename System::primitive_variables_tag::tags_list> cell_prims{
      number_of_cells};
  cell_prims.assign_subset(
      soln.variables(cell_coords, time,
                     typename System::primitive_variables_tag::tags_list{}));

  // The ghost data from the solution in the neighbors
  evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      neighbor_data{};
  DirectionMap<3, DataVector> ghost_data_in_directions{};
  for (const Direction<3>& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    const auto neighbor_prims = soln.variables(
        (*grid_to_inertial_map)(element_map(neighbor_logical_coords)), time,
        typename System::primitive_variables_tag::tags_list{});
    Variables<typename System::primitive_variables_tag::tags_list>
        neighbor_prim_vars{number_of_cells};
    neighbor_prim_vars.assign_subset(neighbor_prims);
    const DataVector neighbor_ghost_vars =
        subcell::PrimitiveGhostVariables::apply(neighbor_prim_vars, 0);
    ghost_data_in_directions[direction] =
        evolution::dg::subcell::slice_data(
            neighbor_ghost_vars, subcell_mesh.extents(),
            recons.ghost_zone_size(), std::unordered_set{direction.opposite()},
            0, {})
            .at(direction.opposite());
    const DirectionalId<3> key{direction,
                               *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        ghost_data_in_directions.at(direction);
  }

  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  Variables<typename variables_tag::tags_list> cell_cons{number_of_cells};
  apply(make_not_null(&cell_cons), ConservativeFromPrimitive{},
        cell_spacetime_vars, cell_prims);

  // Face spacetime variables, as in the test of `subcell::TimeDerivative`
  std::array<typename System::flux_spacetime_variables_tag::type, 3>
      face_spacetime_vars{};
  for (size_t d = 0; d < 3; ++d) {
    auto extents = make_array<3>(subcell_mesh.extents(0));
    gsl::at(extents, d) += 1;
    auto quadrature = make_array<3>(subcell_mesh.quadrature(0));
    gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_mesh{extents, make_array<3>(subcell_mesh.basis(0)),
                            quadrature};
    const auto face_coords =
        (*grid_to_inertial_map)(element_map(logical_coordinates(face_mesh)));
    gsl::at(face_spacetime_vars, d)
        .initialize(face_mesh.number_of_grid_points());
    gsl::at(face_spacetime_vars, d)
        .assign_subset(soln.variables(
            face_coords, time,
            typename System::flux_spacetime_variables_tag::tags_list{}));
  }

  const VariableFixing::FixToAtmosphere<3> fix_to_atmosphere{
      1.0e-30, 1.0e-30, std::nullopt, std::nullopt};
  const auto eos = soln.equation_of_state().promote_to_3d_eos();
  using CellCenteredFluxesTag = evolution::dg::subcell::Tags::CellCenteredFlux<
      typename System::flux_variables, 3>;
  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, evolution::dg::subcell::Tags::Mesh<3>,
          domain::Tags::Mesh<3>, fd::Tags::Reconstructor,
          evolution::Tags::BoundaryCorrection,
          hydro::Tags::GrmhdEquationOfState,
          typename System::spacetime_variables_tag,
          typename System::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::OnSubcellFaces<
              typename System::flux_spacetime_variables_tag, 3>,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
          Tags::ConstraintDampingParameter, evolution::dg::Tags::MortarData<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::Domain<3>,
          domain::Tags::MeshVelocity<3, Frame::Inertial>,
          domain::Tags::DivMeshVelocity,
          evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
          domain::Tags::FunctionsOfTimeInitialize, DummyAnalyticSolutionTag,
          Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars>,
          CellCenteredFluxesTag,
          evolution::dg::subcell::Tags::SubcellOptions<3>,
          evolution::dg::subcell::Tags::ReconstructionOrder<3>,
          evolution::dg::subcell::Tags::GhostZoneInverseJacobian<3>,
          ::Tags::VariableFixer<VariableFixing::FixToAtmosphere<3>>>,
      db::AddComputeTags<
          ::domain::Tags::LogicalCoordinates<3>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              ::domain::Tags::Coordinates<3, Frame::ElementLogical>>,
          ::domain::Tags::InverseJacobianCompute<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              ::domain::Tags::Coordinates<3, Frame::ElementLogical>>,
          ::domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
              ::domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                            Frame::Inertial>>,
          ::domain::Tags::InertialFromGridCoordinatesCompute<3>,
          ::domain::Tags::ElementToInertialInverseJacobian<3>,
          ::domain::Tags::DetInvJacobianCompute<3, Frame::ElementLogical,
                                                Frame::Inertial>,
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
                  3>>>(
      element, subcell_mesh, dg_mesh,
      std::unique_ptr<fd::Reconstructor>{
          std::make_unique<fd::MonotonisedCentralPrim>(recons)},
      std::unique_ptr<evolution::BoundaryCorrection>{
          std::make_unique<Correction>(correction)},
      eos->get_clone(), cell_spacetime_vars, cell_prims,
      Variables<typename dt_variables_tag::tags_list>{number_of_cells},
      cell_cons, face_spacetime_vars, neighbor_data, 1.0,
      evolution::dg::Tags::MortarData<3>::type{},
      ElementMap<3, Frame::Grid>{element.id(),
                                 element_map.block_map().get_clone()},
      grid_to_inertial_map->get_clone(), Domain<3>{},
      std::optional<tnsr::I<DataVector, 3>>{},
      std::optional<Scalar<DataVector>>{},
      typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type{}, time,
      domain::FunctionsOfTimeMap{}, Solutions::SmoothFlow{},
      DummyEvolutionMetaVars{}, typename CellCenteredFluxesTag::type{},
      evolution::dg::subcell::SubcellOptions{
          4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          ::fd::DerivativeOrder::Two, 1, 1, 1},
      typename evolution::dg::subcell::Tags::ReconstructionOrder<3>::type{},
      ghost_zone_inv_jac, fix_to_atmosphere);
  subcell::TimeDerivative::apply(make_not_null(&box));
  const auto& expected_dt_vars = db::get<dt_variables_tag>(box);

  // The same on the device
  const Kokkos::DefaultExecutionSpace exec{};
  const size_t ghost_width = recons.ghost_zone_size();
  const auto padded = ::fd::device::make_volume_view(
      exec, "padded", 1,
      Variables<subcell::PrimitiveGhostVariables::ghost_variables_tag_list>::
          number_of_independent_components,
      subcell_mesh.extents(), ghost_width);
  const auto host_padded = Kokkos::create_mirror_view(padded);
  const DataVector volume_ghost_vars =
      subcell::PrimitiveGhostVariables::apply(cell_prims, 0);
  DirectionMap<3, gsl::span<const double>> ghost_spans{};
  for (const auto& [direction, data] : ghost_data_in_directions) {
    ghost_spans[direction] = gsl::make_span(data.data(), data.size());
  }
  ::fd::device::pack_with_ghost_zones(
      host_padded, 0,
      gsl::make_span(volume_ghost_vars.data(), volume_ghost_vars.size()),
      ghost_spans, subcell_mesh.extents());
  Kokkos::deep_copy(exec, padded, host_padded);

  Variables<subcell::cell_spacetime_tags> cell_spacetime_with_coords{
      number_of_cells};
  cell_spacetime_with_coords.assign_subset(cell_spacetime_vars);
  get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
      cell_spacetime_with_coords) = cell_coords;
  const auto evolved = to_device_batch<subcell::evolved_tags_on_device>(
      exec, std::vector{cell_cons});
  const auto primitives = to_device_batch<hydro::grmhd_tags<DataVector>>(
      exec, std::vector{cell_prims});
  const auto cell_spacetime = to_device_batch<subcell::cell_spacetime_tags>(
      exec, std::vector{cell_spacetime_with_coords});
  const auto face_spacetime = to_device_batch<subcell::face_spacetime_tags>(
      exec,
      std::vector(face_spacetime_vars.begin(), face_spacetime_vars.end()));

  std::vector<ElementMap<3, Frame::Grid>> element_maps{};
  element_maps.emplace_back(element.id(), element_map.block_map().get_clone());
  const auto logical_to_grid =
      fd_patch::logical_to_grid(exec, element_maps, dg_mesh);
  const auto grid_to_inertial = fd_patch::affine_grid_to_inertial(
      *grid_to_inertial_map, time, {}, logical_to_grid.corners);
  BatchedVariables<fd_patch::inertial_face_tags> face_geometry{
      exec, logical_to_grid.faces.number_of_grid_points(), 3};
  BatchedVariables<fd_patch::inertial_cell_tags> cell_geometry{
      exec, number_of_cells, 1};
  fd_patch::inertial_geometry_on_device(exec, make_not_null(&face_geometry),
                                        make_not_null(&cell_geometry),
                                        logical_to_grid, grid_to_inertial);

  subcell::TimeDerivativeOnDevice<Correction> time_derivative{exec, 1, dg_mesh};
  time_derivative.set_face_spacetime_variables(exec, face_spacetime);
  BatchedVariables<subcell::dt_tags_on_device> dt_vars{exec, number_of_cells,
                                                       1};
  const auto fixer = fix_to_atmosphere.device_view();
  const auto status = time_derivative.apply(
      exec, make_not_null(&dt_vars), padded, evolved, primitives,
      cell_spacetime, face_geometry, cell_geometry, grid_to_inertial,
      correction, *eos, &fixer, 1.0, fd_patch::PackagedDataFromDg<3>{},
      Kokkos::View<const double*>{});
  CHECK(status[0].number_of_failures == 0);
  CHECK(status[1].number_of_failures == 0);
  BatchedVariables<subcell::dt_tags_on_device, Kokkos::HostSpace> host_dt{
      number_of_cells, 1};
  deep_copy(exec, make_not_null(&host_dt), dt_vars);
  exec.fence();
  Variables<subcell::dt_tags_on_device> dt_on_device{};
  set_data_ref_to_element(make_not_null(&dt_on_device), host_dt, 0);

  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);
  tmpl::for_each<subcell::evolved_tags_on_device>([&](auto tag_v) {
    using dt_tag = ::Tags::dt<tmpl::type_from<decltype(tag_v)>>;
    CAPTURE(pretty_type::short_name<dt_tag>());
    CHECK_ITERABLE_CUSTOM_APPROX(get<dt_tag>(dt_on_device),
                                 get<dt_tag>(expected_dt_vars), custom_approx);
  });

  // The moving mesh terms vanish for a mesh that doesn't move, which the host
  // doesn't support
  auto not_moving = grid_to_inertial;
  not_moving.is_moving = true;
  BatchedVariables<subcell::dt_tags_on_device> moving_dt_vars{
      exec, number_of_cells, 1};
  time_derivative.apply(exec, make_not_null(&moving_dt_vars), padded, evolved,
                        primitives, cell_spacetime, face_geometry,
                        cell_geometry, not_moving, correction, *eos, &fixer,
                        1.0, fd_patch::PackagedDataFromDg<3>{},
                        Kokkos::View<const double*>{});
  BatchedVariables<subcell::dt_tags_on_device, Kokkos::HostSpace>
      host_moving_dt{number_of_cells, 1};
  deep_copy(exec, make_not_null(&host_moving_dt), moving_dt_vars);
  exec.fence();
  Variables<subcell::dt_tags_on_device> moving_dt_on_device{};
  set_data_ref_to_element(make_not_null(&moving_dt_on_device), host_moving_dt,
                          0);
  CHECK_VARIABLES_APPROX(moving_dt_on_device, dt_on_device);
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.TimeDerivative.Kokkos",
    "[Unit][Evolution]") {
  test(BoundaryCorrections::Rusanov{});
  test(BoundaryCorrections::Hll{1.0e-30, 1.0e-8});
}
}  // namespace grmhd::ValenciaDivClean
