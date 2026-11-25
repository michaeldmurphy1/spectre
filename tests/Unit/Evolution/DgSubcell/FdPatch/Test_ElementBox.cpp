// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/FdPatch/ElementBox.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SubcellAndNonconformingEqualRateRegions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarA : db::SimpleTag {
  using type = Scalar<DataVector>;
};
using variables_tag = ::Tags::Variables<tmpl::list<ScalarA>>;

struct System {
  using variables_tag = ::variables_tag;
};

// As the executables with subcell
using equal_rate_regions = tmpl::list<
    evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<3>>;
using RegionsTag =
    evolution::dg::Tags::ConcreteEqualRateRegions<3, equal_rate_regions>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      tmpl::list<::Tags::ConcreteTimeStepper<TimeStepper>, RegionsTag>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct Metavariables {
  using system = System;
  using equal_rate_regions = ::equal_rate_regions;
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<>;
  };
};

using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical, Frame::Grid, 3>>
block_map() {
  return domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
      Affine3D{Affine{-1.0, 1.0, 0.0, 2.0}, Affine{-1.0, 1.0, -1.0, 0.5},
               Affine{-1.0, 1.0, 1.0, 4.0}});
}

std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
grid_to_inertial_map() {
  // A rotation about the z axis, so that the Jacobians to the grid and the
  // inertial frame differ
  return domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
      domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
          domain::CoordinateMaps::TimeDependent::Rotation<2>,
          domain::CoordinateMaps::Identity<1>>{
          domain::CoordinateMaps::TimeDependent::Rotation<2>{"Rotation"},
          domain::CoordinateMaps::Identity<1>{}});
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
functions_of_time() {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  // The rotation angle
  result["Rotation"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, std::array<DataVector, 3>{{{0.3}, {0.1}, {0.02}}},
          std::numeric_limits<double>::infinity());
  return result;
}

// The DataBox of an element of the `DgElementArray` on the subcell grid, with
// the geometry that `evolution::dg::subcell::Actions::SetSubcellGrid` and the
// executables add for observations
auto host_element_box(const Element<3>& element, const Mesh<3>& dg_mesh,
                      const double time) {
  return db::create<
      db::AddSimpleTags<domain::Tags::Mesh<3>, domain::Tags::Element<3>,
                        domain::Tags::ElementMap<3, Frame::Grid>,
                        domain::CoordinateMaps::Tags::CoordinateMap<
                            3, Frame::Grid, Frame::Inertial>,
                        ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize,
                        evolution::dg::subcell::Tags::ActiveGrid>,
      db::AddComputeTags<
          // The DG grid
          domain::Tags::LogicalCoordinates<3>,
          domain::Tags::MappedCoordinates<
              domain::Tags::ElementMap<3, Frame::Grid>,
              domain::Tags::Coordinates<3, Frame::ElementLogical>>,
          domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
              domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                          Frame::Inertial>>,
          domain::Tags::InertialFromGridCoordinatesCompute<3>,
          domain::Tags::InertialMeshVelocityCompute<3>,
          domain::Tags::MinimumGridSpacingCompute<3, Frame::Inertial>,
          // The subcell grid
          evolution::dg::subcell::Tags::MeshCompute<3>,
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>,
          domain::Tags::MappedCoordinates<
              domain::Tags::ElementMap<3, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<3,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          evolution::dg::subcell::Tags::InertialCoordinatesCompute<
              domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                          Frame::Inertial>>,
          evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGridCompute<
              domain::Tags::ElementMap<3, Frame::Grid>, 3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToGridCompute<3>,
          evolution::dg::subcell::fd::Tags::
              InverseJacobianLogicalToInertialCompute<
                  domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                              Frame::Inertial>,
                  3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToInertialCompute<
                  domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                              Frame::Inertial>,
                  3>,
          // The geometry for observations on the subcell grid
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              3, Frame::ElementLogical>,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<3,
                                                                   Frame::Grid>,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              3, Frame::Inertial>,
          evolution::dg::subcell::Tags::ObserverInverseJacobianCompute<
              3, Frame::ElementLogical, Frame::Inertial>,
          evolution::dg::subcell::Tags::
              ObserverJacobianAndDetInvJacobianCompute<3, Frame::ElementLogical,
                                                       Frame::Inertial>>>(
      dg_mesh, element, ElementMap<3, Frame::Grid>{element.id(), block_map()},
      grid_to_inertial_map(), time, functions_of_time(),
      evolution::dg::subcell::ActiveGrid::Subcell);
}

// The DataBox of an element of a patch has the geometry of the element of the
// `DgElementArray` on the subcell grid, given the geometry at the cell centers
// that the patch computes
void test_element_box() {
  using component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(3), RegionsTag::type{}},
      {functions_of_time()}};
  ActionTesting::emplace_array_component<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  auto& cache = ActionTesting::cache<component>(runner, 0);

  const double time = 1.5;
  const ElementId<3> element_id{
      0, std::array{SegmentId{1, 1}, SegmentId{1, 0}, SegmentId{0, 0}}};
  const Element<3> element{element_id, {}, make_array<3>(domain::Topology::I1)};
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto host_box = host_element_box(element, dg_mesh, time);

  // The cell geometry as the patch computes it
  namespace subcell = evolution::dg::subcell;
  const size_t number_of_cells =
      db::get<subcell::Tags::Mesh<3>>(host_box).number_of_grid_points();
  Variables<evolution::dg::subcell::fd_patch::ElementBox_detail::
                cell_geometry_tags<3>>
      cell_geometry{number_of_cells};
  get<subcell::Tags::Coordinates<3, Frame::Grid>>(cell_geometry) =
      db::get<subcell::Tags::Coordinates<3, Frame::Grid>>(host_box);
  get<subcell::fd::Tags::InverseJacobianLogicalToGrid<3>>(cell_geometry) =
      db::get<subcell::fd::Tags::InverseJacobianLogicalToGrid<3>>(host_box);
  get<subcell::fd::Tags::DetInverseJacobianLogicalToGrid>(cell_geometry) =
      db::get<subcell::fd::Tags::DetInverseJacobianLogicalToGrid>(host_box);
  Variables<evolution::dg::subcell::fd_patch::ElementBox_detail::
                inertial_cell_geometry_tags<3>>
      inertial_cell_geometry{number_of_cells};
  get<subcell::Tags::Coordinates<3, Frame::Inertial>>(inertial_cell_geometry) =
      db::get<subcell::Tags::Coordinates<3, Frame::Inertial>>(host_box);
  get<subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>(
      inertial_cell_geometry) =
      db::get<subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>(host_box);
  get<subcell::fd::Tags::DetInverseJacobianLogicalToInertial>(
      inertial_cell_geometry) =
      db::get<subcell::fd::Tags::DetInverseJacobianLogicalToInertial>(host_box);

  Variables<tmpl::list<ScalarA>> variables{number_of_cells, 2.0};
  const Slab slab{1.0, 2.0};
  const TimeStepId time_step_id{true, 1, slab.start()};
  const auto box = evolution::dg::subcell::fd_patch::make_element_box<
      Metavariables, tmpl::list<variables_tag>, tmpl::list<>>(
      cache, element, dg_mesh,
      ElementMap<3, Frame::Grid>{element_id, block_map()},
      grid_to_inertial_map(), time, time_step_id, slab.duration(), 0,
      std::move(cell_geometry), std::move(inertial_cell_geometry),
      tuples::TaggedTuple<variables_tag>{variables});

  CHECK(db::get<subcell::Tags::ActiveGrid>(box) ==
        subcell::ActiveGrid::Subcell);
  CHECK(db::get<subcell::Tags::Mesh<3>>(box) ==
        db::get<subcell::Tags::Mesh<3>>(host_box));
  CHECK(db::get<::Tags::Time>(box) == time);
  CHECK(db::get<variables_tag>(box) == variables);
  CHECK_ITERABLE_APPROX(
      (db::get<Events::Tags::ObserverCoordinates<3, Frame::ElementLogical>>(
          box)),
      (db::get<Events::Tags::ObserverCoordinates<3, Frame::ElementLogical>>(
          host_box)));
  CHECK_ITERABLE_APPROX(
      (db::get<Events::Tags::ObserverCoordinates<3, Frame::Grid>>(box)),
      (db::get<Events::Tags::ObserverCoordinates<3, Frame::Grid>>(host_box)));
  CHECK_ITERABLE_APPROX(
      (db::get<Events::Tags::ObserverCoordinates<3, Frame::Inertial>>(box)),
      (db::get<Events::Tags::ObserverCoordinates<3, Frame::Inertial>>(
          host_box)));
  using ObserverInverseJacobian =
      Events::Tags::ObserverInverseJacobian<3, Frame::ElementLogical,
                                            Frame::Inertial>;
  using ObserverJacobian =
      Events::Tags::ObserverJacobian<3, Frame::ElementLogical, Frame::Inertial>;
  using ObserverDetInvJacobian =
      Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                           Frame::Inertial>;
  CHECK_ITERABLE_APPROX(db::get<ObserverInverseJacobian>(box),
                        db::get<ObserverInverseJacobian>(host_box));
  CHECK_ITERABLE_APPROX(db::get<ObserverJacobian>(box),
                        db::get<ObserverJacobian>(host_box));
  CHECK_ITERABLE_APPROX(db::get<ObserverDetInvJacobian>(box),
                        db::get<ObserverDetInvJacobian>(host_box));
  // The DG grid, e.g. for step choosers
  CHECK(db::get<domain::Tags::MinimumGridSpacing<3, Frame::Inertial>>(box) ==
        approx(db::get<domain::Tags::MinimumGridSpacing<3, Frame::Inertial>>(
            host_box)));
  CHECK_ITERABLE_APPROX(
      (db::get<domain::Tags::Coordinates<3, Frame::Inertial>>(box)),
      (db::get<domain::Tags::Coordinates<3, Frame::Inertial>>(host_box)));
  REQUIRE(db::get<domain::Tags::MeshVelocity<3>>(box).has_value());
  CHECK_ITERABLE_APPROX(
      db::get<domain::Tags::MeshVelocity<3>>(box).value(),
      db::get<domain::Tags::MeshVelocity<3>>(host_box).value());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.ElementBox",
                  "[Evolution][Unit]") {
  register_classes_with_charm<
      TimeSteppers::AdamsBashforth,
      domain::FunctionsOfTime::PiecewisePolynomial<2>>();
  test_element_box();
}
