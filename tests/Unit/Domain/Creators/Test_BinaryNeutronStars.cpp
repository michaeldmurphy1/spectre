// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/BinaryNeutronStars.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Options/Context.hpp"
#include "Utilities/GetOutput.hpp"

namespace {

// Reasonable parameters satisfying all BinaryNeutronStars constraints
constexpr double center_A = 3.0;
constexpr double center_B = -4.0;
constexpr double inner_radius = 1.5;
constexpr double wedge_outer_radius = 7.0;
constexpr double cylinder_outer_radius = 15.0;
constexpr double outer_radius = 20.0;
constexpr size_t cube_grid_points = 3;
constexpr size_t cube_refinement = 0;
constexpr size_t deformed_cube_radial_grid_points = 3;
constexpr size_t deformed_cube_radial_refinement = 0;
constexpr size_t cylinder_radial_grid_points = 3;
constexpr size_t cylinder_radial_refinement = 0;
constexpr size_t b2_angular_grid_points = 7;
constexpr size_t spherical_shells_radial_refinement = 0;
constexpr size_t spherical_shells_radial_grid_points = 3;
constexpr size_t spherical_harmonic_l = 6;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_outer_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_xi(), 50);
}

// From someone else's branch:
// Checks that every pair of neighboring blocks agrees on the physical
// (inertial) coordinates of a grid of points on their shared face, taking the
// OrientationMap between the blocks into account. The standard
// test_physical_separation helper skips the ZernikeB2 / Fourier (non-hypercube)
// blocks, so this explicit check validates that the single-block filled
// cylinders and annular sides connect conformingly -- in particular the
// cutting-plane and outer-shell angular seams.
// NOTE: This has issues with non-conforming interfaces, which is exactly where
// I am having issues....
void check_face_conformity(const Domain<3>& domain) {
  INFO("Check face conformity of all neighboring blocks");
  const auto& blocks = domain.blocks();
  const Mesh<2> face_mesh{5_st, Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss};
  for (const auto& block : blocks) {
    for (const auto& [direction, block_neighbors] : block.neighbors()) {
      for (const size_t neighbor_id : block_neighbors.ids()) {
        const auto& neighbor = blocks[neighbor_id];
        const auto& orientation = block_neighbors.orientation(neighbor_id);
        const auto xi = interface_logical_coordinates(face_mesh, direction);
        tnsr::I<DataVector, 3, Frame::BlockLogical> xi_host{};
        tnsr::I<DataVector, 3, Frame::BlockLogical> xi_neighbor{};
        for (size_t d = 0; d < 3; ++d) {
          xi_host[d] = xi[d];
          const auto mapped = orientation(Direction<3>(d, Side::Upper));
          xi_neighbor[mapped.dimension()] = xi[d];
          if ((mapped.side() == Side::Lower) xor (d == direction.dimension())) {
            xi_neighbor[mapped.dimension()] *= -1.0;
          }
        }
        const auto x_self = block.stationary_map()(xi_host);
        const auto x_neighbor = neighbor.stationary_map()(xi_neighbor);
        CAPTURE(block.id());
        CAPTURE(neighbor_id);
        CAPTURE(direction);
        CHECK_ITERABLE_APPROX(x_self, x_neighbor);
      }
    }
  }
}

domain::creators::BinaryNeutronStars make_creator(
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> outer_bc =
        nullptr) {
  return domain::creators::BinaryNeutronStars{
      center_A,
      center_B,
      inner_radius,
      wedge_outer_radius,
      cylinder_outer_radius,
      outer_radius,
      cube_grid_points,
      cube_refinement,
      deformed_cube_radial_grid_points,
      deformed_cube_radial_refinement,
      cylinder_radial_grid_points,
      cylinder_radial_refinement,
      b2_angular_grid_points,
      std::array<size_t, 2>{7_st, 3_st},
      spherical_shells_radial_refinement,
      spherical_shells_radial_grid_points,
      spherical_harmonic_l,
      domain::CoordinateMaps::Distribution::Linear,
      nullptr,
      std::move(outer_bc)};
}

std::vector<std::string> expected_block_names() {
  return {"BLeftCentralCube",
          "BRightCentralCube",
          "ALeftCentralCube",
          "ARightCentralCube",
          "BDeformedCubeCap",
          "BLeftDeformedCubeFront",
          "BLeftDeformedCubeTop",
          "BLeftDeformedCubeBack",
          "BLeftDeformedCubeBottom",
          "BRightDeformedCubeFront",
          "BRightDeformedCubeTop",
          "BRightDeformedCubeBack",
          "BRightDeformedCubeBottom",
          "ALeftDeformedCubeFront",
          "ALeftDeformedCubeTop",
          "ALeftDeformedCubeBack",
          "ALeftDeformedCubeBottom",
          "ARightDeformedCubeFront",
          "ARightDeformedCubeTop",
          "ARightDeformedCubeBack",
          "ARightDeformedCubeBottom",
          "ADeformedCubeCap",
          "BFilledCylinder",
          "BHollowCylinder",
          "AHollowCylinder",
          "AFilledCylinder",
          "SphericalShell"};
}

void test_block_names_and_groups() {
  INFO("Test block names and groups");
  const auto creator = make_creator();
  const auto names = creator.block_names();
  CHECK(names == expected_block_names());
  REQUIRE(names.size() == 27);

  const auto groups = creator.block_groups();
  // CenterCubes: 4 blocks
  REQUIRE(groups.count("CenterCubes") == 1);
  CHECK(groups.at("CenterCubes") ==
        std::unordered_set<std::string>{"BLeftCentralCube", "BRightCentralCube",
                                        "ALeftCentralCube",
                                        "ARightCentralCube"});
  // DeformedCubes: 18 blocks (1 cap + 8 sides on each star = 2 caps + 16
  // sides)
  REQUIRE(groups.count("DeformedCubes") == 1);
  CHECK(groups.at("DeformedCubes").size() == 18);
  // Cylinders: 4 blocks
  REQUIRE(groups.count("Cylinders") == 1);
  CHECK(groups.at("Cylinders") ==
        std::unordered_set<std::string>{"BFilledCylinder", "BHollowCylinder",
                                        "AHollowCylinder", "AFilledCylinder"});
  // SphericalShells: 1 block
  REQUIRE(groups.count("SphericalShells") == 1);
  CHECK(groups.at("SphericalShells") ==
        std::unordered_set<std::string>{"SphericalShell"});
}

void test_domain_structure() {
  INFO("Test domain structure");
  const auto creator = make_creator();
  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  REQUIRE(blocks.size() == 27);

  // No excision spheres: neutron stars fill the inner volume
  CHECK(domain.excision_spheres().empty());

  // Only external boundary is the outer face of the spherical shell (block 26)
  // All other blocks should be fully connected to neighbors.
  for (size_t i = 0; i < blocks.size(); ++i) {
    CAPTURE(i);
    CAPTURE(blocks[i].name());
    const auto& ext = blocks[i].external_boundaries();
    if (i == 26) {
      // SphericalShell: upper_xi is the outer boundary
      CHECK(ext.size() == 1);
      CHECK(ext.count(Direction<3>::upper_xi()) == 1);
    } else {
      // All other blocks: fully internally connected
      CHECK(ext.empty());
    }
  }
  check_face_conformity(domain);
}

void test_domain_creator_checks() {
  INFO("Test domain creator consistency checks");
  // Without boundary conditions
  {
    const auto creator = make_creator();
    TestHelpers::domain::creators::test_domain_creator(creator, false);
  }
  // With outer boundary condition
  {
    const auto creator = make_creator(create_outer_boundary_condition());
    TestHelpers::domain::creators::test_domain_creator(creator, true);
  }
}

void test_option_parsing() {
  INFO("Test option parsing");
  const std::string option_string =
      "BinaryNeutronStars:\n"
      "  CenterA: " +
      get_output(center_A) + "\n  CenterB: " + get_output(center_B) +
      "\n  InnerRadius: " + get_output(inner_radius) +
      "\n  WedgeOuterRadius: " + get_output(wedge_outer_radius) +
      "\n  CylinderOuterRadius: " + get_output(cylinder_outer_radius) +
      "\n  OuterRadius: " + get_output(outer_radius) +
      "\n  CubeInitialGridPoints: " + get_output(cube_grid_points) +
      "\n  CubeInitialRefinement: " + get_output(cube_refinement) +
      "\n  DeformedCubeInitialRadialGridPoints: " +
      get_output(deformed_cube_radial_grid_points) +
      "\n  DeformedCubeInitialRadialRefinement: " +
      get_output(deformed_cube_radial_refinement) +
      "\n  CylinderInitialRadialGridPoints: " +
      get_output(cylinder_radial_grid_points) +
      "\n  CylinderInitialRadialRefinement: " +
      get_output(cylinder_radial_refinement) +
      "\n  B2InitialAngularGridPoints: " + get_output(b2_angular_grid_points) +
      "\n  HollowCylinderInitialAngularGridPoints: [7, 3]"
      "\n  SphericalShellsInitialRadialRefinement: " +
      get_output(spherical_shells_radial_refinement) +
      "\n  SphericalShellsInitialRadialGridPoints: " +
      get_output(spherical_shells_radial_grid_points) +
      "\n  InitialSphericalHarmonicL: " + get_output(spherical_harmonic_l) +
      "\n  SphericalShellsRadialDistribution: Linear"
      "\n  TimeDependence: None\n";
  const auto creator = make_creator();
  TestHelpers::domain::creators::test_creation(option_string, creator, false);
}

void test_parse_errors() {
  INFO("Test parse errors");

  // center_A must be positive
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          -1.0, center_B, inner_radius, wedge_outer_radius,
          cylinder_outer_radius, outer_radius, cube_grid_points,
          cube_refinement, deformed_cube_radial_grid_points,
          deformed_cube_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "x-coordinate of the input CenterA is expected to be positive"));

  // center_B must be negative
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, 1.0, inner_radius, wedge_outer_radius,
          cylinder_outer_radius, outer_radius, cube_grid_points,
          cube_refinement, deformed_cube_radial_grid_points,
          deformed_cube_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "x-coordinate of the input CenterB is expected to be negative"));

  // inner_radius must be positive
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, center_B, 0.0, wedge_outer_radius, cylinder_outer_radius,
          outer_radius, cube_grid_points, cube_refinement,
          deformed_cube_radial_grid_points, deformed_cube_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("InnerRadius must be positive"));

  // |center_A| <= |center_B|
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          8.0, center_B, inner_radius, wedge_outer_radius,
          cylinder_outer_radius, outer_radius, cube_grid_points,
          cube_refinement, deformed_cube_radial_grid_points,
          deformed_cube_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("We expect |x_A| <= |x_B|"));

  // wedge_outer_radius must be positive
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, center_B, inner_radius, -1.0, cylinder_outer_radius,
          outer_radius, cube_grid_points, cube_refinement,
          deformed_cube_radial_grid_points, deformed_cube_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "WedgeOuterRadius is expected to be positive"));

  // cylinder_outer_radius must exceed wedge_outer_radius
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, center_B, inner_radius, wedge_outer_radius, 4.0,
          outer_radius, cube_grid_points, cube_refinement,
          deformed_cube_radial_grid_points, deformed_cube_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "CylinderOuterRadius is expected to be greater than "
          "WedgeOuterRadius"));

  // outer_radius must exceed cylinder_outer_radius
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, center_B, inner_radius, wedge_outer_radius,
          cylinder_outer_radius, 8.0, cube_grid_points, cube_refinement,
          deformed_cube_radial_grid_points, deformed_cube_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "OuterRadius is expected to be greater than CylinderOuterRadius"));

  // inner_radius must be smaller than the distance to each star center
  // (center_A = 3.0, so inner_radius = 3.0 triggers the check)
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, center_B, 3.0, wedge_outer_radius, cylinder_outer_radius,
          outer_radius, cube_grid_points, cube_refinement,
          deformed_cube_radial_grid_points, deformed_cube_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "InnerRadius must be smaller than the distance from the origin to "
          "each star center"));

  // FlatOffsetWedge validity: center^2 + 2*inner_radius^2 < wedge_outer^2
  // center_B = -4.0, inner_radius = 1.5, wedge_outer = 7.0:
  //   4^2 + 2*1.5^2 = 16 + 4.5 = 20.5 < 49  (valid)
  // Use inner_radius = 3.0 which gives 4^2 + 2*3^2 = 16+18 = 34 < 49 (valid),
  // but center_A = 3.0 triggers constraint 1 first for inner_radius = 3.0.
  // Use inner_radius = 2.5 and a small wedge_outer_radius to trigger: center_B
  // (-4)^2 + 2*(2.5)^2 = 16 + 12.5 = 28.5. Need wedge_outer^2 <= 28.5, i.e.
  // wedge_outer <= 5.34. Use wedge_outer = 5.0: but then center_A=3 < 5 ok,
  // and cylinder_outer > wedge_outer so use cylinder_outer=8.
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, center_B, 2.5, 5.0, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_refinement, deformed_cube_radial_grid_points,
          deformed_cube_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "FlatOffsetWedge validity condition violated"));

  // Cannot use periodic outer boundary condition
  CHECK_THROWS_WITH(
      domain::creators::BinaryNeutronStars(
          center_A, center_B, inner_radius, wedge_outer_radius,
          cylinder_outer_radius, outer_radius, cube_grid_points,
          cube_refinement, deformed_cube_radial_grid_points,
          deformed_cube_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with a binary domain"));
}
}  // namespace

// this is fake? // [[TimeOut, 60]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryNeutronStars", "[Domain][Unit]") {
  test_block_names_and_groups();
  test_domain_structure();
  test_domain_creator_checks();
  test_option_parsing();
  test_parse_errors();
}
