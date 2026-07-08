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
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Pill.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Options/Context.hpp"
#include "Utilities/GetOutput.hpp"

namespace {

// Parameters satisfying Pill constraints
constexpr double leftmost_x = -6.9;
constexpr double center_B = -4.0;
constexpr double center_A = 3.1;
constexpr double rightmost_x = 5.8;
constexpr double wedge_inner_radius = 2.9;
constexpr double wedge_outer_radius = 7.0;
constexpr double cylinder_outer_radius = 15.0;
constexpr double outer_radius = 20.0;
constexpr std::array<size_t, 4> cube_grid_points{3, 4, 5, 3};
constexpr std::array<size_t, 4> cube_x_refinement{1, 0, 1, 2};
constexpr std::array<size_t, 4> cube_yz_refinement{0, 1, 0, 1};
constexpr size_t wedge_prism_radial_grid_points = 3;
constexpr size_t wedge_prism_radial_refinement = 1;
constexpr size_t cylinder_radial_grid_points = 3;
constexpr size_t cylinder_radial_refinement = 1;
constexpr size_t b2_angular_grid_points = 7;
constexpr size_t spherical_shells_radial_refinement = 2;
constexpr size_t spherical_shells_radial_grid_points = 3;
constexpr size_t spherical_harmonic_l = 6;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_outer_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_xi(), 50);
}

domain::creators::Pill make_creator(
    bool bulge = false,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> outer_bc =
        nullptr) {
  return domain::creators::Pill{center_A,
                                center_B,
                                leftmost_x,
                                rightmost_x,
                                wedge_inner_radius,
                                wedge_outer_radius,
                                cylinder_outer_radius,
                                outer_radius,
                                cube_grid_points,
                                cube_x_refinement,
                                cube_yz_refinement,
                                wedge_prism_radial_grid_points,
                                wedge_prism_radial_refinement,
                                cylinder_radial_grid_points,
                                cylinder_radial_refinement,
                                b2_angular_grid_points,
                                std::array<size_t, 2>{7_st, 3_st},
                                spherical_shells_radial_refinement,
                                spherical_shells_radial_grid_points,
                                spherical_harmonic_l,
                                domain::CoordinateMaps::Distribution::Linear,
                                bulge,
                                std::nullopt,
                                std::move(outer_bc)};
}

std::vector<std::string> expected_block_names() {
  return {// BLeft cubed-cylinder (blocks 0-4)
          "BLeftCubedCylinderCenter", "BLeftCubedCylinderFront",
          "BLeftCubedCylinderTop", "BLeftCubedCylinderBack",
          "BLeftCubedCylinderBottom",
          // BRight cubed-cylinder (blocks 5-9)
          "BRightCubedCylinderCenter", "BRightCubedCylinderFront",
          "BRightCubedCylinderTop", "BRightCubedCylinderBack",
          "BRightCubedCylinderBottom",
          // ALeft cubed-cylinder (blocks 10-14)
          "ALeftCubedCylinderCenter", "ALeftCubedCylinderFront",
          "ALeftCubedCylinderTop", "ALeftCubedCylinderBack",
          "ALeftCubedCylinderBottom",
          // ARight cubed-cylinder (blocks 15-19)
          "ARightCubedCylinderCenter", "ARightCubedCylinderFront",
          "ARightCubedCylinderTop", "ARightCubedCylinderBack",
          "ARightCubedCylinderBottom",
          // Cylinders (blocks 20-25)
          "BFilledCylinder", "BLeftHollowCylinder", "BRightHollowCylinder",
          "ALeftHollowCylinder", "ARightHollowCylinder", "AFilledCylinder",
          // Spherical shell (block 26)
          "SphericalShell"};
}

void test_block_names_and_groups() {
  INFO("Test block names and groups");
  // Block names and groups are the same for bulge=true and bulge=false
  const auto creator = make_creator(false);
  const auto names = creator.block_names();
  CHECK(names == expected_block_names());
  REQUIRE(names.size() == 27);

  const auto groups = creator.block_groups();
  // CubedCylinders: 20 blocks (4 groups of 5)
  REQUIRE(groups.count("CubedCylinders") == 1);
  CHECK(groups.at("CubedCylinders").size() == 20);
  CHECK(groups.at("CubedCylinders").count("BLeftCubedCylinderCenter") == 1);
  CHECK(groups.at("CubedCylinders").count("ARightCubedCylinderBottom") == 1);
  // Cylinders: 6 blocks (2 filled + 4 hollow)
  REQUIRE(groups.count("Cylinders") == 1);
  CHECK(groups.at("Cylinders") ==
        std::unordered_set<std::string>{
            "BFilledCylinder", "BLeftHollowCylinder", "BRightHollowCylinder",
            "ALeftHollowCylinder", "ARightHollowCylinder", "AFilledCylinder"});
  // SphericalShells: 1 block
  REQUIRE(groups.count("SphericalShells") == 1);
  CHECK(groups.at("SphericalShells") ==
        std::unordered_set<std::string>{"SphericalShell"});
}

void test_domain_structure(const bool bulge = false) {
  INFO("Test domain structure (bulge=" << bulge << ")");
  const auto creator = make_creator(bulge);
  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  REQUIRE(blocks.size() == 27);

  // No excision spheres: neutron stars fill the inner volume
  CHECK(domain.excision_spheres().empty());

  // Check external boundaries per block.
  for (size_t i = 0; i < blocks.size(); ++i) {
    CAPTURE(i);
    CAPTURE(blocks[i].name());
    const auto& ext = blocks[i].external_boundaries();
    if (i == 26) {
      // SphericalShell (i==26): upper_xi is the outer boundary
      CHECK(ext.size() == 1);
      CHECK(ext.count(Direction<3>::upper_xi()) == 1);
    } else {
      CHECK(ext.empty());
    }
  }
}

void test_domain_creator_checks() {
  INFO("Test domain creator consistency checks");
  {
    INFO("Flat, no BC");
    const auto creator = make_creator(false);
    TestHelpers::domain::creators::test_domain_creator(creator, false);
    test_non_conforming_interface_logical_coords(
        creator.create_domain().blocks());
  }
  {
    INFO("Flat, with BC");
    const auto creator = make_creator(false, create_outer_boundary_condition());
    TestHelpers::domain::creators::test_domain_creator(creator, true);
  }
  {
    INFO("Bulged, no BC");
    const auto creator = make_creator(true);
    TestHelpers::domain::creators::test_domain_creator(creator, false);
    test_non_conforming_interface_logical_coords(
        creator.create_domain().blocks());
  }
  {
    INFO("Bulged, with BC");
    const auto creator = make_creator(true, create_outer_boundary_condition());
    TestHelpers::domain::creators::test_domain_creator(creator, true);
  }
}

void test_option_parsing() {
  INFO("Test option parsing");
  const auto array_output = [](const std::array<size_t, 4> input) {
    return std::string("[")
        .append(get_output(input[0]))
        .append(", ")
        .append(get_output(input[1]))
        .append(", ")
        .append(get_output(input[2]))
        .append(", ")
        .append(get_output(input[3]))
        .append("]");
  };
  const std::string option_string =
      "Pill:\n"
      "  CenterA: " +
      get_output(center_A) + "\n  CenterB: " + get_output(center_B) +
      "\n  LeftmostX: " + get_output(leftmost_x) +
      "\n  RightmostX: " + get_output(rightmost_x) +
      "\n  WedgeInnerRadius: " + get_output(wedge_inner_radius) +
      "\n  WedgeOuterRadius: " + get_output(wedge_outer_radius) +
      "\n  CylinderOuterRadius: " + get_output(cylinder_outer_radius) +
      "\n  OuterRadius: " + get_output(outer_radius) +
      "\n  CubeInitialGridPoints: " + array_output(cube_grid_points) +
      "\n  CubeInitialXRefinement: " + array_output(cube_x_refinement) +
      "\n  CubeInitialYZRefinement: " + array_output(cube_yz_refinement) +
      "\n  WedgePrismInitialRadialGridPoints: " +
      get_output(wedge_prism_radial_grid_points) +
      "\n  WedgePrismInitialRadialRefinement: " +
      get_output(wedge_prism_radial_refinement) +
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
      "\n  Bulge: false"
      "\n  TimeDependentMaps: None\n";
  const auto creator = make_creator();
  TestHelpers::domain::creators::test_creation(option_string, creator, false);
}

void test_parse_errors() {
  INFO("Test parse errors");

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          -1.0, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("CenterA must be positive"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, 1.0, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("CenterB must be negative"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          8.0, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("|CenterA| must be <= |CenterB|"));

  // LeftmostX must be < CenterB
  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, center_B + 0.1, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "LeftmostX must be less than CenterB"));

  // RightmostX must be > CenterA
  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, center_A - 0.1, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "RightmostX must be greater than CenterA"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, 0.0, wedge_outer_radius,
          cylinder_outer_radius, outer_radius, cube_grid_points,
          cube_x_refinement, cube_yz_refinement, wedge_prism_radial_grid_points,
          wedge_prism_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("WedgeInnerRadius must be positive"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius, -1.0,
          cylinder_outer_radius, outer_radius, cube_grid_points,
          cube_x_refinement, cube_yz_refinement, wedge_prism_radial_grid_points,
          wedge_prism_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("WedgeOuterRadius must be positive"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, 8.0, cube_grid_points,
          cube_x_refinement, cube_yz_refinement, wedge_prism_radial_grid_points,
          wedge_prism_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "OuterRadius must be greater than CylinderOuterRadius"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, 4.0, outer_radius, cube_grid_points,
          cube_x_refinement, cube_yz_refinement, wedge_prism_radial_grid_points,
          wedge_prism_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "CylinderOuterRadius must be greater than WedgeOuterRadius"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius, 2.0,
          cylinder_outer_radius, outer_radius, cube_grid_points,
          cube_x_refinement, cube_yz_refinement, wedge_prism_radial_grid_points,
          wedge_prism_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "WedgeOuterRadius must be greater than WedgeInnerRadius"));

  // CylindricalFlatEndcapInterior 10x radius ratio violated
  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, 80.0, 100.0, cube_grid_points, cube_x_refinement,
          cube_yz_refinement, wedge_prism_radial_grid_points,
          wedge_prism_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "CylindricalFlatEndcapInterior radius ratio condition violated"));

  // CylindricalFlatEndcapInterior t_sphere > 1 violated.
  // With center_A=2, center_B=-8, leftmost_x=-9 (= center_B - wedge_r),
  // rightmost_x=3 (= center_A + wedge_r), wedge_outer_radius=4,
  // cylinder_outer_radius=9: 9^2=81 <= 4^2 + (-9)^2 = 97.
  CHECK_THROWS_WITH(
      domain::creators::Pill(
          2.0, -8.0, -9.0, 3.0, 1.0, 4.0, 9.0, outer_radius, cube_grid_points,
          cube_x_refinement, cube_yz_refinement, wedge_prism_radial_grid_points,
          wedge_prism_radial_refinement, cylinder_radial_grid_points,
          cylinder_radial_refinement, b2_angular_grid_points,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "CylindricalFlatEndcapInterior (the B2s' map) requires"));

  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with a binary domain"));

  // Shape maps are not supported in the Pill domain (filled objects).
  using ShapeMapA = domain::creators::time_dependent_options::ShapeMapOptions<
      false, domain::ObjectLabel::A>;
  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{7_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false,
          domain::creators::bco::TimeDependentMapOptions<true>{
              0.0, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
              ShapeMapA{8_st, std::nullopt}, std::nullopt, std::nullopt},
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("does not support shape maps"));

  // B2InitialAngularGridPoints must be odd.
  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement, 8_st,
          std::array<size_t, 2>{7_st, 3_st}, spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "B2InitialAngularGridPoints must be odd"));

  // HollowCylinderInitialAngularGridPoints[0] (eta) must be odd.
  CHECK_THROWS_WITH(
      domain::creators::Pill(
          center_A, center_B, leftmost_x, rightmost_x, wedge_inner_radius,
          wedge_outer_radius, cylinder_outer_radius, outer_radius,
          cube_grid_points, cube_x_refinement, cube_yz_refinement,
          wedge_prism_radial_grid_points, wedge_prism_radial_refinement,
          cylinder_radial_grid_points, cylinder_radial_refinement,
          b2_angular_grid_points, std::array<size_t, 2>{6_st, 3_st},
          spherical_shells_radial_refinement,
          spherical_shells_radial_grid_points, spherical_harmonic_l,
          domain::CoordinateMaps::Distribution::Linear, false, std::nullopt,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "HollowCylinderInitialAngularGridPoints[0] must be odd"));
}

void test_time_dependent_maps() {
  INFO("Test time-dependent maps");
  using TimeDepOptions = domain::creators::bco::TimeDependentMapOptions<true>;

  const auto make_time_dep_options = []() {
    return TimeDepOptions{
        0.0,
        std::nullopt,
        domain::creators::time_dependent_options::RotationMapOptions<false>{
            std::array{0.0, 0.0, -0.2}},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
  };

  // Build Pill creator with time-dependent options
  const auto creator =
      domain::creators::Pill{center_A,
                             center_B,
                             leftmost_x,
                             rightmost_x,
                             wedge_inner_radius,
                             wedge_outer_radius,
                             cylinder_outer_radius,
                             outer_radius,
                             cube_grid_points,
                             cube_x_refinement,
                             cube_yz_refinement,
                             wedge_prism_radial_grid_points,
                             wedge_prism_radial_refinement,
                             cylinder_radial_grid_points,
                             cylinder_radial_refinement,
                             b2_angular_grid_points,
                             std::array<size_t, 2>{7_st, 3_st},
                             spherical_shells_radial_refinement,
                             spherical_shells_radial_grid_points,
                             spherical_harmonic_l,
                             domain::CoordinateMaps::Distribution::Linear,
                             false,
                             make_time_dep_options(),
                             nullptr};

  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  REQUIRE(blocks.size() == 27);

  // Build reference options with the same build_maps call as Pill's constructor
  auto time_dep_options = make_time_dep_options();
  time_dep_options.build_maps(
      std::array{std::array<double, 3>{center_A, 0., 0.},
                 std::array<double, 3>{center_B, 0., 0.}},
      std::nullopt, std::nullopt, std::array<double, 3>{0., 0., 0.},
      std::optional<std::array<double, 2>>{std::nullopt},
      std::optional<std::array<double, 2>>{std::nullopt}, true, true,
      cylinder_outer_radius, outer_radius);

  // All blocks must be time-dependent; none have a distorted frame (no shape
  // maps in the "both cubes" / BNS mode).
  for (size_t i = 0; i < blocks.size(); ++i) {
    CAPTURE(i);
    CHECK(blocks[i].is_time_dependent());
    CHECK(not blocks[i].has_distorted_frame());
  }

  // Blocks 0-25 (inner region): rigid RotScaleTrans map
  const auto rigid_map =
      time_dep_options.grid_to_inertial_map<domain::ObjectLabel::None>(false,
                                                                       true);
  for (size_t i = 0; i < 26; ++i) {
    CAPTURE(i);
    CHECK(blocks[i].moving_mesh_grid_to_inertial_map() == *rigid_map);
  }

  // Block 26 (SphericalShell): transitioning RotScaleTrans map
  const auto transition_map =
      time_dep_options.grid_to_inertial_map<domain::ObjectLabel::None>(false,
                                                                       false);
  CHECK(blocks[26].moving_mesh_grid_to_inertial_map() == *transition_map);

  // Rotation function of time must be present
  const auto fot = creator.functions_of_time({});
  CHECK(fot.find("Rotation") != fot.end());
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Pill", "[Domain][Unit]") {
  test_block_names_and_groups();
  test_domain_structure(false);
  test_domain_structure(true);
  test_domain_creator_checks();
  test_option_parsing();
  test_parse_errors();
  test_time_dependent_maps();
}
