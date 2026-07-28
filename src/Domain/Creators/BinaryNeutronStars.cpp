// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryNeutronStars.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/FlatOffsetSphericalWedge.hpp"
#include "Domain/CoordinateMaps/FlatOffsetWedge.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"

namespace domain::creators {
BinaryNeutronStars::BinaryNeutronStars(
    double center_A, double center_B, double inner_radius,
    double wedge_outer_radius, double cylinder_outer_radius,
    double outer_radius, size_t cube_grid_points, size_t cube_refinement,
    size_t deformed_cube_radial_grid_points,
    size_t deformed_cube_radial_refinement, size_t cylinder_radial_grid_points,
    size_t cylinder_radial_refinement, size_t b2_angular_grid_points,
    std::array<size_t, 2> hollow_cylinder_angular_grid_points,
    size_t spherical_shells_radial_refinement,
    size_t spherical_shells_radial_grid_points, size_t spherical_harmonic_l,
    domain::CoordinateMaps::Distribution SphericalShellsRadialDistribution,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : center_A_(center_A),
      center_B_(center_B),
      inner_radius_(inner_radius),
      wedge_outer_radius_(wedge_outer_radius),
      cylinder_outer_radius_(cylinder_outer_radius),
      outer_radius_(outer_radius),
      cube_grid_points_(cube_grid_points),
      cube_refinement_(cube_refinement),
      deformed_cube_radial_grid_points_(deformed_cube_radial_grid_points),
      deformed_cube_radial_refinement_(deformed_cube_radial_refinement),
      cylinder_radial_grid_points_(cylinder_radial_grid_points),
      cylinder_radial_refinement_(cylinder_radial_refinement),
      b2_angular_grid_points_(b2_angular_grid_points),
      hollow_cylinder_angular_grid_points_(hollow_cylinder_angular_grid_points),
      spherical_shells_radial_refinement_(spherical_shells_radial_refinement),
      spherical_shells_radial_grid_points_(spherical_shells_radial_grid_points),
      spherical_harmonic_l_(spherical_harmonic_l),
      SphericalShellsRadialDistribution_(SphericalShellsRadialDistribution),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  if (center_A_ <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterA is expected to be positive");
  }
  if (center_B_ >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterB is expected to be negative");
  }
  if (std::abs(center_A_) > std::abs(center_B_)) {
    PARSE_ERROR(context, "We expect |x_A| <= |x_B|.");
  }
  if (inner_radius_ <= 0.0) {
    PARSE_ERROR(context, "InnerRadius must be positive.");
  }
  if (wedge_outer_radius <= 0.0) {
    PARSE_ERROR(context, "WedgeOuterRadius is expected to be positive");
  }
  if (cylinder_outer_radius <= wedge_outer_radius) {
    PARSE_ERROR(
        context,
        "CylinderOuterRadius is expected to be greater than WedgeOuterRadius.");
  }
  if (outer_radius <= cylinder_outer_radius) {
    PARSE_ERROR(
        context,
        "OuterRadius is expected to be greater than CylinderOuterRadius.");
  }
  if (center_A_ >= wedge_outer_radius_ or -center_B_ >= wedge_outer_radius_) {
    PARSE_ERROR(context,
                "The WedgeOuterRadius must be larger than the distance from "
                "the origin to the centers");
  }
  if (sqrt(2) * inner_radius_ >= wedge_outer_radius) {
    PARSE_ERROR(context,
                "The inner cubed region is outside of the wedges, InnerRadius "
                "is too large for the given WedgeOuterRadius");
  }
  if (center_A_ + sqrt(2) * wedge_outer_radius_ >= cylinder_outer_radius_) {
    PARSE_ERROR(context,
                "The wedges go outside of the cylinders, WedgeOuterRadius is "
                "too large for the given CylinderOuterRadius");
  }
  if (inner_radius_ >= center_A_ or inner_radius_ >= -center_B_) {
    PARSE_ERROR(context,
                "InnerRadius must be smaller than the distance from the origin "
                "to each star center (min(CenterA, |CenterB|) = " +
                    std::to_string(std::min(center_A_, -center_B_)) +
                    ", InnerRadius = " + std::to_string(inner_radius_) + ").");
  }
  if (center_A_ * center_A_ + 2.0 * inner_radius_ * inner_radius_ >=
          wedge_outer_radius_ * wedge_outer_radius_ or
      center_B_ * center_B_ + 2.0 * inner_radius_ * inner_radius_ >=
          wedge_outer_radius_ * wedge_outer_radius_) {
    PARSE_ERROR(context,
                "FlatOffsetWedge validity condition violated: center^2 + "
                "2*InnerRadius^2 must be less than WedgeOuterRadius^2 for "
                "both stars.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  number_of_blocks_ = 27;

  // Create block names and groups
  block_names_.reserve(number_of_blocks_);
  auto add_deformed_cube_name = [this](const std::string& prefix) {
    for (const std::string& where : {"Front"s, "Top"s, "Back"s, "Bottom"s}) {
      const std::string name =
          std::string(prefix).append("DeformedCube").append(where);
      block_names_.emplace_back(name);
      block_groups_["DeformedCubes"].insert(name);
    }
  };
  // Central 4 cubes
  block_names_.emplace_back("BLeftCentralCube");
  block_groups_["CenterCubes"].insert("BLeftCentralCube");
  block_names_.emplace_back("BRightCentralCube");
  block_groups_["CenterCubes"].insert("BRightCentralCube");
  block_names_.emplace_back("ALeftCentralCube");
  block_groups_["CenterCubes"].insert("ALeftCentralCube");
  block_names_.emplace_back("ARightCentralCube");
  block_groups_["CenterCubes"].insert("ARightCentralCube");

  // 18 deformed cubes, touching the faces of the central cubes
  block_names_.emplace_back("BDeformedCubeCap");
  block_groups_["DeformedCubes"].insert("BDeformedCubeCap");
  add_deformed_cube_name("BLeft");   // 5 - 8
  add_deformed_cube_name("BRight");  // 9 -12
  add_deformed_cube_name("ALeft");   // 13-16
  add_deformed_cube_name("ARight");  // 17-20
  block_names_.emplace_back("ADeformedCubeCap");
  block_groups_["DeformedCubes"].insert("ADeformedCubeCap");

  // 4 cylinders, between the deformed cubes and spherical shells
  block_names_.emplace_back("BFilledCylinder");
  block_groups_["Cylinders"].insert("BFilledCylinder");
  block_names_.emplace_back("BHollowCylinder");
  block_groups_["Cylinders"].insert("BHollowCylinder");
  block_names_.emplace_back("AHollowCylinder");
  block_groups_["Cylinders"].insert("AHollowCylinder");
  block_names_.emplace_back("AFilledCylinder");
  block_groups_["Cylinders"].insert("AFilledCylinder");

  // 1 spherical shell
  block_names_.emplace_back("SphericalShell");
  block_groups_["SphericalShells"].insert("SphericalShell");

  // initial grid points and refinement
  initial_grid_points_.resize(number_of_blocks_);
  initial_refinement_.resize(number_of_blocks_);

  // Blocks 0-3: central cubes, uniform in all three directions.
  for (size_t i = 0; i < 4; ++i) {
    initial_grid_points_[i] = {cube_grid_points_, cube_grid_points_,
                               cube_grid_points_};
    initial_refinement_[i] = {cube_refinement_, cube_refinement_,
                              cube_refinement_};
  }
  // Blocks 4-21: deformed cubes (Wedge / FlatOffsetWedge).
  // For 3D Wedge: xi,eta = angular (first two); zeta = radial (third).
  // FlatOffsetWedge: xi = x-direction; eta = y-direction; zeta = radial.
  for (size_t i = 4; i < 22; ++i) {
    initial_grid_points_[i] = {cube_grid_points_, cube_grid_points_,
                               deformed_cube_radial_grid_points_};
    initial_refinement_[i] = {cube_refinement_, cube_refinement_,
                              deformed_cube_radial_refinement_};
  }
  // Blocks 22 & 25: filled cylinders (full_cylinder topology).
  // xi (B2Radial) is derived from eta (B2Angular); both cannot be h-refined.
  // eta must be odd. zeta (I1, axial) uses the cylinder radial parameters.
  size_t b2_angular = b2_angular_grid_points_;
  if (b2_angular % 2 == 0) {
    b2_angular += 1;
  }
  const size_t theta_modes = b2_angular / 2;
  const size_t b2_radial = theta_modes / 2 + 1 + theta_modes % 2;
  for (const size_t i : {22_st, 25_st}) {
    initial_grid_points_[i] = {b2_radial, b2_angular,
                               cylinder_radial_grid_points_};
    initial_refinement_[i] = {0, 0, cylinder_radial_refinement_};
  }
  // Blocks 23 & 24: hollow cylinders (cylindrical_shell topology).
  // xi (I1, radial) and eta (S1, angular) cannot be h-refined.
  // eta must be odd. zeta (I1, axial) uses the cylinder radial parameters.
  // hollow_cylinder_angular_grid_points_ = {eta_pts, zeta_pts}.
  size_t hollow_eta = hollow_cylinder_angular_grid_points_[0];
  if (hollow_eta % 2 == 0) {
    hollow_eta += 1;
  }
  for (const size_t i : {23_st, 24_st}) {
    initial_grid_points_[i] = {cylinder_radial_grid_points_, hollow_eta,
                               hollow_cylinder_angular_grid_points_[1]};
    initial_refinement_[i] = {0, 0, cylinder_radial_refinement_};
  }
  // Block 26: spherical shell (spherical_shell topology).
  // eta (S2Colatitude) and zeta (S2Longitude) cannot be h-refined.
  // Spherepack requires zeta_pts = 2 * eta_pts - 1.
  const size_t colatitude_pts = spherical_harmonic_l_ + 1;
  initial_grid_points_[26] = {spherical_shells_radial_grid_points_,
                              colatitude_pts, 2 * colatitude_pts - 1};
  initial_refinement_[26] = {spherical_shells_radial_refinement_, 0, 0};
}

Domain<3> BinaryNeutronStars::create_domain() const {
  using Interval = CoordinateMaps::Interval;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Wedge = CoordinateMaps::Wedge<3>;
  using WedgeHalves = Wedge::WedgeHalves;
  using FlatOffsetWedge = CoordinateMaps::FlatOffsetWedge;
  using Identity1D = CoordinateMaps::Identity<1>;
  using Rotation = CoordinateMaps::DiscreteRotation<3>;
  using ShiftX = CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Identity1D>;

  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};
  coordinate_maps.reserve(number_of_blocks_);
  // Group tag for each block, used to wire up the explicit neighbor graph
  std::vector<std::string> block_tags{};

  const auto aligned = OrientationMap<3>::create_aligned();
  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};

  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  const OrientationMap<3> rotate_to_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
      Direction<3>::lower_eta()}};

  const OrientationMap<3> rotate_to_minus_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
      Direction<3>::upper_eta()}};

  // rotating by changing y, keeping x constant
  const OrientationMap<3> rotate_to_minus_z_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_eta(),
      Direction<3>::lower_zeta()}};

  const OrientationMap<3> half_turn_about_z(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});

  // Effective orientations for BRight FOW blocks (htaz composed with M2).
  // R_eff = R(htaz) · R(M2); all three are 180° rotations and are self-inverse.
  // BRightFront  (htaz · rmy): ξ→-ξ, η→-ζ, ζ→-η
  const OrientationMap<3> half_turn_about_y_minus_z_axis{
      std::array<Direction<3>, 3>{Direction<3>::lower_xi(),
                                  Direction<3>::lower_zeta(),
                                  Direction<3>::lower_eta()}};
  // BRightBack   (htaz · ry):  ξ→-ξ, η→+ζ, ζ→+η
  const OrientationMap<3> half_turn_about_y_plus_z_axis{
      std::array<Direction<3>, 3>{Direction<3>::lower_xi(),
                                  Direction<3>::upper_zeta(),
                                  Direction<3>::upper_eta()}};
  // BRightBottom (htaz · rmz): ξ→-ξ, η→+η, ζ→-ζ
  const OrientationMap<3> half_turn_about_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_xi(), Direction<3>::upper_eta(),
      Direction<3>::lower_zeta()}};

  const bool use_equiangular_maps = false;

  // Add central blocks, only the xi extents changing
  auto add_central_cube = [this, &coordinate_maps, &block_tags](
                              const double lower_xi, const double upper_xi,
                              const std::string& tag) {
    coordinate_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3D{Affine{-1.0, 1.0, lower_xi, upper_xi},
                     Affine{-1.0, 1.0, -inner_radius_, inner_radius_},
                     Affine{-1.0, 1.0, -inner_radius_, inner_radius_}}));
    block_tags.push_back(tag);
  };
  add_central_cube(-inner_radius_ + center_B_, center_B_, "BLeftCentralCube");
  add_central_cube(center_B_, 0.0, "BRightCentralCube");
  add_central_cube(0.0, center_A_, "ALeftCentralCube");
  add_central_cube(center_A_, inner_radius_ + center_A_, "ARightCentralCube");

  // ---- Blocks 4-21: deformed cubes ----
  // "A"-labeled blocks (4-12) surround star B (negative x).
  // "B"-labeled blocks (13-21) surround star A (positive x).
  // When center_A_ != center_B_, the radii of the wedges need to be modified
  // so that their intersection at x = 0 has no overlap. This is done by
  // shrinking one. Negating the if statement will grow the other instead
  double wedge_A_radius{};
  double wedge_B_radius{};
  if (-center_B_ > center_A_) {
    wedge_A_radius = sqrt(square(wedge_outer_radius_) -
                          abs(square(center_A_) - square(center_B_)));
    wedge_B_radius = wedge_outer_radius_;
  } else {
    wedge_A_radius = wedge_outer_radius_;
    wedge_B_radius = sqrt(square(wedge_outer_radius_) -
                          abs(square(center_A_) - square(center_B_)));
  }
  const double sqrt3 = sqrt(3.0);
  const auto shift_to_B =
      ShiftX{Affine{-1.0, 1.0, -1.0 + center_B_, 1.0 + center_B_}, Identity1D{},
             Identity1D{}};
  const auto shift_to_A =
      ShiftX{Affine{-1.0, 1.0, -1.0 + center_A_, 1.0 + center_A_}, Identity1D{},
             Identity1D{}};

  // Block 4: BLeftDeformedCubeCap — full wedge pointing -x from center_B_
  coordinate_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Wedge{sqrt3 * inner_radius_, wedge_B_radius, 0.0, 1.0,
                rotate_to_minus_x_axis, use_equiangular_maps},
          shift_to_B));
  block_tags.emplace_back("BDeformedCubeCap");

  // Blocks 5-8: BLeftDeformedCube* — half-wedges around center_B_
  auto add_half_wedge = [this, &coordinate_maps, &block_tags, sqrt3,
                         wedge_A_radius, wedge_B_radius, use_equiangular_maps](
                            const auto rotation, const auto shift,
                            const std::string& tag, const bool is_A = false) {
    coordinate_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Wedge{sqrt3 * inner_radius_, is_A ? wedge_A_radius : wedge_B_radius,
                  0.0, 1.0, rotation, use_equiangular_maps,
                  is_A ? WedgeHalves::UpperOnly : WedgeHalves::LowerOnly},
            shift));
    block_tags.push_back(tag);
  };
  add_half_wedge(rotate_to_minus_y_axis, shift_to_B, "BLeftDeformedCubeFront");
  add_half_wedge(aligned, shift_to_B, "BLeftDeformedCubeTop");
  add_half_wedge(rotate_to_y_axis, shift_to_B, "BLeftDeformedCubeBack");
  add_half_wedge(rotate_to_minus_z_axis, shift_to_B, "BLeftDeformedCubeBottom");

  // Blocks 9-12: BRightDeformedCube* — flat-offset-wedges (B-facing side of B)
  auto add_flat_offset_wedge_B =
      [this, &coordinate_maps, &block_tags, wedge_B_radius](
          const double offset, const auto rotation1, const auto rotation2,
          const std::string& tag) {
        // B block, requires 2 rotations
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                FlatOffsetWedge{inner_radius_, offset, wedge_B_radius},
                Rotation{rotation1}, Rotation{rotation2}));
        block_tags.push_back(tag);
      };
  add_flat_offset_wedge_B(std::abs(center_B_), half_turn_about_z,
                          rotate_to_minus_y_axis, "BRightDeformedCubeFront");
  add_flat_offset_wedge_B(std::abs(center_B_), half_turn_about_z, aligned,
                          "BRightDeformedCubeTop");
  add_flat_offset_wedge_B(std::abs(center_B_), half_turn_about_z,
                          rotate_to_y_axis, "BRightDeformedCubeBack");
  add_flat_offset_wedge_B(std::abs(center_B_), half_turn_about_z,
                          rotate_to_minus_z_axis, "BRightDeformedCubeBottom");

  // Blocks 13-16: ALeftDeformedCube* — flat-offset-wedges (A-facing side of A)
  auto add_flat_offset_wedge_A =
      [this, &coordinate_maps, &block_tags, wedge_A_radius](
          const double offset, const auto rotation, const std::string& tag) {
        // A Block, requires 1 rotation
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                FlatOffsetWedge{inner_radius_, offset, wedge_A_radius},
                Rotation{rotation}));
        block_tags.push_back(tag);
      };
  add_flat_offset_wedge_A(center_A_, rotate_to_minus_y_axis,
                          "ALeftDeformedCubeFront");
  add_flat_offset_wedge_A(center_A_, aligned, "ALeftDeformedCubeTop");
  add_flat_offset_wedge_A(center_A_, rotate_to_y_axis, "ALeftDeformedCubeBack");
  add_flat_offset_wedge_A(center_A_, rotate_to_minus_z_axis,
                          "ALeftDeformedCubeBottom");

  // Blocks 17-20: ARightDeformedCube* — half-wedges around center_A_
  add_half_wedge(rotate_to_minus_y_axis, shift_to_A, "ARightDeformedCubeFront",
                 true);
  add_half_wedge(aligned, shift_to_A, "ARightDeformedCubeTop", true);
  add_half_wedge(rotate_to_y_axis, shift_to_A, "ARightDeformedCubeBack", true);
  add_half_wedge(rotate_to_minus_z_axis, shift_to_A, "ARightDeformedCubeBottom",
                 true);

  // Block 21: ADeformedCubeCap — full wedge pointing +x from center_A_
  coordinate_maps.emplace_back(
      make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Wedge{sqrt3 * inner_radius_, wedge_A_radius, 0.0, 1.0,
                rotate_to_x_axis, use_equiangular_maps},
          shift_to_A));
  block_tags.emplace_back("ADeformedCubeCap");

  // ---- Lambdas for cylinder and sphere-shell blocks ----

  // ZernikeB2 filled-cylinder (full_cylinder topology).
  // Logical [xi=r, eta=phi, zeta=axial]: xi in [-1,1] -> r in [0,1];
  // PolarToCartesian produces the unit right cylinder the endcap map expects.
  // The minus-x-axis blocks need half_turn_about_z to fix azimuthal handedness.
  const auto add_filled_cylinder = [&coordinate_maps, &block_tags,
                                    &rotate_to_minus_x_axis, &half_turn_about_z,
                                    &aligned](const auto& endcap_map,
                                              const Rotation& rotation_map,
                                              const std::string& tag) {
    const bool minus_side = rotation_map == Rotation{rotate_to_minus_x_axis};
    coordinate_maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            ShiftX{Affine{-1.0, 1.0, 0.0, 1.0}, Identity1D{}, Identity1D{}},
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                           Identity1D>{
                CoordinateMaps::PolarToCartesian{}, Identity1D{}},
            Rotation{minus_side ? half_turn_about_z : aligned}, endcap_map,
            rotation_map));
    block_tags.push_back(tag);
  };

  // Fourier hollow-cylinder (cylindrical_shell topology).
  // xi in [-1,1] -> r in [1,2]; PolarToCartesian maps to the annular region.
  const auto add_hollow_cylinder =
      [&coordinate_maps, &block_tags, &rotate_to_minus_x_axis,
       &half_turn_about_z,
       &aligned](const CoordinateMaps::UniformCylindricalSide& side_map,
                 const Rotation& rotation_map, const std::string& tag) {
        const bool minus_side =
            rotation_map == Rotation{rotate_to_minus_x_axis};
        coordinate_maps.push_back(domain::make_coordinate_map_base<
                                  Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf3Maps<Affine, Identity1D,
                                           CoordinateMaps::Interval>{
                Affine{-1.0, 1.0, 1.0, 2.0}, Identity1D{},
                CoordinateMaps::Interval{-1.0, 1.0, -1.0, 1.0,
                                         CoordinateMaps::Distribution::Linear}},
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                           Identity1D>{
                CoordinateMaps::PolarToCartesian{}, Identity1D{}},
            Rotation{minus_side ? half_turn_about_z : aligned}, side_map,
            rotation_map));
        block_tags.push_back(tag);
      };

  // S2 spherical-harmonic shell (spherical_shell topology).
  const auto add_sphere_shell = [this, &coordinate_maps, &block_tags](
                                    const double inner_radius,
                                    const double outer_radius,
                                    const std::string& tag) {
    coordinate_maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<Interval,
                                           CoordinateMaps::Identity<2>>{
                Interval{-1.0, 1.0, inner_radius, outer_radius,
                         SphericalShellsRadialDistribution_, 0.0},
                CoordinateMaps::Identity<2>{}},
            CoordinateMaps::SphericalToCartesianPfaffian{}));
    block_tags.push_back(tag);
  };

  // ---- Blocks 22-25: cylinders ----
  // The boundary between the filled and hollow cylinders occurs at the boundary
  // between the half wedges and cap wedge. The upper cuts are set such that
  // they align with the line connecting the origin to the lower cut
  const double z_cut_A_lower = center_A_ + wedge_A_radius / sqrt(2);
  const double z_cut_B_lower = -center_B_ + wedge_B_radius / sqrt(2);
  const double z_cut_A_upper =
      cylinder_outer_radius_ /
      sqrt(1.0 + square(wedge_A_radius) /
                     (2 * square(center_A_ + wedge_A_radius / sqrt(2))));
  const double z_cut_B_upper =
      cylinder_outer_radius_ /
      sqrt(1.0 + square(wedge_B_radius) /
                     (2 * square(-center_B_ + wedge_B_radius / sqrt(2))));

  // Block 22: BFilledCylinder
  add_filled_cylinder(
      CoordinateMaps::UniformCylindricalEndcap(
          std::array<double, 3>{0.0, 0.0, -center_B_}, make_array<3>(0.0),
          wedge_B_radius, cylinder_outer_radius_, z_cut_B_lower, z_cut_B_upper),
      Rotation{rotate_to_minus_x_axis}, "BFilledCylinder");

  // Block 23: BHollowCylinder
  add_hollow_cylinder(
      CoordinateMaps::UniformCylindricalSide(
          std::array<double, 3>{0.0, 0.0, -center_B_}, make_array<3>(0.0),
          wedge_B_radius, cylinder_outer_radius_, z_cut_B_lower, 0.0,
          z_cut_B_upper, 0.0),
      Rotation{rotate_to_minus_x_axis}, "BHollowCylinder");

  // Block 24: AHollowCylinder
  add_hollow_cylinder(
      CoordinateMaps::UniformCylindricalSide(
          std::array<double, 3>{0.0, 0.0, center_A_}, make_array<3>(0.0),
          wedge_A_radius, cylinder_outer_radius_, z_cut_A_lower, 0.0,
          z_cut_A_upper, 0.0),
      Rotation{rotate_to_x_axis}, "AHollowCylinder");

  // Block 25: AFilledCylinder
  add_filled_cylinder(
      CoordinateMaps::UniformCylindricalEndcap(
          std::array<double, 3>{0.0, 0.0, center_A_}, make_array<3>(0.0),
          wedge_A_radius, cylinder_outer_radius_, z_cut_A_lower, z_cut_A_upper),
      Rotation{rotate_to_x_axis}, "AFilledCylinder");

  // Block 26: SphericalShell
  add_sphere_shell(cylinder_outer_radius_, outer_radius_, "SphericalShell");

  // ---- Neighbor graph ----
  // Map each block group tag to its block index.
  std::unordered_map<std::string, size_t> tag_to_index{};
  for (size_t i = 0; i < block_tags.size(); ++i) {
    tag_to_index.emplace(block_tags[i], i);
  }
  // Directed neighbor graph keyed by block tag:
  // {self tag, self direction, neighbor tag, orientation of neighbor}.
  // Edges whose neighbor group is absent are skipped; those faces become
  // external boundaries.
  using Dir = Direction<3>;
  // Filled-cylinder mantle (upper_xi) -> abutting side; inverse is orient_sf.
  const OrientationMap<3> orient_fs{
      std::array<Dir, 3>{Dir::lower_zeta(), Dir::upper_eta(), Dir::upper_xi()}};
  // Side z-end (upper_zeta) -> abutting filled cylinder.
  const OrientationMap<3> orient_sf{
      std::array<Dir, 3>{Dir::upper_zeta(), Dir::upper_eta(), Dir::lower_xi()}};
  // Cutting-plane seam between A-side and B-side hollow cylinders.
  const OrientationMap<3> orient_zseam{
      std::array<Dir, 3>{Dir::upper_xi(), Dir::lower_eta(), Dir::lower_zeta()}};

  // ---- Cap <-> side angular connections ----
  // BDeformedCubeCap axes: ξ=+z, η=+y, ζ=-x (outer face in -x direction).
  // All BLeft side blocks have ξ=+x; their outer angular face is lower_xi.
  //   cap.upper_xi(+z) <-> BLeftTop.lower_xi:    aligned (η=η, ζ=ζ)
  //   cap.lower_xi(-z) <-> BLeftBottom.lower_xi: half_turn_about_z (η flips)
  //   cap.lower_eta(-y) <-> BLeftFront.lower_xi: ξ_cap(+z)=η_front(+z), ζ=ζ
  const OrientationMap<3> orient_bcap_to_bfront{
      std::array<Dir, 3>{Dir::upper_eta(), Dir::lower_xi(), Dir::upper_zeta()}};
  const OrientationMap<3> orient_bfront_to_bcap{
      std::array<Dir, 3>{Dir::lower_eta(), Dir::upper_xi(), Dir::upper_zeta()}};
  //   cap.upper_eta(+y) <-> BLeftBack.lower_xi: ξ_cap(+z)=-η_back(-z), ζ=ζ
  const OrientationMap<3> orient_bcap_to_bback{
      std::array<Dir, 3>{Dir::lower_eta(), Dir::upper_xi(), Dir::upper_zeta()}};
  const OrientationMap<3> orient_bback_to_bcap{
      std::array<Dir, 3>{Dir::upper_eta(), Dir::lower_xi(), Dir::upper_zeta()}};

  // ADeformedCubeCap axes: ξ=-z, η=+y, ζ=+x (outer face in +x direction).
  // All ARight side blocks have ξ=+x; their outer angular face is upper_xi.
  //   acap.lower_xi(+z) <-> ARightTop.upper_xi:    aligned (η=η, ζ=ζ)
  //   acap.upper_xi(-z) <-> ARightBottom.upper_xi: half_turn_about_z (η flips)
  //   acap.lower_eta(-y) <-> ARightFront.upper_xi: ξ_cap(-z)=η_front(+z), ζ=ζ
  const OrientationMap<3> orient_acap_to_afront{
      std::array<Dir, 3>{Dir::lower_eta(), Dir::upper_xi(), Dir::upper_zeta()}};
  const OrientationMap<3> orient_afront_to_acap{
      std::array<Dir, 3>{Dir::upper_eta(), Dir::lower_xi(), Dir::upper_zeta()}};
  //   acap.upper_eta(+y) <-> ARightBack.upper_xi: ξ_cap(-z)=η_back(-z), ζ=ζ
  const OrientationMap<3> orient_acap_to_aback{
      std::array<Dir, 3>{Dir::upper_eta(), Dir::lower_xi(), Dir::upper_zeta()}};
  const OrientationMap<3> orient_aback_to_acap{
      std::array<Dir, 3>{Dir::lower_eta(), Dir::upper_xi(), Dir::upper_zeta()}};

  struct EdgeSpec {
    std::string self_tag;
    Dir self_direction;
    std::string neighbor_tag;
    const OrientationMap<3>* orientation;
  };
  const std::vector<EdgeSpec> edges{
      // Inner 4 central cubes to each other (left to right in x:
      // BLeft < BRight < ALeft < ARight)
      {"BLeftCentralCube", Dir::upper_xi(), "BRightCentralCube", &aligned},
      {"BRightCentralCube", Dir::lower_xi(), "BLeftCentralCube", &aligned},

      {"BRightCentralCube", Dir::upper_xi(), "ALeftCentralCube", &aligned},
      {"ALeftCentralCube", Dir::lower_xi(), "BRightCentralCube", &aligned},

      {"ALeftCentralCube", Dir::upper_xi(), "ARightCentralCube", &aligned},
      {"ARightCentralCube", Dir::lower_xi(), "ALeftCentralCube", &aligned},

      // Inner central cubes <-> wedge caps
      // BDeformedCubeCap: Wedge with rotate_to_minus_x_axis, pointing -x from
      // center_B_; inner face (lower_zeta) abuts BLeftCentralCube's lower_xi.
      {"BLeftCentralCube", Dir::lower_xi(), "BDeformedCubeCap",
       &rotate_to_minus_x_axis},
      {"BDeformedCubeCap", Dir::lower_zeta(), "BLeftCentralCube",
       &rotate_to_x_axis},

      // ADeformedCubeCap: Wedge with rotate_to_x_axis, pointing +x from
      // center_A_; inner face (lower_zeta) abuts ARightCentralCube's upper_xi.
      {"ARightCentralCube", Dir::upper_xi(), "ADeformedCubeCap",
       &rotate_to_x_axis},
      {"ADeformedCubeCap", Dir::lower_zeta(), "ARightCentralCube",
       &rotate_to_minus_x_axis},

      // BLeft half-wedges <-> BLeftCentralCube lateral faces.
      // Each Wedge was constructed with the OrientationMap named below; the
      // orientation from the cube face to the wedge inner face (lower_zeta) is
      // exactly that same OrientationMap, and the reverse is its inverse.
      {"BLeftCentralCube", Dir::lower_eta(), "BLeftDeformedCubeFront",
       &rotate_to_minus_y_axis},
      {"BLeftDeformedCubeFront", Dir::lower_zeta(), "BLeftCentralCube",
       &rotate_to_y_axis},

      {"BLeftCentralCube", Dir::upper_zeta(), "BLeftDeformedCubeTop", &aligned},
      {"BLeftDeformedCubeTop", Dir::lower_zeta(), "BLeftCentralCube", &aligned},

      {"BLeftCentralCube", Dir::upper_eta(), "BLeftDeformedCubeBack",
       &rotate_to_y_axis},
      {"BLeftDeformedCubeBack", Dir::lower_zeta(), "BLeftCentralCube",
       &rotate_to_minus_y_axis},

      {"BLeftCentralCube", Dir::lower_zeta(), "BLeftDeformedCubeBottom",
       &rotate_to_minus_z_axis},
      {"BLeftDeformedCubeBottom", Dir::lower_zeta(), "BLeftCentralCube",
       &rotate_to_minus_z_axis},

      // ARight half-wedges <-> ARightCentralCube lateral faces.
      // Same pattern: UpperOnly wedges shifted to center_A_.
      {"ARightCentralCube", Dir::lower_eta(), "ARightDeformedCubeFront",
       &rotate_to_minus_y_axis},
      {"ARightDeformedCubeFront", Dir::lower_zeta(), "ARightCentralCube",
       &rotate_to_y_axis},

      {"ARightCentralCube", Dir::upper_zeta(), "ARightDeformedCubeTop",
       &aligned},
      {"ARightDeformedCubeTop", Dir::lower_zeta(), "ARightCentralCube",
       &aligned},

      {"ARightCentralCube", Dir::upper_eta(), "ARightDeformedCubeBack",
       &rotate_to_y_axis},
      {"ARightDeformedCubeBack", Dir::lower_zeta(), "ARightCentralCube",
       &rotate_to_minus_y_axis},

      {"ARightCentralCube", Dir::lower_zeta(), "ARightDeformedCubeBottom",
       &rotate_to_minus_z_axis},
      {"ARightDeformedCubeBottom", Dir::lower_zeta(), "ARightCentralCube",
       &rotate_to_minus_z_axis},

      // ALeft FOW blocks <-> ALeftCentralCube lateral faces.
      // Each ALeft FOW block has a single rotation M; the orientation from the
      // cube face to the block's inner face (lower_zeta) is exactly M, and the
      // reverse is M's inverse (same pattern as BLeft half-wedges).
      {"ALeftCentralCube", Dir::lower_eta(), "ALeftDeformedCubeFront",
       &rotate_to_minus_y_axis},
      {"ALeftDeformedCubeFront", Dir::lower_zeta(), "ALeftCentralCube",
       &rotate_to_y_axis},

      {"ALeftCentralCube", Dir::upper_zeta(), "ALeftDeformedCubeTop", &aligned},
      {"ALeftDeformedCubeTop", Dir::lower_zeta(), "ALeftCentralCube", &aligned},

      {"ALeftCentralCube", Dir::upper_eta(), "ALeftDeformedCubeBack",
       &rotate_to_y_axis},
      {"ALeftDeformedCubeBack", Dir::lower_zeta(), "ALeftCentralCube",
       &rotate_to_minus_y_axis},

      {"ALeftCentralCube", Dir::lower_zeta(), "ALeftDeformedCubeBottom",
       &rotate_to_minus_z_axis},
      {"ALeftDeformedCubeBottom", Dir::lower_zeta(), "ALeftCentralCube",
       &rotate_to_minus_z_axis},

      // BRight FOW blocks <-> BRightCentralCube lateral faces.
      // Each BRight FOW block has two rotations: htaz then M2. The effective
      // orientation is R(htaz)·R(M2), giving the three new maps defined above.
      // The maps are all 180° rotations and hence self-inverse.
      {"BRightCentralCube", Dir::lower_eta(), "BRightDeformedCubeFront",
       &half_turn_about_y_minus_z_axis},
      {"BRightDeformedCubeFront", Dir::lower_zeta(), "BRightCentralCube",
       &half_turn_about_y_minus_z_axis},

      {"BRightCentralCube", Dir::upper_zeta(), "BRightDeformedCubeTop",
       &half_turn_about_z},
      {"BRightDeformedCubeTop", Dir::lower_zeta(), "BRightCentralCube",
       &half_turn_about_z},

      {"BRightCentralCube", Dir::upper_eta(), "BRightDeformedCubeBack",
       &half_turn_about_y_plus_z_axis},
      {"BRightDeformedCubeBack", Dir::lower_zeta(), "BRightCentralCube",
       &half_turn_about_y_plus_z_axis},

      {"BRightCentralCube", Dir::lower_zeta(), "BRightDeformedCubeBottom",
       &half_turn_about_y_axis},
      {"BRightDeformedCubeBottom", Dir::lower_zeta(), "BRightCentralCube",
       &half_turn_about_y_axis},

      // Seam at x = center_B_: BLeft*.upper_xi <-> BRight*.upper_xi.
      // BLeft LowerOnly wedges and BRight FOW blocks both have their seam face
      // at x = center_B_. The relative orientation is half_turn_about_z
      // because BRight carries an htaz rotation absent from BLeft.
      {"BLeftDeformedCubeFront", Dir::upper_xi(), "BRightDeformedCubeFront",
       &half_turn_about_z},
      {"BRightDeformedCubeFront", Dir::upper_xi(), "BLeftDeformedCubeFront",
       &half_turn_about_z},
      {"BLeftDeformedCubeTop", Dir::upper_xi(), "BRightDeformedCubeTop",
       &half_turn_about_z},
      {"BRightDeformedCubeTop", Dir::upper_xi(), "BLeftDeformedCubeTop",
       &half_turn_about_z},
      {"BLeftDeformedCubeBack", Dir::upper_xi(), "BRightDeformedCubeBack",
       &half_turn_about_z},
      {"BRightDeformedCubeBack", Dir::upper_xi(), "BLeftDeformedCubeBack",
       &half_turn_about_z},
      {"BLeftDeformedCubeBottom", Dir::upper_xi(), "BRightDeformedCubeBottom",
       &half_turn_about_z},
      {"BRightDeformedCubeBottom", Dir::upper_xi(), "BLeftDeformedCubeBottom",
       &half_turn_about_z},

      // Seam at x = 0: BRight*.lower_xi <-> ALeft*.lower_xi.
      // BRight carries htaz (which negates x and y) while ALeft does not, so
      // the relative orientation is again half_turn_about_z.
      {"BRightDeformedCubeFront", Dir::lower_xi(), "ALeftDeformedCubeFront",
       &half_turn_about_z},
      {"ALeftDeformedCubeFront", Dir::lower_xi(), "BRightDeformedCubeFront",
       &half_turn_about_z},
      {"BRightDeformedCubeTop", Dir::lower_xi(), "ALeftDeformedCubeTop",
       &half_turn_about_z},
      {"ALeftDeformedCubeTop", Dir::lower_xi(), "BRightDeformedCubeTop",
       &half_turn_about_z},
      {"BRightDeformedCubeBack", Dir::lower_xi(), "ALeftDeformedCubeBack",
       &half_turn_about_z},
      {"ALeftDeformedCubeBack", Dir::lower_xi(), "BRightDeformedCubeBack",
       &half_turn_about_z},
      {"BRightDeformedCubeBottom", Dir::lower_xi(), "ALeftDeformedCubeBottom",
       &half_turn_about_z},
      {"ALeftDeformedCubeBottom", Dir::lower_xi(), "BRightDeformedCubeBottom",
       &half_turn_about_z},

      // Seam at x = center_A_: ALeft*.upper_xi <-> ARight*.lower_xi.
      // ALeft FOW blocks and ARight UpperOnly wedges share the same single
      // rotation M2, so their coordinate frames are aligned at the seam.
      {"ALeftDeformedCubeFront", Dir::upper_xi(), "ARightDeformedCubeFront",
       &aligned},
      {"ARightDeformedCubeFront", Dir::lower_xi(), "ALeftDeformedCubeFront",
       &aligned},
      {"ALeftDeformedCubeTop", Dir::upper_xi(), "ARightDeformedCubeTop",
       &aligned},
      {"ARightDeformedCubeTop", Dir::lower_xi(), "ALeftDeformedCubeTop",
       &aligned},
      {"ALeftDeformedCubeBack", Dir::upper_xi(), "ARightDeformedCubeBack",
       &aligned},
      {"ARightDeformedCubeBack", Dir::lower_xi(), "ALeftDeformedCubeBack",
       &aligned},
      {"ALeftDeformedCubeBottom", Dir::upper_xi(), "ARightDeformedCubeBottom",
       &aligned},
      {"ARightDeformedCubeBottom", Dir::lower_xi(), "ALeftDeformedCubeBottom",
       &aligned},

      // BDeformedCubeCap <-> BLeft side half-wedge angular faces.
      // BCap axes: ξ=+z, η=+y, ζ=-x; all BLeft sides have ξ=+x (lower_xi is
      // the outer angular face for LowerOnly wedges).
      {"BDeformedCubeCap", Dir::upper_xi(), "BLeftDeformedCubeTop", &aligned},
      {"BLeftDeformedCubeTop", Dir::lower_xi(), "BDeformedCubeCap", &aligned},
      {"BDeformedCubeCap", Dir::lower_xi(), "BLeftDeformedCubeBottom",
       &half_turn_about_z},
      {"BLeftDeformedCubeBottom", Dir::lower_xi(), "BDeformedCubeCap",
       &half_turn_about_z},
      {"BDeformedCubeCap", Dir::lower_eta(), "BLeftDeformedCubeFront",
       &orient_bcap_to_bfront},
      {"BLeftDeformedCubeFront", Dir::lower_xi(), "BDeformedCubeCap",
       &orient_bfront_to_bcap},
      {"BDeformedCubeCap", Dir::upper_eta(), "BLeftDeformedCubeBack",
       &orient_bcap_to_bback},
      {"BLeftDeformedCubeBack", Dir::lower_xi(), "BDeformedCubeCap",
       &orient_bback_to_bcap},

      // ADeformedCubeCap <-> ARight side half-wedge angular faces.
      // ACap axes: ξ=-z, η=+y, ζ=+x; all ARight sides have ξ=+x (upper_xi is
      // the outer angular face for UpperOnly wedges).
      {"ADeformedCubeCap", Dir::lower_xi(), "ARightDeformedCubeTop", &aligned},
      {"ARightDeformedCubeTop", Dir::upper_xi(), "ADeformedCubeCap", &aligned},
      {"ADeformedCubeCap", Dir::upper_xi(), "ARightDeformedCubeBottom",
       &half_turn_about_z},
      {"ARightDeformedCubeBottom", Dir::upper_xi(), "ADeformedCubeCap",
       &half_turn_about_z},
      {"ADeformedCubeCap", Dir::lower_eta(), "ARightDeformedCubeFront",
       &orient_acap_to_afront},
      {"ARightDeformedCubeFront", Dir::upper_xi(), "ADeformedCubeCap",
       &orient_afront_to_acap},
      {"ADeformedCubeCap", Dir::upper_eta(), "ARightDeformedCubeBack",
       &orient_acap_to_aback},
      {"ARightDeformedCubeBack", Dir::upper_xi(), "ADeformedCubeCap",
       &orient_aback_to_acap},

      // BLeft side half-wedge <-> BLeft side half-wedge angular (η)
      // connections.
      // All blocks have ξ→x and ζ→radial; transverse coords at every η-face are
      // aligned, so use aligned orientation for all pairs.
      // Top.lower_eta <-> Front.upper_eta  [y<0, z>0 edge]
      {"BLeftDeformedCubeTop", Dir::lower_eta(), "BLeftDeformedCubeFront",
       &aligned},
      {"BLeftDeformedCubeFront", Dir::upper_eta(), "BLeftDeformedCubeTop",
       &aligned},
      // Top.upper_eta <-> Back.lower_eta   [y>0, z>0 edge]
      {"BLeftDeformedCubeTop", Dir::upper_eta(), "BLeftDeformedCubeBack",
       &aligned},
      {"BLeftDeformedCubeBack", Dir::lower_eta(), "BLeftDeformedCubeTop",
       &aligned},
      // Bottom.lower_eta <-> Back.upper_eta [y>0, z<0 edge]
      {"BLeftDeformedCubeBottom", Dir::lower_eta(), "BLeftDeformedCubeBack",
       &aligned},
      {"BLeftDeformedCubeBack", Dir::upper_eta(), "BLeftDeformedCubeBottom",
       &aligned},
      // Bottom.upper_eta <-> Front.lower_eta [y<0, z<0 edge]
      {"BLeftDeformedCubeBottom", Dir::upper_eta(), "BLeftDeformedCubeFront",
       &aligned},
      {"BLeftDeformedCubeFront", Dir::lower_eta(), "BLeftDeformedCubeBottom",
       &aligned},

      // ARight side half-wedge <-> ARight side half-wedge angular (η)
      // connections. Same axes as BLeft → aligned.
      {"ARightDeformedCubeTop", Dir::lower_eta(), "ARightDeformedCubeFront",
       &aligned},
      {"ARightDeformedCubeFront", Dir::upper_eta(), "ARightDeformedCubeTop",
       &aligned},
      {"ARightDeformedCubeTop", Dir::upper_eta(), "ARightDeformedCubeBack",
       &aligned},
      {"ARightDeformedCubeBack", Dir::lower_eta(), "ARightDeformedCubeTop",
       &aligned},
      {"ARightDeformedCubeBottom", Dir::lower_eta(), "ARightDeformedCubeBack",
       &aligned},
      {"ARightDeformedCubeBack", Dir::upper_eta(), "ARightDeformedCubeBottom",
       &aligned},
      {"ARightDeformedCubeBottom", Dir::upper_eta(), "ARightDeformedCubeFront",
       &aligned},
      {"ARightDeformedCubeFront", Dir::lower_eta(), "ARightDeformedCubeBottom",
       &aligned},

      // BRight FOW <-> BRight FOW angular (η) connections.
      // BRight also has ξ and ζ as the two transverse coords at every η-face
      // → aligned orientation for all pairs.
      {"BRightDeformedCubeTop", Dir::upper_eta(), "BRightDeformedCubeFront",
       &aligned},
      {"BRightDeformedCubeFront", Dir::lower_eta(), "BRightDeformedCubeTop",
       &aligned},
      {"BRightDeformedCubeTop", Dir::lower_eta(), "BRightDeformedCubeBack",
       &aligned},
      {"BRightDeformedCubeBack", Dir::upper_eta(), "BRightDeformedCubeTop",
       &aligned},
      {"BRightDeformedCubeBottom", Dir::lower_eta(), "BRightDeformedCubeFront",
       &aligned},
      {"BRightDeformedCubeFront", Dir::upper_eta(), "BRightDeformedCubeBottom",
       &aligned},
      {"BRightDeformedCubeBottom", Dir::upper_eta(), "BRightDeformedCubeBack",
       &aligned},
      {"BRightDeformedCubeBack", Dir::lower_eta(), "BRightDeformedCubeBottom",
       &aligned},

      // ALeft FOW <-> ALeft FOW angular (η) connections.
      // Same argument → aligned.
      {"ALeftDeformedCubeTop", Dir::lower_eta(), "ALeftDeformedCubeFront",
       &aligned},
      {"ALeftDeformedCubeFront", Dir::upper_eta(), "ALeftDeformedCubeTop",
       &aligned},
      {"ALeftDeformedCubeTop", Dir::upper_eta(), "ALeftDeformedCubeBack",
       &aligned},
      {"ALeftDeformedCubeBack", Dir::lower_eta(), "ALeftDeformedCubeTop",
       &aligned},
      {"ALeftDeformedCubeBottom", Dir::lower_eta(), "ALeftDeformedCubeBack",
       &aligned},
      {"ALeftDeformedCubeBack", Dir::upper_eta(), "ALeftDeformedCubeBottom",
       &aligned},
      {"ALeftDeformedCubeBottom", Dir::upper_eta(), "ALeftDeformedCubeFront",
       &aligned},
      {"ALeftDeformedCubeFront", Dir::lower_eta(), "ALeftDeformedCubeBottom",
       &aligned},

      // B-side: filled endcap <-> hollow side
      {"BFilledCylinder", Dir::upper_xi(), "BHollowCylinder", &orient_fs},
      {"BHollowCylinder", Dir::upper_zeta(), "BFilledCylinder", &orient_sf},
      // A-side: filled endcap <-> hollow side
      {"AFilledCylinder", Dir::upper_xi(), "AHollowCylinder", &orient_fs},
      {"AHollowCylinder", Dir::upper_zeta(), "AFilledCylinder", &orient_sf},
      // Midplane seam between the two hollow cylinders
      {"BHollowCylinder", Dir::lower_zeta(), "AHollowCylinder", &orient_zseam},
      {"AHollowCylinder", Dir::lower_zeta(), "BHollowCylinder", &orient_zseam},
  };

  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors_of_all_blocks(
      coordinate_maps.size());
  for (const auto& edge : edges) {
    const auto self_it = tag_to_index.find(edge.self_tag);
    const auto neighbor_it = tag_to_index.find(edge.neighbor_tag);
    if (self_it == tag_to_index.end() or neighbor_it == tag_to_index.end()) {
      continue;
    }
    neighbors_of_all_blocks[self_it->second].emplace(
        edge.self_direction,
        BlockNeighbors<3>(neighbor_it->second, *edge.orientation));
  }

  // Non-conforming interface between SphericalShell and the outer cylinders.
  // SphericalShell lower_xi abuts the four outer cylinder faces: filled
  // cylinders via upper_zeta, hollow cylinders via upper_xi.
  {
    const OrientationMap<3> sphere_to_filled{
        std::array<Dir, 3>{Dir::upper_zeta(), Dir::self(), Dir::self()}};
    const OrientationMap<3> sphere_to_side{
        std::array<Dir, 3>{Dir::upper_xi(), Dir::self(), Dir::self()}};
    const size_t c = tag_to_index.at("SphericalShell");
    const size_t bf = tag_to_index.at("BFilledCylinder");
    const size_t bh = tag_to_index.at("BHollowCylinder");
    const size_t ah = tag_to_index.at("AHollowCylinder");
    const size_t af = tag_to_index.at("AFilledCylinder");
    neighbors_of_all_blocks[c].emplace(
        Dir::lower_xi(), BlockNeighbors<3>{{bf, bh, ah, af},
                                           {{bf, sphere_to_filled},
                                            {bh, sphere_to_side},
                                            {ah, sphere_to_side},
                                            {af, sphere_to_filled}},
                                           false});
    neighbors_of_all_blocks[bf].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_filled.inverse_map()}}, false});
    neighbors_of_all_blocks[af].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_filled.inverse_map()}}, false});
    neighbors_of_all_blocks[bh].emplace(
        Dir::upper_xi(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_side.inverse_map()}}, false});
    neighbors_of_all_blocks[ah].emplace(
        Dir::upper_xi(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_side.inverse_map()}}, false});
  }

  // Non-conforming interfaces between deformed cubes and cylinders.
  // Filled cylinders: BFilled.lower_zeta <-> BDeformedCubeCap.upper_zeta
  //                   AFilled.lower_zeta <-> ADeformedCubeCap.upper_zeta
  // Hollow cylinders: BHollow.lower_xi   <-> {BLeft*,BRight*}.upper_zeta
  //                   AHollow.lower_xi   <-> {ALeft*,ARight*}.upper_zeta
  //
  // Orientation convention: O stored at block A for neighbor B at face d_A
  // satisfies:  interface face of B = -O.mapped(d_A).
  //
  // cap_to_filled: cap at upper_zeta, O[2]=upper_zeta -> interface=-upper_zeta
  //   = lower_zeta of cylinder. Same map applies from the cylinder side
  //   (cylinder at lower_zeta, O[2]=upper_zeta -> interface=upper_zeta of cap).
  // cube_to_hollow: cube at upper_zeta, O[2]=upper_xi -> interface=lower_xi of
  //   hollow cylinder.
  // hollow_to_cube: hollow at lower_xi, O[0]=upper_zeta -> interface=upper_zeta
  //   of deformed cube.
  {
    const OrientationMap<3> cap_to_filled{
        std::array<Dir, 3>{Dir::self(), Dir::self(), Dir::upper_zeta()}};
    const OrientationMap<3> cube_to_hollow{
        std::array<Dir, 3>{Dir::self(), Dir::self(), Dir::upper_xi()}};
    const OrientationMap<3> hollow_to_cube{
        std::array<Dir, 3>{Dir::upper_zeta(), Dir::self(), Dir::self()}};

    const size_t bf = tag_to_index.at("BFilledCylinder");
    const size_t bh = tag_to_index.at("BHollowCylinder");
    const size_t ah = tag_to_index.at("AHollowCylinder");
    const size_t af = tag_to_index.at("AFilledCylinder");
    const size_t bcap = tag_to_index.at("BDeformedCubeCap");
    const size_t acap = tag_to_index.at("ADeformedCubeCap");

    // update comment
    // BFilledCylinder.lower_zeta <-> BDeformedCubeCap.upper_zeta (1-to-1).
    // BFilled is physically on the B-star (-x) side.
    neighbors_of_all_blocks[bf].emplace(
        Dir::lower_zeta(),
        BlockNeighbors<3>{{bcap}, {{bcap, cap_to_filled}}, false});
    neighbors_of_all_blocks[bcap].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{
            {bf, bh}, {{bf, cap_to_filled}, {bh, cube_to_hollow}}, false});

    // update comment
    // AFilledCylinder.lower_zeta <-> ADeformedCubeCap.upper_zeta (1-to-1).
    // AFilled is physically on the A-star (+x) side.
    neighbors_of_all_blocks[af].emplace(
        Dir::lower_zeta(),
        BlockNeighbors<3>{{acap}, {{acap, cap_to_filled}}, false});
    neighbors_of_all_blocks[acap].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{
            {af, ah}, {{af, cap_to_filled}, {ah, cube_to_hollow}}, false});

    // BHollowCylinder.lower_xi <-> {BLeft*, BRight*}.upper_zeta (1-to-8).
    // BHollow is physically on the B-star (-x) side.
    const size_t blf = tag_to_index.at("BLeftDeformedCubeFront");
    const size_t blt = tag_to_index.at("BLeftDeformedCubeTop");
    const size_t blb = tag_to_index.at("BLeftDeformedCubeBack");
    const size_t blbo = tag_to_index.at("BLeftDeformedCubeBottom");
    const size_t brf = tag_to_index.at("BRightDeformedCubeFront");
    const size_t brt = tag_to_index.at("BRightDeformedCubeTop");
    const size_t brb = tag_to_index.at("BRightDeformedCubeBack");
    const size_t brbo = tag_to_index.at("BRightDeformedCubeBottom");
    neighbors_of_all_blocks[bh].emplace(
        Dir::lower_xi(),
        BlockNeighbors<3>{{blf, blt, blb, blbo, brf, brt, brb, brbo, bcap},
                          {{blf, hollow_to_cube},
                           {blt, hollow_to_cube},
                           {blb, hollow_to_cube},
                           {blbo, hollow_to_cube},
                           {brf, hollow_to_cube},
                           {brt, hollow_to_cube},
                           {brb, hollow_to_cube},
                           {brbo, hollow_to_cube},
                           {bcap, hollow_to_cube}},
                          false});
    for (const size_t idx : {blf, blt, blb, blbo, brf, brt, brb, brbo}) {
      neighbors_of_all_blocks[idx].emplace(
          Dir::upper_zeta(),
          BlockNeighbors<3>{{bh}, {{bh, cube_to_hollow}}, false});
    }

    // AHollowCylinder.lower_xi <-> {ALeft*, ARight*}.upper_zeta (1-to-8).
    // AHollow is physically on the A-star (+x) side.
    const size_t alf = tag_to_index.at("ALeftDeformedCubeFront");
    const size_t alt = tag_to_index.at("ALeftDeformedCubeTop");
    const size_t alb = tag_to_index.at("ALeftDeformedCubeBack");
    const size_t albo = tag_to_index.at("ALeftDeformedCubeBottom");
    const size_t arf = tag_to_index.at("ARightDeformedCubeFront");
    const size_t art = tag_to_index.at("ARightDeformedCubeTop");
    const size_t arb = tag_to_index.at("ARightDeformedCubeBack");
    const size_t arbo = tag_to_index.at("ARightDeformedCubeBottom");
    neighbors_of_all_blocks[ah].emplace(
        Dir::lower_xi(),
        BlockNeighbors<3>{{alf, alt, alb, albo, arf, art, arb, arbo, acap},
                          {{alf, hollow_to_cube},
                           {alt, hollow_to_cube},
                           {alb, hollow_to_cube},
                           {albo, hollow_to_cube},
                           {arf, hollow_to_cube},
                           {art, hollow_to_cube},
                           {arb, hollow_to_cube},
                           {arbo, hollow_to_cube},
                           {acap, hollow_to_cube}},
                          false});
    for (const size_t idx : {alf, alt, alb, albo, arf, art, arb, arbo}) {
      neighbors_of_all_blocks[idx].emplace(
          Dir::upper_zeta(),
          BlockNeighbors<3>{{ah}, {{ah, cube_to_hollow}}, false});
    }
  }

  // ---- No excision spheres: neutron stars occupy the full inner volume ----
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  // ---- Assemble blocks with the correct topology per block type ----
  std::vector<Block<3>> blocks{};
  blocks.reserve(number_of_blocks_);
  for (size_t i = 0; i < coordinate_maps.size(); ++i) {
    // clang-format off
    const std::array<domain::Topology, 3> topology =
        i == 26                ? domain::topologies::spherical_shell
        : (i == 22 or i == 25) ? domain::topologies::full_cylinder
        : (i == 23 or i == 24) ? domain::topologies::cylindrical_shell
        :                        domain::topologies::hypercube<3>;
    // clang-format on
    blocks.emplace_back(std::move(coordinate_maps[i]), i,
                        std::move(neighbors_of_all_blocks[i]), block_names_[i],
                        topology);
  }

  Domain<3> domain{std::move(blocks), std::move(excision_spheres),
                   block_groups_};

  if (not time_dependence_->is_none()) {
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps_grid_to_inertial =
            time_dependence_->block_maps_grid_to_inertial(number_of_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        block_maps_grid_to_distorted =
            time_dependence_->block_maps_grid_to_distorted(number_of_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        block_maps_distorted_to_inertial =
            time_dependence_->block_maps_distorted_to_inertial(
                number_of_blocks_);
    for (size_t block_id = 0; block_id < number_of_blocks_; ++block_id) {
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(block_maps_grid_to_inertial[block_id]),
          std::move(block_maps_grid_to_distorted[block_id]),
          std::move(block_maps_distorted_to_inertial[block_id]));
    }
  }
  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
BinaryNeutronStars::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks_};
  // Outer boundary: upper_xi face of SphericalShell (block 26).
  // Neutron stars are not excised, so there is no inner boundary.
  boundary_conditions[26][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> BinaryNeutronStars::initial_extents() const {
  return initial_grid_points_;
}

std::vector<std::array<size_t, 3>>
BinaryNeutronStars::initial_refinement_levels() const {
  return initial_refinement_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
BinaryNeutronStars::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators

// Array Index: [B4,(L0I0,L0I0,L0I0)]
// Array Index: [B5,(L0I0,L0I0,L0I0)]
// Array Index: [B6,(L0I0,L0I0,L0I0)]
// Array Index: [B7,(L0I0,L0I0,L0I0)]
// Array Index: [B8,(L0I0,L0I0,L0I0)]
// Array Index: [B9,(L0I0,L0I0,L0I0)]
// Array Index: [B10,(L0I0,L0I0,L0I0)]
// Array Index: [B11,(L0I0,L0I0,L0I0)]
// Array Index: [B13,(L0I0,L0I0,L0I0)]
// Array Index: [B14,(L0I0,L0I0,L0I0)]
// Array Index: [B15,(L0I0,L0I0,L0I0)]
// Array Index: [B16,(L0I0,L0I0,L0I0)]
// Array Index: [B17,(L0I0,L0I0,L0I0)]
// Array Index: [B18,(L0I0,L0I0,L0I0)]
// Array Index: [B19,(L0I0,L0I0,L0I0)]
// Array Index: [B20,(L0I0,L0I0,L0I0)]
// Array Index: [B21,(L0I0,L0I0,L0I0)]
// Array Index: [B22,(L0I0,L0I0,L0I0)]
// Array Index: [B25,(L0I0,L0I0,L0I0)]
// something is broken for the neighbors between the blocks and the cylinders
// ask claude about it
// ask claude to make test for physical seperation for non-hypercubes?
