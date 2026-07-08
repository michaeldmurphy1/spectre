// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Pill.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/CylindricalFlatEndcapInterior.hpp"
#include "Domain/CoordinateMaps/CylindricalSphericalShell.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/FlatOffsetWedge.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
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
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::creators {
Pill::Pill(
    double center_A, double center_B, double leftmost_x, double rightmost_x,
    double wedge_inner_radius, double wedge_outer_radius,
    double cylinder_outer_radius, double outer_radius,
    std::array<size_t, 4> cube_grid_points,
    std::array<size_t, 4> cube_x_refinement,
    std::array<size_t, 4> cube_yz_refinement,
    size_t wedge_prism_radial_grid_points, size_t wedge_prism_radial_refinement,
    size_t cylinder_radial_grid_points, size_t cylinder_radial_refinement,
    size_t b2_angular_grid_points,
    std::array<size_t, 2> hollow_cylinder_angular_grid_points,
    size_t spherical_shells_radial_refinement,
    size_t spherical_shells_radial_grid_points, size_t spherical_harmonic_l,
    domain::CoordinateMaps::Distribution SphericalShellsRadialDistribution,
    bool bulge,
    std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : center_A_(center_A),
      center_B_(center_B),
      leftmost_x_(leftmost_x),
      rightmost_x_(rightmost_x),
      wedge_inner_radius_(wedge_inner_radius),
      wedge_outer_radius_(wedge_outer_radius),
      cylinder_outer_radius_(cylinder_outer_radius),
      outer_radius_(outer_radius),
      cube_grid_points_(cube_grid_points),
      cube_x_refinement_(cube_x_refinement),
      cube_yz_refinement_(cube_yz_refinement),
      wedge_prism_radial_grid_points_(wedge_prism_radial_grid_points),
      wedge_prism_radial_refinement_(wedge_prism_radial_refinement),
      cylinder_radial_grid_points_(cylinder_radial_grid_points),
      cylinder_radial_refinement_(cylinder_radial_refinement),
      b2_angular_grid_points_(b2_angular_grid_points),
      hollow_cylinder_angular_grid_points_(hollow_cylinder_angular_grid_points),
      spherical_shells_radial_refinement_(spherical_shells_radial_refinement),
      spherical_shells_radial_grid_points_(spherical_shells_radial_grid_points),
      spherical_harmonic_l_(spherical_harmonic_l),
      SphericalShellsRadialDistribution_(SphericalShellsRadialDistribution),
      bulge_(bulge),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      time_dependent_options_(std::move(time_dependent_options)) {
  if (center_A_ <= 0.0) {
    PARSE_ERROR(context, "CenterA must be positive, got CenterA="
                             << center_A_ << ". Set CenterA > 0.");
  }
  if (center_B_ >= 0.0) {
    PARSE_ERROR(context, "CenterB must be negative, got CenterB="
                             << center_B_ << ". Set CenterB < 0.");
  }
  if (std::abs(center_A_) > std::abs(center_B_)) {
    PARSE_ERROR(context, "|CenterA| must be <= |CenterB|, got |CenterA|="
                             << std::abs(center_A_)
                             << " > |CenterB|=" << std::abs(center_B_)
                             << ". Increase |CenterB| or decrease CenterA.");
  }
  if (leftmost_x_ >= center_B_) {
    PARSE_ERROR(context, "LeftmostX must be less than CenterB, got LeftmostX="
                             << leftmost_x_ << " and CenterB=" << center_B_
                             << ". Set LeftmostX < CenterB.");
  }
  if (rightmost_x_ <= center_A_) {
    PARSE_ERROR(context,
                "RightmostX must be greater than CenterA, got RightmostX="
                    << rightmost_x_ << " and CenterA=" << center_A_
                    << ". Set RightmostX > CenterA.");
  }
  if (wedge_inner_radius_ <= 0.0) {
    PARSE_ERROR(context, "WedgeInnerRadius must be positive, got "
                             << wedge_inner_radius_ << ".");
  }
  if (wedge_outer_radius_ <= 0.0) {
    PARSE_ERROR(context, "WedgeOuterRadius must be positive, got "
                             << wedge_outer_radius_ << ".");
  }
  if (outer_radius_ <= cylinder_outer_radius_) {
    PARSE_ERROR(
        context,
        "OuterRadius must be greater than CylinderOuterRadius, got "
            << outer_radius_ << " <= " << cylinder_outer_radius_
            << ". Increase OuterRadius or decrease CylinderOuterRadius.");
  }
  if (cylinder_outer_radius_ <= wedge_outer_radius_) {
    PARSE_ERROR(
        context,
        "CylinderOuterRadius must be greater than WedgeOuterRadius, got "
            << cylinder_outer_radius_ << " <= " << wedge_outer_radius_
            << ". Increase CylinderOuterRadius or decrease WedgeOuterRadius.");
  }
  if (wedge_outer_radius_ <= wedge_inner_radius_) {
    PARSE_ERROR(context,
                "WedgeOuterRadius must be greater than WedgeInnerRadius, got "
                    << wedge_outer_radius_ << " <= " << wedge_inner_radius_
                    << ". Increase WedgeOuterRadius or decrease "
                       "WedgeInnerRadius.");
  }
  // Check CylindricalFlatEndcapInterior validity condition t_sphere > 1.
  // With proj_center = center_two = origin, t_sphere = R2/sqrt(w^2 + c1z^2)
  // where c1z is the x-coordinate of the flat disk center (leftmost_x_ on the
  // B side, rightmost_x_ on the A side).  t_sphere > 1 requires
  // R2^2 > w^2 + c1z^2. Check whichever side has the larger |c1z|.
  const double c1z_max_sq =
      std::max(leftmost_x_ * leftmost_x_, rightmost_x_ * rightmost_x_);
  if (cylinder_outer_radius_ * cylinder_outer_radius_ <=
      wedge_outer_radius_ * wedge_outer_radius_ + c1z_max_sq) {
    PARSE_ERROR(
        context,
        "CylindricalFlatEndcapInterior (the B2s' map) requires "
        "CylinderOuterRadius^2 > WedgeOuterRadius^2 + max(LeftmostX^2, "
        "RightmostX^2), but got "
            << cylinder_outer_radius_ * cylinder_outer_radius_
            << " <= " << wedge_outer_radius_ * wedge_outer_radius_ + c1z_max_sq
            << " (CylinderOuterRadius=" << cylinder_outer_radius_
            << ", WedgeOuterRadius=" << wedge_outer_radius_
            << ", max|x|=" << sqrt(c1z_max_sq)
            << "). Increase CylinderOuterRadius or bring LeftmostX/RightmostX "
               "closer to zero (decrease |LeftmostX| and |RightmostX|).");
  }
  // CylindricalFlatEndcapInterior requires radius_two/radius_one in [1/10,10].
  // With the new setup radius_one = wedge_outer_radius, radius_two =
  // cylinder_outer_radius, so we need cylinder_outer_radius <= 10 *
  // wedge_outer_radius (the other direction is already ensured above).
  if (cylinder_outer_radius_ > 10.0 * wedge_outer_radius_) {
    PARSE_ERROR(context,
                "CylindricalFlatEndcapInterior radius ratio condition "
                "violated: CylinderOuterRadius > 10 * WedgeOuterRadius. "
                "Increase WedgeOuterRadius or decrease CylinderOuterRadius.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  if (bulge_) {
    // Compute effective radii for the two FOW sphere groups (they must be
    // different radii if center_A_ != center_B_ for their slices to match at
    // x=0)
    const double r_inner = wedge_inner_radius_;
    double R_B = wedge_outer_radius_;
    double R_A = wedge_outer_radius_;
    if (std::abs(center_B_) > center_A_) {
      R_A = std::sqrt(R_B * R_B -
                      (center_B_ * center_B_ - center_A_ * center_A_));
    } else if (center_A_ > std::abs(center_B_)) {
      R_B = std::sqrt(R_A * R_A -
                      (center_A_ * center_A_ - center_B_ * center_B_));
    }
    // FOW constraint D^2 + 2L^2 < R^2; L = r_inner/sqrt(2), so
    // D^2 + r_inner^2 < R^2.
    // BLeft: D = center_B_ - leftmost_x_
    const double d_bleft = center_B_ - leftmost_x_;  // > 0
    if (d_bleft * d_bleft + r_inner * r_inner >= R_B * R_B) {
      PARSE_ERROR(context,
                  "Bulge=true FlatOffsetWedge BLeft constraint "
                  "D_BLeft^2 + r^2 < R_B^2 violated: "
                      << d_bleft * d_bleft + r_inner * r_inner
                      << " >= R_B^2 = " << R_B * R_B
                      << ". Decrease WedgeInnerRadius, bring LeftmostX closer "
                         "to CenterB, or increase WedgeOuterRadius.");
    }
    // ARight: D = rightmost_x_ - center_A_
    const double d_aright = rightmost_x_ - center_A_;
    if (d_aright * d_aright + r_inner * r_inner >= R_A * R_A) {
      PARSE_ERROR(context,
                  "Bulge=true FlatOffsetWedge ARight constraint "
                  "D_ARight^2 + r^2 < R_A^2 violated: "
                      << d_aright * d_aright + r_inner * r_inner
                      << " >= R_A^2 = " << R_A * R_A
                      << ". Decrease WedgeInnerRadius, bring RightmostX closer "
                         "to CenterA, or increase WedgeOuterRadius.");
    }
    // BRight: D = |cB|  ->  cB^2 + r^2 < R_B^2
    if (center_B_ * center_B_ + r_inner * r_inner >= R_B * R_B) {
      PARSE_ERROR(
          context,
          "Bulge=true FlatOffsetWedge BRight constraint "
          "cB^2 + r^2 < R_B^2 violated: "
              << center_B_ * center_B_ + r_inner * r_inner
              << " >= " << R_B * R_B
              << ". Decrease WedgeInnerRadius or the star separations, or "
                 "increase WedgeOuterRadius.");
    }
    // ALeft: D = cA  ->  cA^2 + r^2 < R_A^2
    if (center_A_ * center_A_ + r_inner * r_inner >= R_A * R_A) {
      PARSE_ERROR(
          context,
          "Bulge=true FlatOffsetWedge ALeft constraint "
          "cA^2 + r^2 < R_A^2 violated: "
              << center_A_ * center_A_ + r_inner * r_inner
              << " >= " << R_A * R_A
              << ". Decrease WedgeInnerRadius or the star separations, or "
                 "increase WedgeOuterRadius.");
    }
    // UniformCylindricalSide requires the upper cut of the inner sphere to
    // subtend a polar angle < 72 deg from its north pole, i.e.
    // (sphere_center_to_cut_plane) / R > cos(72 deg) ~ 0.309.
    // For BLeft the cut height is d_bleft = center_B_ - leftmost_x_;
    // for ARight the cut height is d_aright = rightmost_x_ - center_A_.
    if (d_bleft / R_B <= cos(std::numbers::pi * 0.4)) {
      PARSE_ERROR(context,
                  "Bulge=true UniformCylindricalSide BLeft constraint "
                  "(center_B - LeftmostX)/R_B > cos(72 deg) ~ 0.309 violated: "
                      << d_bleft << "/" << R_B << " = " << d_bleft / R_B
                      << ". Decrease LeftmostX (move it farther from CenterB), "
                         "or decrease WedgeOuterRadius.");
    }
    if (d_aright / R_A <= cos(std::numbers::pi * 0.4)) {
      PARSE_ERROR(
          context,
          "Bulge=true UniformCylindricalSide ARight constraint "
          "(RightmostX - center_A)/R_A > cos(72 deg) ~ 0.309 violated: "
              << d_aright << "/" << R_A << " = " << d_aright / R_A
              << ". Increase RightmostX (move it farther from CenterA), "
                 "or decrease WedgeOuterRadius.");
    }
    const auto far_corner_of_bulge = [r_inner](
                                         const double distance_to_center,
                                         const double wedge_outer_radius) {
      return sqrt(square(distance_to_center + r_inner) +
                  square(wedge_outer_radius) - square(r_inner));
    };
    const double fc_aright = far_corner_of_bulge(center_A_, R_A);
    const double fc_bleft_far = far_corner_of_bulge(std::abs(center_B_), R_B);
    const double fc_bleft_near = far_corner_of_bulge(d_bleft, R_B);
    const double fc_aright_near = far_corner_of_bulge(d_aright, R_A);
    if (fc_aright >= cylinder_outer_radius_ or
        fc_bleft_far >= cylinder_outer_radius_ or
        fc_bleft_near >= cylinder_outer_radius_ or
        fc_aright_near >= cylinder_outer_radius_) {
      PARSE_ERROR(
          context,
          "Bulge=true: the far corner of a FlatOffsetWedge block lies outside "
          "the outer cylinder (CylinderOuterRadius="
              << cylinder_outer_radius_ << "). Far-corner radii: ALeft="
              << fc_aright << ", BRight=" << fc_bleft_far
              << ", BLeft=" << fc_bleft_near << ", ARight=" << fc_aright_near
              << ". Decrease WedgeOuterRadius or WedgeInnerRadius, bring "
                 "LeftmostX/RightmostX closer to zero (decrease their "
                 "magnitude), or increase CylinderOuterRadius.");
    }
  } else {
    const double fc_a =
        sqrt(square(rightmost_x_) + square(wedge_outer_radius_));
    const double fc_b = sqrt(square(leftmost_x_) + square(wedge_outer_radius_));
    if (fc_a >= cylinder_outer_radius_ or fc_b >= cylinder_outer_radius_) {
      PARSE_ERROR(
          context,
          "Bulge=false: the far corner of a cubed-cylinder block lies outside "
          "the outer cylinder (CylinderOuterRadius="
              << cylinder_outer_radius_
              << "). Far-corner radii: A-side=" << fc_a << ", B-side=" << fc_b
              << ". Decrease WedgeOuterRadius, bring LeftmostX/RightmostX "
                 "closer to zero (decrease their magnitude), or increase "
                 "CylinderOuterRadius.");
    }
  }

  number_of_blocks_ = 27;

  // Create block names and groups
  block_names_.reserve(number_of_blocks_);
  auto add_cubed_cylinder_name = [this](const std::string& prefix) {
    for (const std::string& where :
         {"Center"s, "Front"s, "Top"s, "Back"s, "Bottom"s}) {
      const std::string name =
          std::string(prefix).append("CubedCylinder").append(where);
      block_names_.emplace_back(name);
      block_groups_["CubedCylinders"].insert(name);
    }
  };

  // 4 cubed cylinders
  add_cubed_cylinder_name("BLeft");   // 0 - 4
  add_cubed_cylinder_name("BRight");  // 5 - 9
  add_cubed_cylinder_name("ALeft");   // 10-14
  add_cubed_cylinder_name("ARight");  // 15-19

  // 6 cylinders, between the deformed cubes and spherical shells
  block_names_.emplace_back("BFilledCylinder");
  block_groups_["Cylinders"].insert("BFilledCylinder");
  block_names_.emplace_back("BLeftHollowCylinder");
  block_groups_["Cylinders"].insert("BLeftHollowCylinder");
  block_names_.emplace_back("BRightHollowCylinder");
  block_groups_["Cylinders"].insert("BRightHollowCylinder");
  block_names_.emplace_back("ALeftHollowCylinder");
  block_groups_["Cylinders"].insert("ALeftHollowCylinder");
  block_names_.emplace_back("ARightHollowCylinder");
  block_groups_["Cylinders"].insert("ARightHollowCylinder");
  block_names_.emplace_back("AFilledCylinder");
  block_groups_["Cylinders"].insert("AFilledCylinder");

  // 1 spherical shell
  block_names_.emplace_back("SphericalShell");
  block_groups_["SphericalShells"].insert("SphericalShell");

  // initial grid points and refinement
  initial_grid_points_.resize(number_of_blocks_);
  initial_refinement_.resize(number_of_blocks_);

  // Blocks 0-19: 4 sets of cubed-cylinders (5 blocks each, stride 5).
  // Bulge=false: wedge prisms have xi piercing curved surface, eta "angular"
  //   along the curve, zeta along the extruded (axial) dimension.
  // Bulge=true: FOW blocks have xi along the axial dimension, eta "angular",
  //   zeta piercing the curved (sphere) surface.
  for (size_t g = 0; g < 4; ++g) {
    initial_grid_points_[5 * g] = make_array<3>(gsl::at(cube_grid_points_, g));
    initial_refinement_[5 * g] = {gsl::at(cube_x_refinement_, g),
                                  gsl::at(cube_yz_refinement_, g),
                                  gsl::at(cube_yz_refinement_, g)};
    for (size_t j = 1; j < 5; ++j) {
      if (bulge_) {
        // FOW: xi=axial, eta=angular, zeta=radial
        initial_grid_points_[5 * g + j] = {gsl::at(cube_grid_points_, g),
                                           gsl::at(cube_grid_points_, g),
                                           wedge_prism_radial_grid_points_};
        initial_refinement_[5 * g + j] = {gsl::at(cube_x_refinement_, g),
                                          gsl::at(cube_yz_refinement_, g),
                                          wedge_prism_radial_refinement_};
      } else {
        // 2D wedge "prism"
        initial_grid_points_[5 * g + j] = {wedge_prism_radial_grid_points_,
                                           gsl::at(cube_grid_points_, g),
                                           gsl::at(cube_grid_points_, g)};
        initial_refinement_[5 * g + j] = {wedge_prism_radial_refinement_,
                                          gsl::at(cube_yz_refinement_, g),
                                          gsl::at(cube_x_refinement_, g)};
      }
    }
  }
  // Blocks 20 & 25: filled cylinders (full_cylinder topology).
  // xi (B2, radial) is tied to (B2, angular); both cannot be h-refined. eta
  // must be odd. zeta (I1, axial) uses the cylinder radial parameters.
  // For both, zeta points along positive x, which is also true for the hollow
  // cylinders, so that the angular coordinates of all cylinders align (with
  // DiscreteRotations about x)
  if (b2_angular_grid_points_ % 2 == 0) {
    PARSE_ERROR(context, "B2InitialAngularGridPoints must be odd, but got "
                             << b2_angular_grid_points_ << ".");
  }
  const size_t theta_modes = b2_angular_grid_points_ / 2;
  // set by equating radial/angular modal space sizes for ZernikeB2
  const size_t b2_radial = theta_modes / 2 + 1 + theta_modes % 2;
  for (const size_t i : {20_st, 25_st}) {
    initial_grid_points_[i] = {b2_radial, b2_angular_grid_points_,
                               cylinder_radial_grid_points_};
    initial_refinement_[i] = {0, 0, cylinder_radial_refinement_};
  }
  // Blocks 21 - 24: hollow cylinders (cylindrical_shell topology).
  // xi (I1, radial), eta (S1, angular) cannot be h-refined, must be odd.
  // zeta (I1, axial) uses the cylinder radial parameters.
  const auto [hollow_eta, hollow_zeta] = hollow_cylinder_angular_grid_points_;
  if (hollow_eta % 2 == 0) {
    PARSE_ERROR(context,
                "HollowCylinderInitialAngularGridPoints[0] must be odd, but "
                "got "
                    << hollow_eta << ".");
  }
  for (const size_t i : {21_st, 22_st, 23_st, 24_st}) {
    initial_grid_points_[i] = {cylinder_radial_grid_points_, hollow_eta,
                               hollow_zeta};
    initial_refinement_[i] = {cylinder_radial_refinement_, 0, 0};
  }
  // Block 26: spherical shell (spherical_shell topology).
  // xi (I1, radial)
  // eta (S2Colatitude) and zeta (S2Longitude) cannot be h-refined.
  // Spherepack requires zeta_pts = 2 * eta_pts - 1.
  const size_t colatitude_pts = spherical_harmonic_l_ + 1;
  initial_grid_points_[26] = {spherical_shells_radial_grid_points_,
                              colatitude_pts, 2 * colatitude_pts - 1};
  initial_refinement_[26] = {spherical_shells_radial_refinement_, 0, 0};

  grid_anchors_ =
      bco::create_grid_anchors({center_A_, 0., 0.}, {center_B_, 0., 0.});

  if (time_dependent_options_.has_value()) {
    if (time_dependent_options_->has_distorted_frame_options(
            domain::ObjectLabel::A) or
        time_dependent_options_->has_distorted_frame_options(
            domain::ObjectLabel::B)) {
      PARSE_ERROR(context,
                  "The Pill domain does not support shape maps because the "
                  "objects are filled (no excision surfaces). Set ShapeMapA "
                  "and ShapeMapB to None.");
    }
    time_dependent_options_->build_maps(
        std::array{std::array<double, 3>{center_A_, 0., 0.},
                   std::array<double, 3>{center_B_, 0., 0.}},
        std::nullopt, std::nullopt, std::array<double, 3>{0., 0., 0.},
        std::optional<std::array<double, 2>>{std::nullopt},
        std::optional<std::array<double, 2>>{std::nullopt}, true, true,
        cylinder_outer_radius_, outer_radius_);
  }
}

Domain<3> Pill::create_domain() const {
  using Interval = CoordinateMaps::Interval;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Wedge = CoordinateMaps::Wedge<2>;
  using WedgePrism = CoordinateMaps::ProductOf2Maps<Wedge, Affine>;
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
  const auto wedge_2d_aligned = OrientationMap<2>::create_aligned();

  // naming as starting from upper z axis
  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};

  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  const OrientationMap<3> half_turn_about_z(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});

  const OrientationMap<3> rotate_to_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
      Direction<3>::lower_eta()}};

  const OrientationMap<3> rotate_to_minus_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
      Direction<3>::upper_eta()}};

  const OrientationMap<3> rotate_to_minus_z_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_eta(),
      Direction<3>::lower_zeta()}};

  const OrientationMap<3> half_turn_about_y_minus_z_axis{
      std::array<Direction<3>, 3>{Direction<3>::lower_xi(),
                                  Direction<3>::lower_zeta(),
                                  Direction<3>::lower_eta()}};

  const OrientationMap<3> half_turn_about_y_plus_z_axis{
      std::array<Direction<3>, 3>{Direction<3>::lower_xi(),
                                  Direction<3>::upper_zeta(),
                                  Direction<3>::upper_eta()}};

  const OrientationMap<3> half_turn_about_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_xi(), Direction<3>::upper_eta(),
      Direction<3>::lower_zeta()}};

  // For Bulge=false: Wedges must first apply rotate_to_minus_x_axis, then
  // rotate about x axis
  const OrientationMap<3> rotate_wedge_front{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
      Direction<3>::upper_eta()}};

  const OrientationMap<3> rotate_wedge_top = rotate_to_minus_x_axis;

  const OrientationMap<3> rotate_wedge_back{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_xi(),
      Direction<3>::lower_eta()}};

  const OrientationMap<3> rotate_wedge_bottom{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::lower_eta(),
      Direction<3>::lower_xi()}};

  // For Bulge=true: compute effective sphere radii so that both FOW groups
  // share a conforming sphere at x=0
  double R_B = wedge_outer_radius_;
  double R_A = wedge_outer_radius_;
  if (bulge_) {
    if (std::abs(center_B_) > center_A_) {
      R_A = std::sqrt(R_B * R_B -
                      (center_B_ * center_B_ - center_A_ * center_A_));
    } else if (center_A_ > std::abs(center_B_)) {
      R_B = std::sqrt(R_A * R_A -
                      (center_A_ * center_A_ - center_B_ * center_B_));
    }
  }
  const double L =
      wedge_inner_radius_ / std::sqrt(2.0);  // FOW flat-face half-width

  // Blocks 0 - 19
  if (not bulge_) {
    // Wedges are extruded 2D wedges
    auto add_cubed_cylinder = [this, &coordinate_maps, &block_tags,
                               &rotate_wedge_front, &rotate_wedge_top,
                               &rotate_wedge_back, &rotate_wedge_bottom,
                               &wedge_2d_aligned](const double lower_xi,
                                                  const double upper_xi,
                                                  const std::string& tag) {
      // central cube
      coordinate_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Affine3D{Affine{-1.0, 1.0, lower_xi, upper_xi},
                       Affine{-1.0, 1.0, -wedge_inner_radius_ / sqrt(2),
                              wedge_inner_radius_ / sqrt(2)},
                       Affine{-1.0, 1.0, -wedge_inner_radius_ / sqrt(2),
                              wedge_inner_radius_ / sqrt(2)}}));
      block_tags.push_back(tag + "CentralCube"s);
      // 4 surrounding wedges (Front, Top, Back, Bottom)
      const std::array<const char*, 4> wedge_wheres{"Front", "Top", "Back",
                                                    "Bottom"};
      size_t where_idx = 0;
      for (const auto& rotation : {rotate_wedge_front, rotate_wedge_top,
                                   rotate_wedge_back, rotate_wedge_bottom}) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                WedgePrism{Wedge{wedge_inner_radius_, wedge_outer_radius_, 0.0,
                                 1.0, wedge_2d_aligned, false},
                           Affine{-1.0, 1.0, -upper_xi, -lower_xi}},
                Rotation{rotation}));
        block_tags.push_back(tag + "DeformedCube"s +
                             gsl::at(wedge_wheres, where_idx++));
      }
    };
    add_cubed_cylinder(leftmost_x_, center_B_, "BLeft");
    add_cubed_cylinder(center_B_, 0.0, "BRight");
    add_cubed_cylinder(0.0, center_A_, "ALeft");
    add_cubed_cylinder(center_A_, rightmost_x_, "ARight");
  } else {
    // Bulge=true: FOW blocks
    const double half_r = wedge_inner_radius_ / std::sqrt(2.0);
    const auto add_central_cube_block = [&coordinate_maps, &block_tags, half_r](
                                            const double lower_x,
                                            const double upper_x,
                                            const std::string& tag) {
      coordinate_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Affine3D{Affine{-1.0, 1.0, lower_x, upper_x},
                       Affine{-1.0, 1.0, -half_r, half_r},
                       Affine{-1.0, 1.0, -half_r, half_r}}));
      block_tags.push_back(tag + "CentralCube"s);
    };

    // BLeft (x in [leftmost_x_, center_B_])
    add_central_cube_block(leftmost_x_, center_B_, "BLeft");
    {
      const double d_bl = center_B_ - leftmost_x_;
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_bl{{{rotate_to_minus_y_axis, "Front"},
                       {aligned, "Top"},
                       {rotate_to_y_axis, "Back"},
                       {rotate_to_minus_z_axis, "Bottom"}}};
      for (const auto& [map, where] : fow_rots_bl) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, d_bl, R_B},
                ShiftX{Affine{0.0, d_bl, leftmost_x_, center_B_}, Identity1D{},
                       Identity1D{}},
                Rotation{map}));
        block_tags.push_back("BLeftDeformedCube"s + where);
      }
    }

    // BRight (x in [center_B_, 0])
    add_central_cube_block(center_B_, 0.0, "BRight");
    {
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_br{{{half_turn_about_y_minus_z_axis, "Front"},
                       {half_turn_about_z, "Top"},
                       {half_turn_about_y_plus_z_axis, "Back"},
                       {half_turn_about_y_axis, "Bottom"}}};
      for (const auto& [combined, where] : fow_rots_br) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, std::abs(center_B_), R_B},
                Rotation{combined}));
        block_tags.push_back("BRightDeformedCube"s + where);
      }
    }

    // ALeft (x in [0, center_A_])
    add_central_cube_block(0.0, center_A_, "ALeft");
    {
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_al{{{rotate_to_minus_y_axis, "Front"},
                       {aligned, "Top"},
                       {rotate_to_y_axis, "Back"},
                       {rotate_to_minus_z_axis, "Bottom"}}};
      for (const auto& [map, where] : fow_rots_al) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, center_A_, R_A},
                ShiftX{Affine{0.0, center_A_, 0.0, center_A_}, Identity1D{},
                       Identity1D{}},
                Rotation{map}));
        block_tags.push_back("ALeftDeformedCube"s + where);
      }
    }

    // ARight (x in [center_A_, rightmost_x_])
    add_central_cube_block(center_A_, rightmost_x_, "ARight");
    {
      const double d_ar = rightmost_x_ - center_A_;
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_ar{{{rotate_to_minus_y_axis, "Front"},
                       {aligned, "Top"},
                       {rotate_to_y_axis, "Back"},
                       {rotate_to_minus_z_axis, "Bottom"}}};
      for (const auto& [map, where] : fow_rots_ar) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, d_ar, R_A},
                Rotation{half_turn_about_z},
                ShiftX{Affine{-d_ar, 0.0, center_A_, rightmost_x_},
                       Identity1D{}, Identity1D{}},
                Rotation{map}));
        block_tags.push_back("ARightDeformedCube"s + where);
      }
    }
  }

  // Lambdas for cylinder and sphere-shell blocks

  // ZernikeB2 filled-cylinder (full_cylinder topology).
  const auto add_filled_cylinder =
      [&coordinate_maps, &block_tags, &rotate_to_minus_x_axis,
       &half_turn_about_y_axis, &aligned](
          const CoordinateMaps::CylindricalFlatEndcapInterior& endcap_map,
          const Rotation& rotation_map, const std::string& tag) {
        const bool minus_side =
            rotation_map == Rotation{rotate_to_minus_x_axis};
        coordinate_maps.push_back(
            domain::make_coordinate_map_base<Frame::BlockLogical,
                                             Frame::Inertial>(
                ShiftX{Affine{-1.0, 1.0, 0.0, 1.0}, Identity1D{}, Identity1D{}},
                CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                               Identity1D>{
                    CoordinateMaps::PolarToCartesian{}, Identity1D{}},
                Rotation{minus_side ? half_turn_about_y_axis : aligned},
                endcap_map, rotation_map));
        block_tags.push_back(tag);
      };

  // Bulge=false hollow cylinder: CylindricalSphericalShell.
  const auto add_hollow_cylinder_flat = [this, &coordinate_maps, &block_tags,
                                         &rotate_to_y_axis](
                                            const double x_inner_lower,
                                            const double x_inner_upper,
                                            const double x_outer_lower,
                                            const double x_outer_upper,
                                            const std::string& tag) {
    coordinate_maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::CylindricalSphericalShell{
                x_inner_lower, x_inner_upper, x_outer_lower, x_outer_upper,
                wedge_outer_radius_, cylinder_outer_radius_},
            Rotation{rotate_to_y_axis}));
    block_tags.push_back(tag);
  };

  // Bulge=true hollow cylinder: UniformCylindricalSide.
  const auto add_hollow_cylinder_bulged = [this, &coordinate_maps, &block_tags,
                                           &rotate_to_minus_x_axis,
                                           &half_turn_about_y_axis, &aligned](
                                              const double center,
                                              const double inner_radius,
                                              const double z_inner_upper,
                                              const double z_inner_lower,
                                              const double z_outer_upper,
                                              const double z_outer_lower,
                                              const Rotation& rotation_map,
                                              const std::string& tag) {
    const bool minus_side = rotation_map == Rotation{rotate_to_minus_x_axis};
    coordinate_maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf3Maps<Affine, Identity1D,
                                           CoordinateMaps::Interval>{
                Affine{-1.0, 1.0, 1.0, 2.0}, Identity1D{},
                CoordinateMaps::Interval{-1.0, 1.0, -1.0, 1.0,
                                         CoordinateMaps::Distribution::Linear}},
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                           Identity1D>{
                CoordinateMaps::PolarToCartesian{}, Identity1D{}},
            Rotation{minus_side ? half_turn_about_y_axis : aligned},
            CoordinateMaps::UniformCylindricalSide(
                std::array<double, 3>{0.0, 0.0, center}, make_array<3>(0.0),
                inner_radius, cylinder_outer_radius_, z_inner_upper,
                z_inner_lower, z_outer_upper, z_outer_lower),
            rotation_map));
    block_tags.push_back(tag);
  };

  // S2 spherical-harmonic shell
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

  // Blocks 20-26: filled cylinders, hollow cylinders, shell
  // z_sphere_extent controls how far around the outer sphere the filled
  // cylinders extend (set by where the edges of the inner deformed cubes are,
  // which is bulge dependent)
  const double z_sphere_extent_b_flat =
      cylinder_outer_radius_ * leftmost_x_ /
      std::sqrt(wedge_outer_radius_ * wedge_outer_radius_ +
                leftmost_x_ * leftmost_x_);
  const double z_sphere_extent_a_flat =
      cylinder_outer_radius_ * rightmost_x_ /
      std::sqrt(wedge_outer_radius_ * wedge_outer_radius_ +
                rightmost_x_ * rightmost_x_);

  const double z_sphere_extent_b_bulge =
      -cylinder_outer_radius_ /
      sqrt(1 + (square(R_B) - square(center_B_ - leftmost_x_)) /
                   square(leftmost_x_));
  const double z_sphere_extent_a_bulge =
      cylinder_outer_radius_ /
      sqrt(1 + (square(R_A) - square(rightmost_x_ - center_A_)) /
                   square(rightmost_x_));

  const double z_sphere_extent_b =
      bulge_ ? z_sphere_extent_b_bulge : z_sphere_extent_b_flat;
  const double z_sphere_extent_a =
      bulge_ ? z_sphere_extent_a_bulge : z_sphere_extent_a_flat;

  // Finds the z value for an equal arc between two z slices
  const auto get_x_midpoint = [this](const double lower, const double upper) {
    return cylinder_outer_radius_ *
           sin(0.5 * (asin(upper / cylinder_outer_radius_) +
                      asin(lower / cylinder_outer_radius_)));
  };

  // Block 20: BFilledCylinder
  add_filled_cylinder(CoordinateMaps::CylindricalFlatEndcapInterior(
                          std::array<double, 3>{0.0, 0.0, leftmost_x_},
                          std::array<double, 3>{0.0, 0.0, 0.0},
                          std::array<double, 3>{0.0, 0.0, 0.0},
                          z_sphere_extent_b, cylinder_outer_radius_),
                      Rotation{rotate_to_x_axis}, "BFilledCylinder");

  // Blocks 21-24: hollow cylinders.
  if (not bulge_) {
    const double central_midpoint =
        get_x_midpoint(z_sphere_extent_b, z_sphere_extent_a);
    const double b_midpoint =
        get_x_midpoint(z_sphere_extent_b, central_midpoint);
    const double a_midpoint =
        get_x_midpoint(central_midpoint, z_sphere_extent_a);

    add_hollow_cylinder_flat(leftmost_x_, center_B_, z_sphere_extent_b_flat,
                             b_midpoint, "BLeftHollowCylinder");
    add_hollow_cylinder_flat(center_B_, 0.0, b_midpoint, central_midpoint,
                             "BRightHollowCylinder");
    add_hollow_cylinder_flat(0.0, center_A_, central_midpoint, a_midpoint,
                             "ALeftHollowCylinder");
    add_hollow_cylinder_flat(center_A_, rightmost_x_, a_midpoint,
                             z_sphere_extent_a, "ARightHollowCylinder");
  } else {
    // Having central_midpoint be shifted causes issues with the
    // UniformCylindricalSide map
    const double central_midpoint = 0.0;
    const double b_midpoint =
        get_x_midpoint(z_sphere_extent_b, central_midpoint);
    const double a_midpoint =
        get_x_midpoint(central_midpoint, z_sphere_extent_a);
    // UniformCylindricalSide also requires the outer-sphere upper cut to
    // subtend a polar angle < 72 deg from its north pole, i.e.
    // midpoint / CylinderOuterRadius > cos(72 deg) ~ 0.309.
    if (-b_midpoint / cylinder_outer_radius_ <= cos(std::numbers::pi * 0.4)) {
      ERROR(
          "Pill: Bulge=true UniformCylindricalSide BRight outer constraint "
          "-b_midpoint / CylinderOuterRadius > cos(72 deg) ~ 0.309 "
          "violated: "
          << -b_midpoint << " / " << cylinder_outer_radius_ << " = "
          << -b_midpoint / cylinder_outer_radius_
          << ". Try decreasing the star separation or WedgeOuterRadius.");
    }
    if (a_midpoint / cylinder_outer_radius_ <= cos(std::numbers::pi * 0.4)) {
      ERROR(
          "Pill: Bulge=true UniformCylindricalSide ALeft outer constraint "
          "a_midpoint / CylinderOuterRadius > cos(72 deg) ~ 0.309 "
          "violated: "
          << a_midpoint << " / " << cylinder_outer_radius_ << " = "
          << a_midpoint / cylinder_outer_radius_
          << ". Try decreasing the star separation or WedgeOuterRadius.");
    }
    add_hollow_cylinder_bulged(
        -center_B_, R_B, -leftmost_x_, -center_B_, -z_sphere_extent_b,
        -b_midpoint, Rotation{rotate_to_minus_x_axis}, "BLeftHollowCylinder");
    add_hollow_cylinder_bulged(
        -center_B_, R_B, -center_B_, 0.0, -b_midpoint, central_midpoint,
        Rotation{rotate_to_minus_x_axis}, "BRightHollowCylinder");
    add_hollow_cylinder_bulged(center_A_, R_A, center_A_, 0.0, a_midpoint,
                               central_midpoint, Rotation{rotate_to_x_axis},
                               "ALeftHollowCylinder");
    add_hollow_cylinder_bulged(
        center_A_, R_A, rightmost_x_, center_A_, z_sphere_extent_a, a_midpoint,
        Rotation{rotate_to_x_axis}, "ARightHollowCylinder");
  }

  // Block 25: AFilledCylinder
  add_filled_cylinder(CoordinateMaps::CylindricalFlatEndcapInterior(
                          std::array<double, 3>{0.0, 0.0, -rightmost_x_},
                          std::array<double, 3>{0.0, 0.0, 0.0},
                          std::array<double, 3>{0.0, 0.0, 0.0},
                          -z_sphere_extent_a, cylinder_outer_radius_),
                      Rotation{rotate_to_minus_x_axis}, "AFilledCylinder");

  // Block 26: SphericalShell
  add_sphere_shell(cylinder_outer_radius_, outer_radius_, "SphericalShell");

  // Neighbor graph
  using Dir = Direction<3>;

  // Directed neighbor graph keyed by block tag:
  // {self tag, self direction, neighbor tag, orientation of neighbor}.
  struct EdgeSpec {
    std::string self_tag;
    Dir self_direction;
    std::string neighbor_tag;
    OrientationMap<3> orientation;
    bool are_conforming = true;
  };
  std::vector<EdgeSpec> edges;

  std::unordered_map<std::string, size_t> tag_to_index{};
  for (size_t i = 0; i < block_tags.size(); ++i) {
    tag_to_index.emplace(block_tags[i], i);
  }

  // Central cubes are always conforming and use aligned orientation.
  for (const auto& [a, b] : {std::pair{"BLeft"s, "BRight"s},
                             {"BRight"s, "ALeft"s},
                             {"ALeft"s, "ARight"s}}) {
    edges.push_back(
        {a + "CentralCube", Dir::upper_xi(), b + "CentralCube", aligned});
    edges.push_back(
        {b + "CentralCube", Dir::lower_xi(), a + "CentralCube", aligned});
  }

  if (not bulge_) {
    // Bulge=false: WedgePrism blocks connect to central cubes via
    // rotate_wedge_* and to each other at axial (zeta) seams with aligned
    // orientation.
    const std::vector<
        std::tuple<std::string, Dir, std::string, OrientationMap<3>>>
        cube_face_defs{
            {"BLeft", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"BLeft", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"BLeft", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"BLeft", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
            {"ARight", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"ARight", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"ARight", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"ARight", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
            {"ALeft", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"ALeft", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"ALeft", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"ALeft", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
            {"BRight", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"BRight", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"BRight", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"BRight", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
        };
    for (const auto& [grp, face_dir, where, rot] : cube_face_defs) {
      const std::string cube = std::string(grp).append("CentralCube"s);
      const std::string wedge =
          std::string(grp).append("DeformedCube"s).append(where);
      edges.push_back({cube, face_dir, wedge, rot});
      edges.push_back({wedge, Dir::lower_xi(), cube, rot.inverse_map()});
    }

    // Axial seams between groups (zeta direction, aligned).
    for (const std::string where : {"Front", "Top", "Back", "Bottom"}) {
      // x = center_B_
      edges.push_back({"BLeftDeformedCube" + where, Dir::lower_zeta(),
                       "BRightDeformedCube" + where, aligned});
      edges.push_back({"BRightDeformedCube" + where, Dir::upper_zeta(),
                       "BLeftDeformedCube" + where, aligned});
      // x = 0
      edges.push_back({"BRightDeformedCube" + where, Dir::lower_zeta(),
                       "ALeftDeformedCube" + where, aligned});
      edges.push_back({"ALeftDeformedCube" + where, Dir::upper_zeta(),
                       "BRightDeformedCube" + where, aligned});
      // x = center_A_
      edges.push_back({"ALeftDeformedCube" + where, Dir::lower_zeta(),
                       "ARightDeformedCube" + where, aligned});
      edges.push_back({"ARightDeformedCube" + where, Dir::upper_zeta(),
                       "ALeftDeformedCube" + where, aligned});
    }
  } else {
    // Bulge=true
    const std::vector<
        std::tuple<std::string, Dir, std::string, OrientationMap<3>>>
        fow_cube_defs{
            // {group, cube_face_dir, where, map}
            {"BLeft", Dir::lower_eta(), "Front", rotate_to_minus_y_axis},
            {"BLeft", Dir::upper_zeta(), "Top", aligned},
            {"BLeft", Dir::upper_eta(), "Back", rotate_to_y_axis},
            {"BLeft", Dir::lower_zeta(), "Bottom", rotate_to_minus_z_axis},
            {"ALeft", Dir::lower_eta(), "Front", rotate_to_minus_y_axis},
            {"ALeft", Dir::upper_zeta(), "Top", aligned},
            {"ALeft", Dir::upper_eta(), "Back", rotate_to_y_axis},
            {"ALeft", Dir::lower_zeta(), "Bottom", rotate_to_minus_z_axis},
            // BRight
            {"BRight", Dir::lower_eta(), "Front",
             half_turn_about_y_minus_z_axis},
            {"BRight", Dir::upper_zeta(), "Top", half_turn_about_z},
            {"BRight", Dir::upper_eta(), "Back", half_turn_about_y_plus_z_axis},
            {"BRight", Dir::lower_zeta(), "Bottom", half_turn_about_y_axis},
            // ARight
            {"ARight", Dir::lower_eta(), "Front",
             half_turn_about_y_minus_z_axis},
            {"ARight", Dir::upper_zeta(), "Top", half_turn_about_z},
            {"ARight", Dir::upper_eta(), "Back", half_turn_about_y_plus_z_axis},
            {"ARight", Dir::lower_zeta(), "Bottom", half_turn_about_y_axis}};
    for (const auto& [grp, face_dir, where, map] : fow_cube_defs) {
      const std::string cube = std::string(grp).append("CentralCube"s);
      const std::string fow =
          std::string(grp).append("DeformedCube"s).append(where);
      edges.push_back({cube, face_dir, fow, map});
      edges.push_back({fow, Dir::lower_zeta(), cube, map.inverse_map()});
    }

    // Axial seams between FOW groups (xi direction)
    for (const std::string where : {"Front", "Top", "Back", "Bottom"}) {
      edges.push_back({"BLeftDeformedCube" + where, Dir::upper_xi(),
                       "BRightDeformedCube" + where, half_turn_about_z});
      edges.push_back({"BRightDeformedCube" + where, Dir::upper_xi(),
                       "BLeftDeformedCube" + where, half_turn_about_z});
      edges.push_back({"BRightDeformedCube" + where, Dir::lower_xi(),
                       "ALeftDeformedCube" + where, half_turn_about_z});
      edges.push_back({"ALeftDeformedCube" + where, Dir::lower_xi(),
                       "BRightDeformedCube" + where, half_turn_about_z});
      edges.push_back({"ALeftDeformedCube" + where, Dir::upper_xi(),
                       "ARightDeformedCube" + where, half_turn_about_z});
      edges.push_back({"ARightDeformedCube" + where, Dir::upper_xi(),
                       "ALeftDeformedCube" + where, half_turn_about_z});
    }
  }

  // Angular connections within each group (aligned orientation in all cases)
  const std::array<std::string, 4> sides{"Front", "Top", "Back", "Bottom"};
  for (const std::string grp : {"BLeft", "BRight", "ALeft", "ARight"}) {
    const bool reversed = bulge_ and (grp == "BRight" or grp == "ARight");
    for (size_t i = 0; i < 4; ++i) {
      const std::string a = grp + "DeformedCube" + gsl::at(sides, i);
      const std::string b = grp + "DeformedCube" + gsl::at(sides, (i + 1) % 4);
      if (reversed) {
        edges.push_back({a, Dir::lower_eta(), b, aligned});
        edges.push_back({b, Dir::upper_eta(), a, aligned});
      } else {
        edges.push_back({a, Dir::upper_eta(), b, aligned});
        edges.push_back({b, Dir::lower_eta(), a, aligned});
      }
    }
  }

  // Cylinder connections
  const OrientationMap<3> orient_b_filled_to_hollow{
      std::array<Dir, 3>{Dir::upper_zeta(), Dir::upper_eta(), Dir::lower_xi()}};

  const OrientationMap<3> orient_a_filled_to_hollow{
      std::array<Dir, 3>{Dir::lower_zeta(), Dir::upper_eta(), Dir::upper_xi()}};

  edges.push_back({"BFilledCylinder", Dir::upper_xi(), "BLeftHollowCylinder",
                   orient_b_filled_to_hollow});
  edges.push_back({"BLeftHollowCylinder", Dir::lower_zeta(), "BFilledCylinder",
                   orient_b_filled_to_hollow.inverse_map()});
  edges.push_back({"AFilledCylinder", Dir::upper_xi(), "ARightHollowCylinder",
                   orient_a_filled_to_hollow});
  edges.push_back({"ARightHollowCylinder", Dir::upper_zeta(), "AFilledCylinder",
                   orient_a_filled_to_hollow.inverse_map()});

  edges.push_back({"BLeftHollowCylinder", Dir::upper_zeta(),
                   "BRightHollowCylinder", aligned});
  edges.push_back({"BRightHollowCylinder", Dir::lower_zeta(),
                   "BLeftHollowCylinder", aligned});
  edges.push_back({"BRightHollowCylinder", Dir::upper_zeta(),
                   "ALeftHollowCylinder", aligned});
  edges.push_back({"ALeftHollowCylinder", Dir::lower_zeta(),
                   "BRightHollowCylinder", aligned});
  edges.push_back({"ALeftHollowCylinder", Dir::upper_zeta(),
                   "ARightHollowCylinder", aligned});
  edges.push_back({"ARightHollowCylinder", Dir::lower_zeta(),
                   "ALeftHollowCylinder", aligned});

  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors(
      coordinate_maps.size());
  for (const auto& edge : edges) {
    const auto self_it = tag_to_index.find(edge.self_tag);
    const auto neighbor_it = tag_to_index.find(edge.neighbor_tag);
    if (self_it == tag_to_index.end() or neighbor_it == tag_to_index.end()) {
      continue;
    }
    neighbors[self_it->second].emplace(
        edge.self_direction,
        BlockNeighbors<3>({neighbor_it->second},
                          {{neighbor_it->second, edge.orientation}},
                          edge.are_conforming));
  }

  // Non-conforming interface between SphericalShell and the outer cylinders.
  {
    const OrientationMap<3> sphere_to_b_filled{
        std::array<Dir, 3>{Dir::lower_zeta(), Dir::self(), Dir::self()}};
    const OrientationMap<3> sphere_to_a_filled{
        std::array<Dir, 3>{Dir::upper_zeta(), Dir::self(), Dir::self()}};
    const OrientationMap<3> sphere_to_side{
        std::array<Dir, 3>{Dir::upper_xi(), Dir::self(), Dir::self()}};
    const size_t c = tag_to_index.at("SphericalShell");
    const size_t bf = tag_to_index.at("BFilledCylinder");
    const size_t blh = tag_to_index.at("BLeftHollowCylinder");
    const size_t brh = tag_to_index.at("BRightHollowCylinder");
    const size_t alh = tag_to_index.at("ALeftHollowCylinder");
    const size_t arh = tag_to_index.at("ARightHollowCylinder");
    const size_t af = tag_to_index.at("AFilledCylinder");
    neighbors[c].emplace(Dir::lower_xi(),
                         BlockNeighbors<3>{{bf, blh, brh, alh, arh, af},
                                           {{bf, sphere_to_b_filled},
                                            {blh, sphere_to_side},
                                            {brh, sphere_to_side},
                                            {alh, sphere_to_side},
                                            {arh, sphere_to_side},
                                            {af, sphere_to_a_filled}},
                                           false});
    neighbors[bf].emplace(
        Dir::lower_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_b_filled.inverse_map()}}, false});
    neighbors[af].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_a_filled.inverse_map()}}, false});
    for (const size_t hc : {blh, brh, alh, arh}) {
      neighbors[hc].emplace(
          Dir::upper_xi(),
          BlockNeighbors<3>{{c}, {{c, sphere_to_side.inverse_map()}}, false});
    }
  }

  // Non-conforming interfaces between deformed cubes and hollow cylinders.
  {
    const OrientationMap<3> cube_to_hollow =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{Dir::self(), Dir::self(),
                                                      Dir::upper_xi()}}
               : OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_xi(), Dir::self(), Dir::self()}};
    const OrientationMap<3> hollow_to_cube =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{Dir::upper_zeta(),
                                                      Dir::self(), Dir::self()}}
               : OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_xi(), Dir::self(), Dir::self()}};
    const Dir cube_outer_face = bulge_ ? Dir::upper_zeta() : Dir::upper_xi();

    // BLeftHollowCylinder <-> BLeft deformed cubes
    const size_t blh = tag_to_index.at("BLeftHollowCylinder");
    const size_t blf = tag_to_index.at("BLeftDeformedCubeFront");
    const size_t blt = tag_to_index.at("BLeftDeformedCubeTop");
    const size_t blb = tag_to_index.at("BLeftDeformedCubeBack");
    const size_t blbo = tag_to_index.at("BLeftDeformedCubeBottom");
    neighbors[blh].emplace(Dir::lower_xi(),
                           BlockNeighbors<3>{{blf, blt, blb, blbo},
                                             {{blf, hollow_to_cube},
                                              {blt, hollow_to_cube},
                                              {blb, hollow_to_cube},
                                              {blbo, hollow_to_cube}},
                                             false});
    for (const size_t idx : {blf, blt, blb, blbo}) {
      neighbors[idx].emplace(
          cube_outer_face,
          BlockNeighbors<3>{{blh}, {{blh, cube_to_hollow}}, false});
    }

    // BRightHollowCylinder <-> BRight deformed cubes
    const size_t brh = tag_to_index.at("BRightHollowCylinder");
    const size_t brf = tag_to_index.at("BRightDeformedCubeFront");
    const size_t brt = tag_to_index.at("BRightDeformedCubeTop");
    const size_t brb = tag_to_index.at("BRightDeformedCubeBack");
    const size_t brbo = tag_to_index.at("BRightDeformedCubeBottom");
    neighbors[brh].emplace(Dir::lower_xi(),
                           BlockNeighbors<3>{{brf, brt, brb, brbo},
                                             {{brf, hollow_to_cube},
                                              {brt, hollow_to_cube},
                                              {brb, hollow_to_cube},
                                              {brbo, hollow_to_cube}},
                                             false});
    for (const size_t idx : {brf, brt, brb, brbo}) {
      neighbors[idx].emplace(
          cube_outer_face,
          BlockNeighbors<3>{{brh}, {{brh, cube_to_hollow}}, false});
    }

    // ALeftHollowCylinder <-> ALeft deformed cubes
    const size_t alh = tag_to_index.at("ALeftHollowCylinder");
    const size_t alf = tag_to_index.at("ALeftDeformedCubeFront");
    const size_t alt = tag_to_index.at("ALeftDeformedCubeTop");
    const size_t alb = tag_to_index.at("ALeftDeformedCubeBack");
    const size_t albo = tag_to_index.at("ALeftDeformedCubeBottom");
    neighbors[alh].emplace(Dir::lower_xi(),
                           BlockNeighbors<3>{{alf, alt, alb, albo},
                                             {{alf, hollow_to_cube},
                                              {alt, hollow_to_cube},
                                              {alb, hollow_to_cube},
                                              {albo, hollow_to_cube}},
                                             false});
    for (const size_t idx : {alf, alt, alb, albo}) {
      neighbors[idx].emplace(
          cube_outer_face,
          BlockNeighbors<3>{{alh}, {{alh, cube_to_hollow}}, false});
    }

    // ARightHollowCylinder <-> ARight deformed cubes
    const size_t arh = tag_to_index.at("ARightHollowCylinder");
    const size_t arf = tag_to_index.at("ARightDeformedCubeFront");
    const size_t art = tag_to_index.at("ARightDeformedCubeTop");
    const size_t arb = tag_to_index.at("ARightDeformedCubeBack");
    const size_t arbo = tag_to_index.at("ARightDeformedCubeBottom");
    neighbors[arh].emplace(Dir::lower_xi(),
                           BlockNeighbors<3>{{arf, art, arb, arbo},
                                             {{arf, hollow_to_cube},
                                              {art, hollow_to_cube},
                                              {arb, hollow_to_cube},
                                              {arbo, hollow_to_cube}},
                                             false});
    for (const size_t idx : {arf, art, arb, arbo}) {
      neighbors[idx].emplace(
          cube_outer_face,
          BlockNeighbors<3>{{arh}, {{arh, cube_to_hollow}}, false});
    }
  }

  {
    // BFilledCylinder <-> BLeft
    const OrientationMap<3> bfc_to_blcc{
        std::array<Dir, 3>{Dir::self(), Dir::self(), Dir::upper_xi()}};
    const OrientationMap<3> bfc_to_bldeformed =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_zeta(), Dir::self(), Dir::upper_xi()}}
               : OrientationMap<3>{std::array<Dir, 3>{Dir::self(), Dir::self(),
                                                      Dir::lower_zeta()}};
    const Dir bl_fc_face = bulge_ ? Dir::lower_xi() : Dir::upper_zeta();

    const size_t bf = tag_to_index.at("BFilledCylinder");
    const size_t blcc = tag_to_index.at("BLeftCentralCube");
    const size_t blf = tag_to_index.at("BLeftDeformedCubeFront");
    const size_t blt = tag_to_index.at("BLeftDeformedCubeTop");
    const size_t blb = tag_to_index.at("BLeftDeformedCubeBack");
    const size_t blbo = tag_to_index.at("BLeftDeformedCubeBottom");
    neighbors[bf].emplace(Dir::upper_zeta(),
                          BlockNeighbors<3>{{blcc, blf, blt, blb, blbo},
                                            {{blcc, bfc_to_blcc},
                                             {blf, bfc_to_bldeformed},
                                             {blt, bfc_to_bldeformed},
                                             {blb, bfc_to_bldeformed},
                                             {blbo, bfc_to_bldeformed}},
                                            false});
    neighbors[blcc].emplace(
        Dir::lower_xi(),
        BlockNeighbors<3>{{bf}, {{bf, bfc_to_blcc.inverse_map()}}, false});
    for (const size_t idx : {blf, blt, blb, blbo}) {
      neighbors[idx].emplace(
          bl_fc_face,
          BlockNeighbors<3>{
              {bf}, {{bf, bfc_to_bldeformed.inverse_map()}}, false});
    }

    // AFilledCylinder <-> ARight
    const OrientationMap<3> afc_to_arcc{
        std::array<Dir, 3>{Dir::self(), Dir::self(), Dir::upper_xi()}};
    const OrientationMap<3> afc_to_ardeformed =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_zeta(), Dir::self(), Dir::lower_xi()}}
               : OrientationMap<3>{std::array<Dir, 3>{Dir::self(), Dir::self(),
                                                      Dir::lower_zeta()}};
    const Dir ar_fc_face = bulge_ ? Dir::lower_xi() : Dir::lower_zeta();

    const size_t af = tag_to_index.at("AFilledCylinder");
    const size_t arcc = tag_to_index.at("ARightCentralCube");
    const size_t arf = tag_to_index.at("ARightDeformedCubeFront");
    const size_t art = tag_to_index.at("ARightDeformedCubeTop");
    const size_t arb = tag_to_index.at("ARightDeformedCubeBack");
    const size_t arbo = tag_to_index.at("ARightDeformedCubeBottom");
    neighbors[af].emplace(Dir::lower_zeta(),
                          BlockNeighbors<3>{{arcc, arf, art, arb, arbo},
                                            {{arcc, afc_to_arcc},
                                             {arf, afc_to_ardeformed},
                                             {art, afc_to_ardeformed},
                                             {arb, afc_to_ardeformed},
                                             {arbo, afc_to_ardeformed}},
                                            false});
    neighbors[arcc].emplace(
        Dir::upper_xi(),
        BlockNeighbors<3>{{af}, {{af, afc_to_arcc.inverse_map()}}, false});
    for (const size_t idx : {arf, art, arb, arbo}) {
      neighbors[idx].emplace(
          ar_fc_face,
          BlockNeighbors<3>{
              {af}, {{af, afc_to_ardeformed.inverse_map()}}, false});
    }
  }

  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  std::vector<Block<3>> blocks{};
  blocks.reserve(number_of_blocks_);
  for (size_t i = 0; i < coordinate_maps.size(); ++i) {
    const std::array<domain::Topology, 3> topology =
        i == 26                 ? domain::topologies::spherical_shell
        : (i == 20 or i == 25)  ? domain::topologies::full_cylinder
        : (i >= 21 and i <= 24) ? domain::topologies::cylindrical_shell
                                : domain::topologies::hypercube<3>;
    blocks.emplace_back(std::move(coordinate_maps[i]), i,
                        std::move(neighbors[i]), block_names_[i], topology);
  }

  Domain<3> domain{std::move(blocks), std::move(excision_spheres),
                   block_groups_};

  if (time_dependent_options_.has_value()) {
    // All blocks get only a Grid->Inertial map (no distorted frame since there
    // are no excision surfaces / shape maps in the Pill domain).
    // Blocks 0-25 (inner region): rigid RotScaleTrans.
    // Block 26 (SphericalShell): transitioning RotScaleTrans.
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        grid_to_inertial_block_maps{number_of_blocks_};
    for (size_t block_id = 0; block_id < 26; ++block_id) {
      grid_to_inertial_block_maps[block_id] =
          time_dependent_options_
              ->grid_to_inertial_map<domain::ObjectLabel::None>(false, true);
    }
    grid_to_inertial_block_maps[26] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false, false);
    for (size_t block_id = 0; block_id < number_of_blocks_; ++block_id) {
      if (grid_to_inertial_block_maps[block_id] == nullptr) {
        continue;
      }
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(grid_to_inertial_block_maps[block_id]), nullptr,
          nullptr);
    }
  }
  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
Pill::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks_};
  // Outer boundary: upper_xi face of SphericalShell (block 26).
  boundary_conditions[26][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> Pill::initial_extents() const {
  return initial_grid_points_;
}

std::vector<std::array<size_t, 3>> Pill::initial_refinement_levels() const {
  return initial_refinement_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Pill::functions_of_time(const std::unordered_map<std::string, double>&
                            initial_expiration_times) const {
  return time_dependent_options_.has_value()
             ? time_dependent_options_->create_functions_of_time(
                   initial_expiration_times)
             : std::unordered_map<
                   std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{};
}
}  // namespace domain::creators
