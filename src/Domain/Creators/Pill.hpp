// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CylindricalFlatEndcapInterior.hpp"
#include "Domain/CoordinateMaps/CylindricalSphericalShell.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/FlatOffsetWedge.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
struct BlockLogical;
}  // namespace Frame
/// \endcond

namespace domain::creators {

/*!
 * \ingroup ComputationalDomainGroup
 *
 * \brief A domain for two neutron stars.
 *
 */
class Pill : public DomainCreator<3> {
 public:
  using maps_list = tmpl::flatten<tmpl::list<
      // Central cubes (blocks 0, 5, 10, 15): Affine3D
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      // Bulge=false: Wedge prisms (blocks 1-4, 6-9, 11-14, 16-19):
      // Wedge<2> x Affine + rotation to final orientation
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                         CoordinateMaps::Affine>,
          CoordinateMaps::DiscreteRotation<3>>,
      // Bulge=true: ALeft/BLeft FOW blocks: FOW + ShiftX + Rotation
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial, CoordinateMaps::FlatOffsetWedge,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>>,
      // Bulge=true: BRight FOW blocks: FOW + Rotation
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::FlatOffsetWedge,
                            CoordinateMaps::DiscreteRotation<3>>,
      // Bulge=true: ARight FOW blocks:
      // FOW + Rotation + ShiftX + Rotation
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial, CoordinateMaps::FlatOffsetWedge,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>>,
      // Filled cylinders (blocks 20, 25): radial Affine +
      // polar-to-Cartesian + half-turn + interior endcap + rotation to
      // x-axis + rotation to align angle
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::CylindricalFlatEndcapInterior,
          CoordinateMaps::DiscreteRotation<3>>,
      // Hollow cylinders (blocks 21-24): bulge = false
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::CylindricalSphericalShell,
                            CoordinateMaps::DiscreteRotation<3>>,
      // Hollow cylinders (blocks 21-24): bulge = true
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Interval>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::UniformCylindricalSide,
          CoordinateMaps::DiscreteRotation<3>>,
      // Spherical shell (block 26): radial Interval + S2
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                         CoordinateMaps::Identity<2>>,
          CoordinateMaps::SphericalToCartesianPfaffian>,
      bco::TimeDependentMapOptions<true>::maps_list>>;

  /*!
   * \brief Grid coordinates for center of Object A
   */
  struct CenterA {
    using type = double;
    static constexpr Options::String help = {
        "Grid coordinates of center for Object A, which is at x>0."};
  };
  /*!
   * \brief Grid coordinates for center of Object B
   */
  struct CenterB {
    using type = double;
    static constexpr Options::String help = {
        "Grid coordinates of center for Object B, which is at x<0."};
  };
  /*!
   * \brief Grid coordinate of the left-most face of the inner cubes.
   *
   * Must satisfy LeftmostX < CenterB < 0.
   */
  struct LeftmostX {
    using type = double;
    static constexpr Options::String help = {
        "x-coordinate of the left face of the BLeft inner-cube group. "
        "Must satisfy LeftmostX < CenterB < 0."};
  };
  /*!
   * \brief x-coordinate of the right-most face of the inner cubes
   *
   * Must satisfy 0 < CenterA < RightmostX.
   */
  struct RightmostX {
    using type = double;
    static constexpr Options::String help = {
        "x-coordinate of the right face of the ARight inner-cube group. "
        "Must satisfy 0 < CenterA < RightmostX."};
  };
  /*!
   * \brief Grid-coordinate radius of deformed-cube wedges
   *
   * This in turn sets the y/z grid coordinates
   */
  struct WedgeInnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius of central-cube blocks."};
  };
  /*!
   * \brief Grid-coordinate radius for outer boundary of deformed-cube wedges
   */
  struct WedgeOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius for outer boundary of wedges about central "
        "cubes."};
  };
  /*!
   * \brief Grid-coordinate radius for outer boundary of cylinders (also the
   * inner interface of the spherical shells)
   */
  struct CylinderOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius for outer boundary of cylinders about the "
        "cubed spheres."};
  };
  /*!
   * \brief Grid-coordinate radius for outer boundary of spherical shells
   */
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius for outer boundary of spherical shells."};
  };

  /*!
   * \brief Initial number of grid points for inner cubes
   */
  struct CubeInitialGridPoints {
    using type = std::array<size_t, 4>;
    static constexpr Options::String help = {
        "Initial number of grid points for inner cubes. This both sets the "
        "cubes, but also the angular part of the deformed cubes surrounding "
        "the center 4 cubes."};
  };
  /*!
   * \brief Initial \f$x\f$ refinement for inner cube
   */
  struct CubeInitialXRefinement {
    using type = std::array<size_t, 4>;
    static constexpr Options::String help = {
        "Initial x refinement for inner cubes. This both sets the cubes, but "
        "also the x part of the deformed cubes surrounding the center 4 "
        "cubes."};
  };
  /*!
   * \brief Initial \f$y\f$ and \f$z\f$ refinement for inner cube
   */
  struct CubeInitialYZRefinement {
    using type = std::array<size_t, 4>;
    static constexpr Options::String help = {
        "Initial y/z refinement for inner cubes. This both sets the cubes, but "
        "also the y/z of the deformed cubes surrounding the center 4 "
        "cubes."};
  };
  /*!
   * \brief Initial number of radial grid points for wedge prisms
   */
  struct WedgePrismInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for wedge prisms surrounding "
        "the inner cubes."};
  };
  /*!
   * \brief Initial radial refinement for wedge prisms
   */
  struct WedgePrismInitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement for deformed wedge prisms surrounding the "
        "inner cubes."};
  };
  /*!
   * \brief Initial number of radial grid points for cylinders
   */
  struct CylinderInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for cylinders."};
  };
  /*!
   * \brief Initial radial refinement for cylinders
   */
  struct CylinderInitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement for cylinders."};
  };
  /*!
   * \brief Initial number of angular grid points for B2s
   */
  struct B2InitialAngularGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of angular grid points for B2."};
  };
  /*!
   * \brief Initial number of angular grid points for hollow cylinder
   */
  struct HollowCylinderInitialAngularGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of angular grid points for hollow cylinders, passed as "
        "logical [eta, zeta]."};
  };
  /*!
   * \brief Initial radial refinement of spherical shells
   */
  struct SphericalShellsInitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement for spherical shells."};
  };
  /*!
   * \brief Initial number of radial grid points for spherical shells
   */
  struct SphericalShellsInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for spherical shells."};
  };
  /*!
   * \brief Initial spherical harmonic resolution for spherical shells
   */
  struct InitialSphericalHarmonicL {
    using type = size_t;
    static size_t lower_bound() { return 6; }
    static constexpr Options::String help = {
        "Initial spherical harmonic resolution specified as the highest "
        "spherical harmonic represented on the grid.  Minimum value is 6."};
  };
  /*!
   * \brief Radial distribution for spherical shells
   */
  struct SphericalShellsRadialDistribution {
    using type = domain::CoordinateMaps::Distribution;
    static constexpr Options::String help = {
        "Radial distribution for grid points in each spherical shell."};
  };

  /*!
   * \brief Whether to use a bulging coordinate map for the cubed-wedge and
   * hollow-cylinder regions.
   *
   * When `true`, uses `FlatOffsetWedge` for the four cubed-wedge groups and
   * `UniformCylindricalSide` for the four hollow cylinders, giving a smoother
   * outer boundary.  When `false`, uses `WedgePrism` +
   * `CylindricalSphericalShell`.
   */
  struct Bulge {
    using type = bool;
    static constexpr Options::String help = {
        "If true, use FlatOffsetWedge maps for the cubed-wedge regions."};
  };

  /*!
   * \brief Outer boundary condition
   */
  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static std::string name() { return "OuterBoundary"; }
    static constexpr Options::String help =
        "Options for the outer boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Time dependent maps
   */
  struct TimeDependentMaps {
    using type = Options::Auto<bco::TimeDependentMapOptions<true>,
                               Options::AutoLabel::None>;
    static constexpr Options::String help =
        bco::TimeDependentMapOptions<true>::help;
  };

  template <typename Metavariables>
  using options = tmpl::append<
      tmpl::list<
          CenterA, CenterB, LeftmostX, RightmostX, WedgeInnerRadius,
          WedgeOuterRadius, CylinderOuterRadius, OuterRadius,
          CubeInitialGridPoints, CubeInitialXRefinement,
          CubeInitialYZRefinement, WedgePrismInitialRadialGridPoints,
          WedgePrismInitialRadialRefinement, CylinderInitialRadialGridPoints,
          CylinderInitialRadialRefinement, B2InitialAngularGridPoints,
          HollowCylinderInitialAngularGridPoints,
          SphericalShellsInitialRadialRefinement,
          SphericalShellsInitialRadialGridPoints, InitialSphericalHarmonicL,
          SphericalShellsRadialDistribution, Bulge, TimeDependentMaps>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
          tmpl::list<>>>;

  static constexpr Options::String help{
      "The Pill domain is for binary neutron stars. There are 4 central cubes, "
      "that transition to outer spherical shells first by deformed-cube "
      "wedges, and then hollow and filled cylinders. There is an option to "
      "\"bulge\" the wedges which allows the grid-point distribution to change "
      "at block corners, notable the hollow cylinders"};

  Pill(double center_A, double center_B, double leftmost_x, double rightmost_x,
       double wedge_inner_radius, double wedge_outer_radius,
       double cylinder_outer_radius, double outer_radius,
       std::array<size_t, 4> cube_grid_points,
       std::array<size_t, 4> cube_x_refinement,
       std::array<size_t, 4> cube_yz_refinement,
       size_t wedge_prism_radial_grid_points,
       size_t wedge_prism_radial_refinement, size_t cylinder_radial_grid_points,
       size_t cylinder_radial_refinement, size_t b2_angular_grid_points,
       std::array<size_t, 2> hollow_cylinder_angular_grid_points,
       size_t spherical_shells_radial_refinement,
       size_t spherical_shells_radial_grid_points, size_t spherical_harmonic_l,
       domain::CoordinateMaps::Distribution SphericalShellsRadialDistribution,
       bool bulge = false,
       std::optional<bco::TimeDependentMapOptions<true>>
           time_dependent_options = std::nullopt,
       std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
           outer_boundary_condition = nullptr,
       const Options::Context& context = {});

  Pill() = default;
  Pill(const Pill&) = delete;
  Pill(Pill&&) = default;
  Pill& operator=(const Pill&) = delete;
  Pill& operator=(Pill&&) = default;
  ~Pill() override = default;

  Domain<3> create_domain() const override;

  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
  grid_anchors() const override {
    return grid_anchors_;
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  /// \brief The block names, which use left/right directionality as from a
  /// viewer looking down the \f$y\f$-axis.
  ///
  /// \details The block names start with the central cubes:
  ///  - `BLeftCubedCylinderCenter`, `BLeftCubedCylinderFront`,
  /// `BLeftCubedCylinderTop`, `BLeftCubedCylinderBack`,
  /// `BLeftCubedCylinderBottom`
  ///  - `BRightCubedCylinderCenter`, `BRightCubedCylinder`, ...
  ///  - `ALeftCubedCylinderCenter`, ...
  ///  - `ARightCubedCylinderCenter`, ...
  ///
  /// Then there are the cylinders:
  ///  - `BFilledCylinder`
  ///  - `BLeftHollowCylinder`
  ///  - `BRightHollowCylinder`
  ///  - `ALeftHollowCylinder`
  ///  - `ARightHollowCylinder`
  ///  - `AFilledCylinder`
  ///
  /// Finally, there is `SphericalShell`.
  std::vector<std::string> block_names() const override { return block_names_; }

  /// \brief The block groups, which are `CubedCylinders`, `Cylinders`, and
  /// `SphericalShells`.
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  double center_A_{};
  double center_B_{};
  double leftmost_x_{};
  double rightmost_x_{};
  double wedge_inner_radius_{};
  double wedge_outer_radius_{};
  double cylinder_outer_radius_{};
  double outer_radius_{};
  std::array<size_t, 4> cube_grid_points_{};
  std::array<size_t, 4> cube_x_refinement_{};
  std::array<size_t, 4> cube_yz_refinement_{};
  size_t wedge_prism_radial_grid_points_{};
  size_t wedge_prism_radial_refinement_{};
  size_t cylinder_radial_grid_points_{};
  size_t cylinder_radial_refinement_{};
  size_t b2_angular_grid_points_{};
  std::array<size_t, 2> hollow_cylinder_angular_grid_points_{};
  size_t spherical_shells_radial_refinement_{};
  size_t spherical_shells_radial_grid_points_{};
  size_t spherical_harmonic_l_{};
  domain::CoordinateMaps::Distribution SphericalShellsRadialDistribution_{};
  bool bulge_{};
  size_t number_of_blocks_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options_{};
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_grid_points_{};
};
}  // namespace domain::creators
