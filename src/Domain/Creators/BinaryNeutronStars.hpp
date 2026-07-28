// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class FlatOffsetWedge;
class Interval;
class PolarToCartesian;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
class SphericalToCartesianPfaffian;
template <size_t VolumeDim>
class Wedge;
template <size_t VolumeDim>
class DiscreteRotation;
class UniformCylindricalEndcap;
class UniformCylindricalSide;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;

namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain

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
class BinaryNeutronStars : public DomainCreator<3> {
 public:
  using maps_list = tmpl::flatten<tmpl::list<
      // Central cubes (blocks 0-3): simple Affine3D
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      // Cap and half-wedge blocks (4-8, 17-21): Wedge<3> + x-shift
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial, CoordinateMaps::Wedge<3>,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>>,
      // FlatOffsetWedge blocks (9-16): FOW + two rotations + x-shift
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial, CoordinateMaps::FlatOffsetWedge,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::DiscreteRotation<3>>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::FlatOffsetWedge,
                            CoordinateMaps::DiscreteRotation<3>>,
      // Filled cylinders (22, 25): radial shift + polar-to-Cartesian +
      // rotation + endcap + rotation
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::UniformCylindricalEndcap,
          CoordinateMaps::DiscreteRotation<3>>,
      // Hollow cylinders (23, 24): annular shift + polar-to-Cartesian +
      // rotation + side + rotation
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
      // Spherical shell (26): radial interval + S2
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                         CoordinateMaps::Identity<2>>,
          CoordinateMaps::SphericalToCartesianPfaffian>>>;

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
   * \brief Grid-coordinate radius of central-cube blocks.
   */
  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius of central-cube blocks."};
  };
  /*!
   * \brief Grid-coordinate radius for outer boundary of wedges about central
   * cubes
   */
  struct WedgeOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius for outer boundary of wedges about central "
        "cubes."};
  };
  /*!
   * \brief Grid-coordinate radius for outer boundary of cylinders about central
   * cubed sphered
   */
  struct CylinderOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius for outer boundary of cylinders about central"
        "cubed sphered."};
  };
  /*!
   * \brief Grid-coordinate radius for outer boundary of spherical shells
   */
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius for outer boundary of spherical shells."};
  };

  // struct UseEquiangularMap {
  //   using type = bool;
  //   static constexpr Options::String help = {
  //       "Distribute grid points equiangularly in 2d wedges."};
  //   static bool suggested_value() { return false; }
  // };

  /*!
   * \brief Initial number of grid points for inner cubes
   */
  struct CubeInitialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points for inner cubes. This both sets the "
        "cubes, but also the angular part of the deformed cubes surrounding "
        "the center 4 cubes."};
  };
  /*!
   * \brief Initial refinement for inner cube
   */
  struct CubeInitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement for inner cubes. This both sets the cubes, but "
        "also the angular part of the deformed cubes surrounding the center 4 "
        "cubes."};
  };
  /*!
   * \brief Initial number of radial grid points for deformed cubes
   */
  struct DeformedCubeInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for deformed cubes surrounding "
        "the inner cubes."};
  };
  /*!
   * \brief Initial radial refinement for deformed cubes
   */
  struct DeformedCubeInitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement for deformed cubes surrounding the inner "
        "cubes."};
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
   * \brief Initial number of radial radial grid points for spherical shells
   */
  struct SphericalShellsInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for spherical shells."};
  };
  /*!
   * \brief Initial number of radial radial grid points for spherical shells
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
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependant maps."};
  };

  template <typename Metavariables>
  using options = tmpl::append<
      tmpl::list<
          CenterA, CenterB, InnerRadius, WedgeOuterRadius, CylinderOuterRadius,
          OuterRadius, CubeInitialGridPoints, CubeInitialRefinement,
          DeformedCubeInitialRadialGridPoints,
          DeformedCubeInitialRadialRefinement, CylinderInitialRadialGridPoints,
          CylinderInitialRadialRefinement, B2InitialAngularGridPoints,
          HollowCylinderInitialAngularGridPoints,
          SphericalShellsInitialRadialRefinement,
          SphericalShellsInitialRadialGridPoints, InitialSphericalHarmonicL,
          SphericalShellsRadialDistribution, TimeDependence>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
          tmpl::list<>>>;

  static constexpr Options::String help{
      "The BinaryNeutronStars domain is for binary neutron stars."};

  BinaryNeutronStars(
      double center_A, double center_B, double inner_radius,
      double wedge_outer_radius, double cylinder_outer_radius,
      double outer_radius, size_t cube_grid_points, size_t cube_refinement,
      size_t deformed_cube_radial_grid_points,
      size_t deformed_cube_radial_refinement,
      size_t cylinder_radial_grid_points, size_t cylinder_radial_refinement,
      size_t b2_angular_grid_points,
      std::array<size_t, 2> hollow_cylinder_angular_grid_points,
      size_t spherical_shells_radial_refinement,
      size_t spherical_shells_radial_grid_points, size_t spherical_harmonic_l,
      domain::CoordinateMaps::Distribution SphericalShellsRadialDistribution,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  BinaryNeutronStars() = default;
  BinaryNeutronStars(const BinaryNeutronStars&) = delete;
  BinaryNeutronStars(BinaryNeutronStars&&) = default;
  BinaryNeutronStars& operator=(const BinaryNeutronStars&) = delete;
  BinaryNeutronStars& operator=(BinaryNeutronStars&&) = default;
  ~BinaryNeutronStars() override = default;

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

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  double center_A_{};
  double center_B_{};
  double inner_radius_{};
  double wedge_outer_radius_{};
  double cylinder_outer_radius_{};
  double outer_radius_{};
  size_t cube_grid_points_{};
  size_t cube_refinement_{};
  size_t deformed_cube_radial_grid_points_{};
  size_t deformed_cube_radial_refinement_{};
  size_t cylinder_radial_grid_points_{};
  size_t cylinder_radial_refinement_{};
  size_t b2_angular_grid_points_{};
  std::array<size_t, 2> hollow_cylinder_angular_grid_points_{};
  size_t spherical_shells_radial_refinement_{};
  size_t spherical_shells_radial_grid_points_{};
  size_t spherical_harmonic_l_{};
  domain::CoordinateMaps::Distribution SphericalShellsRadialDistribution_{};
  size_t number_of_blocks_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_{};
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_grid_points_{};
};
}  // namespace domain::creators
