// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
/// \endcond

namespace evolution::patches {
/*!
 * \brief A grid to inertial map at one time that is affine:
 * \f$x^i = M^i{}_j x^j_{\mathrm{grid}} + T^i\f$.
 *
 * \details This covers the identity and compositions of rotations, uniform
 * expansions and uniform translations, e.g. the rigid region of a
 * `RotScaleTrans` map. The mesh velocity is then
 * \f$v^i = \dot{M}^i{}_j x^j_{\mathrm{grid}} + \dot{T}^i\f$, and the inverse
 * Jacobian \f$(M^{-1})^j{}_i\f$ and the divergence of the mesh velocity
 * \f$\dot{M}^i{}_j (M^{-1})^j{}_i\f$ are constant, so kernels can apply the
 * map to time-independent logical to grid geometry without evaluating it at
 * every point. Maps that aren't affine are evaluated pointwise with
 * `domain::CoordinateMaps::DeviceMap` (see `device_grid_to_inertial`).
 */
struct AffineGridToInertial {
  /// \f$(M^{-1})^j{}_i = \partial x^j_{\mathrm{grid}} / \partial x^i\f$
  InverseJacobian<double, 3, Frame::Grid, Frame::Inertial> inverse_jacobian{};
  double det_inverse_jacobian = 1.0;
  /// \f$\dot{M}^i{}_j\f$
  Jacobian<double, 3, Frame::Grid, Frame::Inertial> velocity_matrix{};
  /// \f$\dot{T}^i\f$
  tnsr::I<double, 3, Frame::Inertial> velocity_offset{};
  double div_mesh_velocity = 0.0;
  /// Whether the mesh moves, i.e. the map isn't the identity
  bool is_moving = false;
};

/*!
 * \brief The `AffineGridToInertial` of the `grid_to_inertial` map at `time`.
 *
 * Errors if the map isn't affine at the `sample_points`, e.g. the corners of
 * the elements with the map, to a relative tolerance. Without sample points,
 * the map is assumed to be affine.
 */
AffineGridToInertial affine_grid_to_inertial(
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
    const tnsr::I<DataVector, 3, Frame::Grid>& sample_points);

/*!
 * \brief The `grid_to_inertial` map at `time` for device kernels.
 *
 * \details This is the map's `CoordinateMapBase::device_map`. Maps without a
 * device version must be affine at the `sample_points` (see
 * `affine_grid_to_inertial`), and are sampled into a
 * `domain::CoordinateMaps::DeviceAffineMap`.
 */
::domain::CoordinateMaps::DeviceMap<3> device_grid_to_inertial(
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
    const tnsr::I<DataVector, 3, Frame::Grid>& sample_points);
}  // namespace evolution::patches
