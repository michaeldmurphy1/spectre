// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
 * expansions and uniform translations. The mesh velocity is then
 * \f$v^i = \dot{M}^i{}_j x^j_{\mathrm{grid}} + \dot{T}^i\f$, and the inverse
 * Jacobian \f$(M^{-1})^j{}_i\f$ and the divergence of the mesh velocity
 * \f$\dot{M}^i{}_j (M^{-1})^j{}_i\f$ are constant. Since the projections from
 * the DG grid are linear and exact for linear functions, projecting the
 * logical to grid geometry once and applying this map gives the same face
 * data as projecting the logical to inertial geometry, as the host does.
 * The divergence of the mesh velocity is exact, whereas the host differentiates
 * the mesh velocity on the DG grid, so the two differ by the truncation error
 * of the spectral derivative on curved elements (they agree for affine
 * element maps).
 *
 * \note Non-affine maps aren't supported yet. In particular the binary
 * neutron star domains expand with a `CubicScale` map whose inner and outer
 * expansion factors differ, so their grid to inertial map isn't affine (it
 * also rotates and translates). The plan is a device version of these maps:
 * the host evaluates the functions of time at each substep, and a kernel
 * evaluates the map, its inverse Jacobian and the mesh velocity at the face
 * grid coordinates (already in `LogicalToGrid`) and at the cell grid
 * coordinates (to be added). Unlike the
 * affine case, this then differs from the host (which projects the inertial
 * geometry from the DG grid) by the truncation error of the projection.
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
 * the elements with the map, to a relative tolerance.
 */
AffineGridToInertial affine_grid_to_inertial(
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
    const tnsr::I<DataVector, 3, Frame::Grid>& sample_points);
}  // namespace evolution::patches
