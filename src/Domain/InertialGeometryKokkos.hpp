// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariablesKokkos.hpp"

/// \cond
namespace Frame {
struct ElementLogical;
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain {
namespace Tags {
/// The grid to inertial map of an element at one time, for device kernels
/// (e.g. as a `::Tags::PerElement` argument)
template <size_t Dim>
struct DeviceGridToInertial : db::SimpleTag {
  using type = CoordinateMaps::DeviceMap<Dim>;
};

/// The mesh velocity as a tensor. Unlike `domain::Tags::MeshVelocity` it isn't
/// a `std::optional`, so it can be part of a `BatchedVariables`.
template <size_t Dim>
struct MeshVelocityTensor : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};
}  // namespace Tags

/// The time-independent geometry at points of elements that
/// `inertial_geometry_on_device` maps to the inertial frame
template <size_t Dim>
using logical_to_grid_geometry_tags =
    tmpl::list<Tags::Coordinates<Dim, Frame::Grid>,
               Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Grid>>;

/// The geometry at points of elements at one time that
/// `inertial_geometry_on_device` computes, with or without the mesh velocity
template <size_t Dim, bool WithMeshVelocity>
using inertial_geometry_tags = tmpl::append<
    tmpl::list<
        Tags::Coordinates<Dim, Frame::Inertial>,
        Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Inertial>,
        Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>,
    tmpl::conditional_t<WithMeshVelocity,
                        tmpl::list<Tags::MeshVelocityTensor<Dim>>,
                        tmpl::list<>>>;

/*!
 * \brief Map the geometry of a batch of elements at a set of points (e.g. the
 * DG grid points or the FD cell centers) from the grid to the inertial frame,
 * on the device. Enqueued on `exec`.
 *
 * \details `logical_to_grid` holds the grid coordinates and the inverse
 * Jacobian of the element map at the points, which don't change in time.
 * `grid_to_inertial` holds the grid to inertial map of each element of the
 * batch at the current time, e.g. from `CoordinateMapBase::device_map`, so the
 * elements can be in different blocks.
 *
 * The results agree with the host to roundoff:
 * - the inertial coordinates, the logical to inertial inverse Jacobian and
 *   the mesh velocity with `domain::Tags::InertialFromGridCoordinatesCompute`,
 *   `domain::Tags::ElementToInertialInverseJacobian` and
 *   `domain::Tags::InertialMeshVelocityCompute` on the DG grid, and with
 *   `evolution::dg::subcell::Tags::InertialCoordinatesCompute` and
 *   `evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertialCompute`
 *   at the FD cell centers;
 * - the determinant of the inverse Jacobian with `determinant` of the inverse
 *   Jacobian, as `domain::Tags::DetInvJacobianCompute`.
 *
 * Where the map is the identity, the results equal the grid frame values.
 */
template <bool WithMeshVelocity>
void inertial_geometry_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<
        BatchedVariables<inertial_geometry_tags<3, WithMeshVelocity>>*>
        inertial,
    const BatchedVariables<logical_to_grid_geometry_tags<3>>& logical_to_grid,
    const Kokkos::View<const CoordinateMaps::DeviceMap<3>*>& grid_to_inertial);
}  // namespace domain
#endif  // SPECTRE_KOKKOS
