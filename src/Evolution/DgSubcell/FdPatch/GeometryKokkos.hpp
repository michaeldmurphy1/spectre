// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyMatricesKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/InertialGeometryKokkos.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
template <size_t Dim, typename TargetFrame>
class ElementMap;
/// \endcond

namespace evolution::dg::subcell::fd_patch {
/// Tags for the geometry of the elements of an FD patch
namespace Tags {
/// \brief Row \f$\hat{\imath}\f$ of the inverse Jacobian from the element
/// logical to the grid frame on the faces in direction \f$\hat{\imath}\f$,
/// i.e. \f$\partial \xi^{\hat{\imath}} / \partial x^j_{\mathrm{grid}}\f$,
/// projected from the DG grid.
struct LogicalToGridFaceConormal : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Grid>;
};

/// The determinant of the inverse Jacobian from the element logical to the
/// grid frame on the faces, projected from the DG grid.
struct LogicalToGridFaceDetInverseJacobian : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The grid coordinates on the faces, projected from the DG grid.
struct FaceGridCoordinates : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};

/// The determinant of the inverse Jacobian from the element logical to the
/// grid frame at the cell centers.
struct LogicalToGridCellDetInverseJacobian : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \brief Row \f$\hat{\imath}\f$ of the inverse Jacobian from the element
/// logical to the inertial frame on the faces in direction \f$\hat{\imath}\f$,
/// i.e. the unnormalized normal covector
/// \f$\partial \xi^{\hat{\imath}} / \partial x^j\f$.
struct FaceConormal : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};

/// The determinant of the inverse Jacobian from the element logical to the
/// inertial frame on the faces.
struct FaceDetInverseJacobian : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The mesh velocity on the faces.
struct FaceMeshVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

/// The determinant of the inverse Jacobian from the element logical to the
/// inertial frame at the cell centers.
struct CellDetInverseJacobian : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The mesh velocity at the cell centers, projected from the DG grid
struct CellMeshVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

/// The divergence of the mesh velocity at the cell centers, computed on the DG
/// grid and projected
struct CellDivMeshVelocity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags

/// The time-independent geometry on the faces of a patch
using logical_to_grid_face_tags =
    tmpl::list<Tags::LogicalToGridFaceConormal,
               Tags::LogicalToGridFaceDetInverseJacobian,
               Tags::FaceGridCoordinates>;
/// The time-independent geometry at the cell centers of a patch
using logical_to_grid_cell_tags =
    tmpl::list<Tags::LogicalToGridCellDetInverseJacobian>;
/// The time-independent geometry at the cell centers of the elements of a
/// patch on the host, as the subcell compute tags of an element compute it
/// (see `make_element_box`)
using host_logical_to_grid_cell_tags = tmpl::list<
    evolution::dg::subcell::Tags::Coordinates<3, Frame::Grid>,
    evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<3>,
    evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToGrid>;
/// The geometry on the faces of a patch at a time
using inertial_face_tags =
    tmpl::list<Tags::FaceConormal, Tags::FaceDetInverseJacobian,
               Tags::FaceMeshVelocity>;
/// The geometry at the cell centers of a patch at a time
using inertial_cell_tags = tmpl::list<Tags::CellDetInverseJacobian>;
/// The inertial coordinates, the logical to inertial inverse Jacobian and its
/// determinant at the cell centers of a patch at a time, evaluated there as
/// `evolution::dg::subcell::Tags::InertialCoordinatesCompute` and
/// `evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertialCompute`
using cell_geometry_tags = ::domain::inertial_geometry_tags<3, false>;
/// The mesh velocity and its divergence at the cell centers of a patch at a
/// time, projected from the DG grid as
/// `grmhd::GhValenciaDivClean::subcell::TimeDerivative` does
using cell_mesh_velocity_tags =
    tmpl::list<Tags::CellMeshVelocity, Tags::CellDivMeshVelocity>;

/*!
 * \brief The time-independent part of the geometry of the elements of a patch,
 * i.e. the element logical to grid map, in device memory.
 *
 * \details The face data is a batch of `3 * number_of_elements` face sets,
 * ordered by direction and then element, like the face variables of
 * `grmhd::ValenciaDivClean::fd::face_tags`. As on the host (see
 * `grmhd::ValenciaDivClean::subcell::TimeDerivative`), the face data is
 * projected from the DG grid with
 * `evolution::dg::subcell::fd::project_to_faces`, and the cell data is
 * evaluated at the cell centers.
 */
struct LogicalToGrid {
  BatchedVariables<logical_to_grid_face_tags> faces{};
  BatchedVariables<logical_to_grid_cell_tags> cells{};
  /// The grid coordinates and the logical to grid inverse Jacobian at the cell
  /// centers and at the DG grid points, where the grid to inertial maps are
  /// evaluated (see `TimeDependentGeometry`). Empty if not requested.
  BatchedVariables<::domain::logical_to_grid_geometry_tags<3>> cell_points{};
  BatchedVariables<::domain::logical_to_grid_geometry_tags<3>> dg_points{};
  /// The same geometry at the cell centers on the host, as the subcell compute
  /// tags of an element compute it, so the DataBoxes of the elements for
  /// events don't have to compute it again (see `make_element_box`)
  BatchedVariables<host_logical_to_grid_cell_tags, Kokkos::HostSpace>
      host_cells{};
  /// The grid coordinates of the corners of all elements on the host, e.g. to
  /// check where a grid to inertial map is affine.
  tnsr::I<DataVector, 3, Frame::Grid> corners{};
};

/*!
 * \brief Compute the `LogicalToGrid` geometry of the elements with the
 * `element_maps` (in slot order) and the shared `dg_mesh`, and copy it to the
 * device on `exec`.
 *
 * The `LogicalToGrid::cell_points` and `LogicalToGrid::dg_points` are only
 * computed `with_evaluation_points`. The `LogicalToGrid::host_cells` are
 * always computed.
 */
LogicalToGrid logical_to_grid(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::vector<ElementMap<3, Frame::Grid>>& element_maps,
    const Mesh<3>& dg_mesh, bool with_evaluation_points = true);

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

namespace Tags {
/// The `AffineGridToInertial` map of an element, as a `::Tags::PerElement`
/// argument of device kernels
struct AffineGridToInertial : db::SimpleTag {
  using type = fd_patch::AffineGridToInertial;
};
}  // namespace Tags

/*!
 * \brief The `AffineGridToInertial` of the `grid_to_inertial` map at `time`.
 *
 * Errors if the map isn't affine at the `sample_points`, e.g. the
 * `LogicalToGrid::corners`, to a relative tolerance.
 */
AffineGridToInertial affine_grid_to_inertial(
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
    const tnsr::I<DataVector, 3, Frame::Grid>& sample_points);

/*!
 * \brief Compute the geometry of the elements of a patch in the inertial frame
 * on the device, from the `logical_to_grid` geometry and the grid to inertial
 * maps of the elements at the current time. Enqueued on `exec`.
 *
 * `faces` and `cells` must have the same number of grid points and elements
 * as `logical_to_grid.faces` and `logical_to_grid.cells`. The mesh velocity is
 * zero where the mesh doesn't move.
 *
 * `grid_to_inertial` holds the map of each face set of `faces`, i.e. entry
 * `d * number_of_elements + e` holds the map of element `e` (for the faces in
 * direction `d`). The cells of element `e` use entry `e`. The maps are read on
 * the device, so they can change between substeps without changing the
 * arguments of the kernels.
 */
void inertial_geometry_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<inertial_face_tags>*> faces,
    gsl::not_null<BatchedVariables<inertial_cell_tags>*> cells,
    const LogicalToGrid& logical_to_grid,
    const Kokkos::View<const AffineGridToInertial*>& grid_to_inertial);

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

/*!
 * \brief Computes the geometry of the elements of a patch at a time on the
 * device for grid to inertial maps that aren't affine, as the host does.
 *
 * \details `compute_cells` evaluates the maps (`domain::CoordinateMaps::
 * DeviceMap`s, one per element) at the DG grid points and at the cell centers
 * with `domain::inertial_geometry_on_device` and then
 * - evaluates the `cell_geometry_tags` at the cell centers, as the host
 *   subcell compute tags;
 * - projects the mesh velocity from the DG grid to the cell centers, and
 *   computes its divergence on the DG grid with spectral derivatives (as
 *   `evolution::domain::Tags::DivMeshVelocityCompute`) and projects it, as
 *   `grmhd::GhValenciaDivClean::subcell::TimeDerivative`.
 *
 * `compute_faces` then projects the logical to inertial inverse Jacobian, its
 * determinant and the mesh velocity from the DG grid to the faces, as the
 * host does. For affine maps `inertial_geometry_on_device` with
 * `AffineGridToInertial` gives the same faces with less work.
 *
 * Everything is only enqueued, and the buffers are allocated once, so the
 * geometry can be updated in every substep. The projections are applied
 * dimension by dimension (see `DeviceApplyMatrices`).
 *
 * \note Only Legendre bases on Cartesian (isotropic) subcell grids are
 * supported, as for FD patches in general.
 */
class TimeDependentGeometry {
 public:
  TimeDependentGeometry() = default;
  TimeDependentGeometry(const Kokkos::DefaultExecutionSpace& exec,
                        const Mesh<3>& dg_mesh, size_t number_of_elements);

  /// Enqueue computing the geometry at the cell centers for the maps
  /// `grid_to_inertial` (one per element)
  void compute_cells(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<BatchedVariables<cell_geometry_tags>*> cell_geometry,
      gsl::not_null<BatchedVariables<cell_mesh_velocity_tags>*>
          cell_mesh_velocity,
      const LogicalToGrid& logical_to_grid,
      const Kokkos::View<const ::domain::CoordinateMaps::DeviceMap<3>*>&
          grid_to_inertial);

  /// Enqueue projecting the geometry on the DG grid of the last
  /// `compute_cells` to the faces (laid out as `LogicalToGrid::faces`)
  void compute_faces(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<BatchedVariables<inertial_face_tags>*> faces);

 private:
  size_t number_of_elements_ = 0;
  // The inertial geometry and the divergence of the mesh velocity on the DG
  // grid, and the logical derivatives of the mesh velocity
  BatchedVariables<::domain::inertial_geometry_tags<3, true>> dg_geometry_{};
  BatchedVariables<tmpl::list<::Tags::TempScalar<0>>> dg_div_mesh_velocity_{};
  BatchedVariables<tmpl::list<::Tags::TempI<0, 3, Frame::Inertial>,
                              ::Tags::TempI<1, 3, Frame::Inertial>,
                              ::Tags::TempI<2, 3, Frame::Inertial>>>
      dg_logical_derivatives_{};
  std::array<DeviceApplyMatrices, 3> logical_derivatives_{};
  DeviceApplyMatrices mesh_velocity_to_cells_{};
  DeviceApplyMatrices div_mesh_velocity_to_cells_{};
  std::array<DeviceApplyMatrices, 3> to_faces_{};
};
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
