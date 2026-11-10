// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Evolution/Patches/GridToInertial.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
template <size_t Dim, typename TargetFrame>
class ElementMap;
template <size_t Dim>
class Mesh;
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
}  // namespace Tags

/// The time-independent geometry on the faces of a patch
using logical_to_grid_face_tags =
    tmpl::list<Tags::LogicalToGridFaceConormal,
               Tags::LogicalToGridFaceDetInverseJacobian,
               Tags::FaceGridCoordinates>;
/// The time-independent geometry at the cell centers of a patch
using logical_to_grid_cell_tags =
    tmpl::list<Tags::LogicalToGridCellDetInverseJacobian>;
/// The geometry on the faces of a patch at a time
using inertial_face_tags =
    tmpl::list<Tags::FaceConormal, Tags::FaceDetInverseJacobian,
               Tags::FaceMeshVelocity>;
/// The geometry at the cell centers of a patch at a time
using inertial_cell_tags = tmpl::list<Tags::CellDetInverseJacobian>;

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
  /// The grid coordinates of the corners of all elements on the host, e.g. to
  /// check where a grid to inertial map is affine.
  tnsr::I<DataVector, 3, Frame::Grid> corners{};
};

/*!
 * \brief Compute the `LogicalToGrid` geometry of the elements with the
 * `element_maps` (in slot order) and the shared `dg_mesh`, and copy it to the
 * device on `exec`.
 */
LogicalToGrid logical_to_grid(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::vector<ElementMap<3, Frame::Grid>>& element_maps,
    const Mesh<3>& dg_mesh);

/*!
 * \brief Compute the geometry of the elements of a patch in the inertial frame
 * on the device, from the `logical_to_grid` geometry and the
 * `grid_to_inertial` map at the current time. Enqueued on `exec`.
 *
 * `faces` and `cells` must have the same number of grid points and elements
 * as `logical_to_grid.faces` and `logical_to_grid.cells`. The mesh velocity is
 * zero if the mesh doesn't move.
 */
void inertial_geometry_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<inertial_face_tags>*> faces,
    gsl::not_null<BatchedVariables<inertial_cell_tags>*> cells,
    const LogicalToGrid& logical_to_grid,
    const evolution::patches::AffineGridToInertial& grid_to_inertial);
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
