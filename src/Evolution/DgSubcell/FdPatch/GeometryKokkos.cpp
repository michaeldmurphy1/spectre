// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/ElementMap.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd_patch {
namespace {
// The parity arguments of `project_to_faces` used by the host for the
// component `j` of row `i` of the inverse Jacobian and for scalars
Spectral::Parity parity(const size_t i, const size_t j) {
  return (i == 0) != (j == 0) ? Spectral::Parity::Odd : Spectral::Parity::Even;
}

struct InertialFaceGeometry {
  using return_tags = inertial_face_tags;
  using argument_tags =
      tmpl::push_back<logical_to_grid_face_tags,
                      ::Tags::PerElement<Tags::AffineGridToInertial>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*> conormal,
      const gsl::not_null<Scalar<double>*> det_inverse_jacobian,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> mesh_velocity,
      const tnsr::i<double, 3, Frame::Grid>& grid_conormal,
      const Scalar<double>& grid_det_inverse_jacobian,
      const tnsr::I<double, 3, Frame::Grid>& grid_coords,
      const AffineGridToInertial& map) {
    for (size_t i = 0; i < 3; ++i) {
      conormal->get(i) = 0.0;
      mesh_velocity->get(i) = map.velocity_offset.get(i);
      for (size_t j = 0; j < 3; ++j) {
        conormal->get(i) +=
            grid_conormal.get(j) * map.inverse_jacobian.get(j, i);
        mesh_velocity->get(i) +=
            map.velocity_matrix.get(i, j) * grid_coords.get(j);
      }
    }
    get(*det_inverse_jacobian) =
        get(grid_det_inverse_jacobian) * map.det_inverse_jacobian;
  }
};

struct InertialCellGeometry {
  using return_tags = inertial_cell_tags;
  using argument_tags =
      tmpl::push_back<logical_to_grid_cell_tags,
                      ::Tags::PerElement<Tags::AffineGridToInertial>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> det_inverse_jacobian,
      const Scalar<double>& grid_det_inverse_jacobian,
      const AffineGridToInertial& map) {
    get(*det_inverse_jacobian) =
        get(grid_det_inverse_jacobian) * map.det_inverse_jacobian;
  }
};
}  // namespace

LogicalToGrid logical_to_grid(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::vector<ElementMap<3, Frame::Grid>>& element_maps,
    const Mesh<3>& dg_mesh) {
  const size_t number_of_elements = element_maps.size();
  const Mesh<3> subcell_mesh = fd::mesh(dg_mesh);
  const Index<3>& subcell_extents = subcell_mesh.extents();
  for (size_t d = 0; d < 3; ++d) {
    if (subcell_extents[d] != subcell_extents[0]) {
      ERROR("FD patches need isotropic subcell extents, but got "
            << subcell_extents);
    }
  }
  Index<3> face_extents = subcell_extents;
  face_extents[0] += 1;
  const size_t number_of_face_points = face_extents.product();

  BatchedVariables<logical_to_grid_face_tags, Kokkos::HostSpace> host_faces{
      Kokkos::DefaultHostExecutionSpace{}, number_of_face_points,
      3 * number_of_elements};
  BatchedVariables<logical_to_grid_cell_tags, Kokkos::HostSpace> host_cells{
      Kokkos::DefaultHostExecutionSpace{}, subcell_mesh.number_of_grid_points(),
      number_of_elements};
  LogicalToGrid result{};
  result.corners = tnsr::I<DataVector, 3, Frame::Grid>{8 * number_of_elements};

  const auto dg_logical_coords = logical_coordinates(dg_mesh);
  const auto cell_logical_coords = logical_coordinates(subcell_mesh);
  for (size_t e = 0; e < number_of_elements; ++e) {
    const auto& element_map = element_maps[e];
    // As `fd::Tags::DetInverseJacobianLogicalToGridCompute`
    Variables<logical_to_grid_cell_tags> cells{};
    set_data_ref_to_element(make_not_null(&cells), host_cells, e);
    get<Tags::LogicalToGridCellDetInverseJacobian>(cells) =
        determinant(element_map.inv_jacobian(cell_logical_coords));

    // As `grmhd::ValenciaDivClean::subcell::TimeDerivative`
    const auto dg_inv_jacobian = element_map.inv_jacobian(dg_logical_coords);
    const auto dg_det_inv_jacobian = determinant(dg_inv_jacobian);
    const auto dg_grid_coords = element_map(dg_logical_coords);
    for (size_t i = 0; i < 3; ++i) {
      Index<3> face_mesh_extents = subcell_extents;
      face_mesh_extents[i] += 1;
      Variables<logical_to_grid_face_tags> faces{};
      set_data_ref_to_element(make_not_null(&faces), host_faces,
                              i * number_of_elements + e);
      for (size_t j = 0; j < 3; ++j) {
        get<Tags::LogicalToGridFaceConormal>(faces).get(j) =
            fd::project_to_faces(dg_inv_jacobian.get(i, j), dg_mesh,
                                 face_mesh_extents, i, parity(i, j));
        // As the mesh velocity
        get<Tags::FaceGridCoordinates>(faces).get(j) = fd::project_to_faces(
            dg_grid_coords.get(j), dg_mesh, face_mesh_extents, i, parity(0, j));
      }
      get(get<Tags::LogicalToGridFaceDetInverseJacobian>(faces)) =
          fd::project_to_faces(get(dg_det_inv_jacobian), dg_mesh,
                               face_mesh_extents, i, Spectral::Parity::Even);
    }

    for (size_t corner = 0; corner < 8; ++corner) {
      tnsr::I<double, 3, Frame::ElementLogical> logical_corner{};
      for (size_t d = 0; d < 3; ++d) {
        logical_corner.get(d) = ((corner >> d) & 1) == 0 ? -1.0 : 1.0;
      }
      const auto grid_corner = element_map(logical_corner);
      for (size_t d = 0; d < 3; ++d) {
        result.corners.get(d)[8 * e + corner] = grid_corner.get(d);
      }
    }
  }

  result.faces = BatchedVariables<logical_to_grid_face_tags>{
      exec, number_of_face_points, 3 * number_of_elements};
  result.cells = BatchedVariables<logical_to_grid_cell_tags>{
      exec, subcell_mesh.number_of_grid_points(), number_of_elements};
  deep_copy(exec, make_not_null(&result.faces), host_faces);
  deep_copy(exec, make_not_null(&result.cells), host_cells);
  // The host data must stay valid until the copies are done
  exec.fence("fd_patch::logical_to_grid");
  return result;
}

AffineGridToInertial affine_grid_to_inertial(
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const tnsr::I<DataVector, 3, Frame::Grid>& sample_points) {
  AffineGridToInertial result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result.inverse_jacobian.get(i, j) = i == j ? 1.0 : 0.0;
      result.velocity_matrix.get(i, j) = 0.0;
    }
    result.velocity_offset.get(i) = 0.0;
  }
  if (grid_to_inertial.is_identity()) {
    return result;
  }
  result.is_moving = true;

  // The map and the mesh velocity at the origin and the unit vectors
  // determine an affine map
  const auto at = [&](const tnsr::I<double, 3, Frame::Grid>& point) {
    return grid_to_inertial.coords_frame_velocity_jacobians(point, time,
                                                            functions_of_time);
  };
  const auto [offset, inverse_jacobian, matrix, origin_velocity] =
      at(tnsr::I<double, 3, Frame::Grid>{0.0});
  result.inverse_jacobian = inverse_jacobian;
  result.velocity_offset = origin_velocity;
  for (size_t j = 0; j < 3; ++j) {
    tnsr::I<double, 3, Frame::Grid> unit{0.0};
    unit.get(j) = 1.0;
    const auto velocity = std::get<3>(at(unit));
    for (size_t i = 0; i < 3; ++i) {
      result.velocity_matrix.get(i, j) =
          velocity.get(i) - result.velocity_offset.get(i);
    }
  }
  result.det_inverse_jacobian = get(determinant(inverse_jacobian));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result.div_mesh_velocity +=
          result.velocity_matrix.get(i, j) * result.inverse_jacobian.get(j, i);
    }
  }

  // Check that the map is affine at the sample points
  constexpr double tolerance = 1.0e-10;
  const auto max_abs = [](const auto& tensor) {
    double value = 0.0;
    for (const double component : tensor) {
      value = std::max(value, std::abs(component));
    }
    return value;
  };
  const double matrix_scale = max_abs(matrix);
  const double inverse_scale = max_abs(result.inverse_jacobian);
  const double velocity_scale = std::max(max_abs(result.velocity_matrix),
                                         max_abs(result.velocity_offset));
  for (size_t p = 0; p < get<0>(sample_points).size(); ++p) {
    tnsr::I<double, 3, Frame::Grid> point{};
    for (size_t d = 0; d < 3; ++d) {
      point.get(d) = sample_points.get(d)[p];
    }
    const double point_scale = 1.0 + max_abs(point);
    const auto [mapped, inv_jac, jac, velocity] = at(point);
    for (size_t i = 0; i < 3; ++i) {
      double expected_mapped = offset.get(i);
      double expected_velocity = result.velocity_offset.get(i);
      for (size_t j = 0; j < 3; ++j) {
        expected_mapped += matrix.get(i, j) * point.get(j);
        expected_velocity += result.velocity_matrix.get(i, j) * point.get(j);
        if (std::abs(inv_jac.get(j, i) - inverse_jacobian.get(j, i)) >
            tolerance * inverse_scale) {
          ERROR(
              "The grid to inertial map isn't affine: its inverse Jacobian "
              "at the grid point "
              << point << " is " << inv_jac << " but at the origin it is "
              << inverse_jacobian
              << ". FD patches only support affine grid to inertial maps.");
        }
      }
      if (std::abs(mapped.get(i) - expected_mapped) >
              tolerance * matrix_scale * point_scale or
          std::abs(velocity.get(i) - expected_velocity) >
              tolerance * velocity_scale * point_scale) {
        ERROR("The grid to inertial map isn't affine at the grid point "
              << point << ", which it maps to " << mapped
              << " with mesh velocity " << velocity
              << ". FD patches only support affine grid to inertial maps.");
      }
    }
  }
  return result;
}

void inertial_geometry_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<inertial_face_tags>*> faces,
    const gsl::not_null<BatchedVariables<inertial_cell_tags>*> cells,
    const LogicalToGrid& logical_to_grid,
    const Kokkos::View<const AffineGridToInertial*>& grid_to_inertial) {
  ASSERT(faces->number_of_grid_points() ==
                 logical_to_grid.faces.number_of_grid_points() and
             faces->number_of_elements() ==
                 logical_to_grid.faces.number_of_elements() and
             cells->number_of_grid_points() ==
                 logical_to_grid.cells.number_of_grid_points() and
             cells->number_of_elements() ==
                 logical_to_grid.cells.number_of_elements() and
             grid_to_inertial.extent(0) ==
                 logical_to_grid.faces.number_of_elements(),
         "The inertial geometry must have the shape of the logical to grid "
         "geometry, with a map per face set");
  apply_pointwise_on_device<InertialFaceGeometry>(
      exec, faces, InertialFaceGeometry::return_tags{},
      InertialFaceGeometry::argument_tags{},
      make_device_arguments<Tags::LogicalToGridFaceConormal,
                            Tags::LogicalToGridFaceDetInverseJacobian,
                            Tags::FaceGridCoordinates,
                            ::Tags::PerElement<Tags::AffineGridToInertial>>(
          exec,
          get<::Tags::BatchView<Tags::LogicalToGridFaceConormal>>(
              logical_to_grid.faces),
          get<::Tags::BatchView<Tags::LogicalToGridFaceDetInverseJacobian>>(
              logical_to_grid.faces),
          get<::Tags::BatchView<Tags::FaceGridCoordinates>>(
              logical_to_grid.faces),
          grid_to_inertial));
  apply_pointwise_on_device<InertialCellGeometry>(
      exec, cells, InertialCellGeometry::return_tags{},
      InertialCellGeometry::argument_tags{},
      make_device_arguments<Tags::LogicalToGridCellDetInverseJacobian,
                            ::Tags::PerElement<Tags::AffineGridToInertial>>(
          exec,
          get<::Tags::BatchView<Tags::LogicalToGridCellDetInverseJacobian>>(
              logical_to_grid.cells),
          // The first face sets are those of the elements in order
          Kokkos::View<const AffineGridToInertial*>{Kokkos::subview(
              grid_to_inertial,
              Kokkos::make_pair(size_t{0},
                                logical_to_grid.cells.number_of_elements()))}));
}
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
