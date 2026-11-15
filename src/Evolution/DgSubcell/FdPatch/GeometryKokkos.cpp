// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/ApplyMatricesKokkos.hpp"
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InertialGeometryKokkos.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
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
    const Mesh<3>& dg_mesh, const bool with_evaluation_points) {
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
  using PointTags = ::domain::logical_to_grid_geometry_tags<3>;
  const size_t number_of_point_elements =
      with_evaluation_points ? number_of_elements : 0;
  BatchedVariables<PointTags, Kokkos::HostSpace> host_cell_points{
      Kokkos::DefaultHostExecutionSpace{}, subcell_mesh.number_of_grid_points(),
      number_of_point_elements};
  BatchedVariables<PointTags, Kokkos::HostSpace> host_dg_points{
      Kokkos::DefaultHostExecutionSpace{}, dg_mesh.number_of_grid_points(),
      number_of_point_elements};
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

    // As the subcell and domain compute tags
    if (with_evaluation_points) {
      Variables<PointTags> cell_points{};
      set_data_ref_to_element(make_not_null(&cell_points), host_cell_points, e);
      get<::domain::Tags::Coordinates<3, Frame::Grid>>(cell_points) =
          element_map(cell_logical_coords);
      get<::domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                          Frame::Grid>>(cell_points) =
          element_map.inv_jacobian(cell_logical_coords);
      Variables<PointTags> dg_points{};
      set_data_ref_to_element(make_not_null(&dg_points), host_dg_points, e);
      get<::domain::Tags::Coordinates<3, Frame::Grid>>(dg_points) =
          element_map(dg_logical_coords);
      get<::domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                          Frame::Grid>>(dg_points) =
          element_map.inv_jacobian(dg_logical_coords);
    }

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
  result.cell_points = BatchedVariables<PointTags>{
      exec, subcell_mesh.number_of_grid_points(), number_of_point_elements};
  result.dg_points = BatchedVariables<PointTags>{
      exec, dg_mesh.number_of_grid_points(), number_of_point_elements};
  deep_copy(exec, make_not_null(&result.faces), host_faces);
  deep_copy(exec, make_not_null(&result.cells), host_cells);
  deep_copy(exec, make_not_null(&result.cell_points), host_cell_points);
  deep_copy(exec, make_not_null(&result.dg_points), host_dg_points);
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

::domain::CoordinateMaps::DeviceMap<3> device_grid_to_inertial(
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const tnsr::I<DataVector, 3, Frame::Grid>& sample_points) {
  if (auto device_map = grid_to_inertial.device_map(time, functions_of_time);
      device_map.has_value()) {
    return *device_map;
  }
  // Errors if the map isn't affine
  const AffineGridToInertial affine = affine_grid_to_inertial(
      grid_to_inertial, time, functions_of_time, sample_points);
  ::domain::CoordinateMaps::DeviceMap<3> result{};
  if (not affine.is_moving) {
    return result;
  }
  const auto [offset, inverse_jacobian, jacobian, velocity] =
      grid_to_inertial.coords_frame_velocity_jacobians(
          tnsr::I<double, 3, Frame::Grid>{0.0}, time, functions_of_time);
  ::domain::CoordinateMaps::DeviceAffineMap<3> map{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(map.offset, i) = offset.get(i);
    gsl::at(map.offset_derivative, i) = affine.velocity_offset.get(i);
    for (size_t j = 0; j < 3; ++j) {
      map.matrix.get(i, j) = jacobian.get(i, j);
      map.matrix_derivative.get(i, j) = affine.velocity_matrix.get(i, j);
    }
  }
  result.push_back(map);
  return result;
}

namespace {
// The divergence of the mesh velocity from its logical derivatives, as
// `divergence`
struct DivMeshVelocity {
  using return_tags = tmpl::list<::Tags::TempScalar<0>>;
  using argument_tags =
      tmpl::list<::domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                                 Frame::Inertial>,
                 ::Tags::TempI<0, 3, Frame::Inertial>,
                 ::Tags::TempI<1, 3, Frame::Inertial>,
                 ::Tags::TempI<2, 3, Frame::Inertial>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> div_mesh_velocity,
      const InverseJacobian<double, 3, Frame::ElementLogical, Frame::Inertial>&
          inverse_jacobian,
      const tnsr::I<double, 3, Frame::Inertial>& d_xi_mesh_velocity,
      const tnsr::I<double, 3, Frame::Inertial>& d_eta_mesh_velocity,
      const tnsr::I<double, 3, Frame::Inertial>& d_zeta_mesh_velocity) {
    const std::array<const tnsr::I<double, 3, Frame::Inertial>*, 3>
        logical_derivatives{&d_xi_mesh_velocity, &d_eta_mesh_velocity,
                            &d_zeta_mesh_velocity};
    get(*div_mesh_velocity) = 0.0;
    for (size_t i0 = 0; i0 < 3; ++i0) {
      for (size_t d = 0; d < 3; ++d) {
        get(*div_mesh_velocity) += inverse_jacobian.get(d, i0) *
                                   gsl::at(logical_derivatives, d)->get(i0);
      }
    }
  }
};

// The components of `Tag` in a `BatchedVariables` with `Tags`
template <typename Tag, typename... Tags>
size_t first_component(tmpl::list<Tags...> /*meta*/) {
  size_t result = 0;
  bool found = false;
  const auto count =
      [&result, &found]<typename LocalTag>(tmpl::type_<LocalTag> /*meta*/) {
        if (std::is_same_v<LocalTag, Tag>) {
          found = true;
        }
        if (not found) {
          result += LocalTag::type::size();
        }
      };
  (count(tmpl::type_<Tags>{}), ...);
  return result;
}
}  // namespace

TimeDependentGeometry::TimeDependentGeometry(
    const Kokkos::DefaultExecutionSpace& exec, const Mesh<3>& dg_mesh,
    const size_t number_of_elements)
    : number_of_elements_(number_of_elements) {
  for (size_t d = 0; d < 3; ++d) {
    if (dg_mesh.basis(d) != Spectral::Basis::Legendre) {
      ERROR("FD patches only support Legendre bases, but the DG mesh is "
            << dg_mesh);
    }
  }
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const Index<3>& subcell_extents = subcell_mesh.extents();
  const size_t dg_points = dg_mesh.number_of_grid_points();
  dg_geometry_ = decltype(dg_geometry_){exec, dg_points, number_of_elements};
  dg_div_mesh_velocity_ =
      decltype(dg_div_mesh_velocity_){exec, dg_points, number_of_elements};
  dg_logical_derivatives_ =
      decltype(dg_logical_derivatives_){exec, dg_points, number_of_elements};

  // The components of the geometry on the DG grid
  using dg_tags = ::domain::inertial_geometry_tags<3, true>;
  using InverseJacobianTag =
      ::domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                      Frame::Inertial>;
  const size_t inverse_jacobian_component =
      first_component<InverseJacobianTag>(dg_tags{});
  const size_t det_component = first_component<
      ::domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
      dg_tags{});
  const size_t mesh_velocity_component =
      first_component<::domain::Tags::MeshVelocityTensor<3>>(dg_tags{});
  const std::vector<size_t> mesh_velocity_components{
      mesh_velocity_component, mesh_velocity_component + 1,
      mesh_velocity_component + 2};

  const auto projection = [&dg_mesh, &subcell_extents](
                              const size_t d,
                              const Spectral::Quadrature quadrature) {
    // Face-centered points include both ends of the element
    return evolution::dg::subcell::fd::projection_matrix(
        dg_mesh.slice_through(d),
        subcell_extents[d] +
            (quadrature == Spectral::Quadrature::FaceCentered ? 1 : 0),
        quadrature, Spectral::Parity::Uninitialized);
  };
  std::array<Matrix, 3> to_cells{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(to_cells, d) = projection(d, Spectral::Quadrature::CellCentered);
    std::array<Matrix, 3> derivative{};
    gsl::at(derivative, d) =
        Spectral::differentiation_matrix(dg_mesh.slice_through(d));
    gsl::at(logical_derivatives_, d) = DeviceApplyMatrices{
        derivative, dg_mesh.extents(), mesh_velocity_components};
  }
  mesh_velocity_to_cells_ = DeviceApplyMatrices{to_cells, dg_mesh.extents(),
                                                mesh_velocity_components};
  div_mesh_velocity_to_cells_ =
      DeviceApplyMatrices{to_cells, dg_mesh.extents()};
  const InverseJacobian<double, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{};
  for (size_t d = 0; d < 3; ++d) {
    std::array<Matrix, 3> to_faces = to_cells;
    gsl::at(to_faces, d) = projection(d, Spectral::Quadrature::FaceCentered);
    // As `inertial_face_tags`: row `d` of the inverse Jacobian, its
    // determinant and the mesh velocity
    std::vector<size_t> components{};
    components.reserve(7);
    for (size_t j = 0; j < 3; ++j) {
      components.push_back(inverse_jacobian_component +
                           inverse_jacobian.get_storage_index(d, j));
    }
    components.push_back(det_component);
    components.insert(components.end(), mesh_velocity_components.begin(),
                      mesh_velocity_components.end());
    gsl::at(to_faces_, d) =
        DeviceApplyMatrices{to_faces, dg_mesh.extents(), components};
  }
}

void TimeDependentGeometry::compute_cells(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<cell_geometry_tags>*> cell_geometry,
    const gsl::not_null<BatchedVariables<cell_mesh_velocity_tags>*>
        cell_mesh_velocity,
    const LogicalToGrid& logical_to_grid,
    const Kokkos::View<const ::domain::CoordinateMaps::DeviceMap<3>*>&
        grid_to_inertial) {
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::TimeDependentGeometry::compute_cells"};
  ASSERT(grid_to_inertial.extent(0) == number_of_elements_ and
             cell_geometry->number_of_elements() == number_of_elements_ and
             cell_mesh_velocity->number_of_elements() == number_of_elements_,
         "Expected " << number_of_elements_ << " elements");
  ::domain::inertial_geometry_on_device<true>(
      exec, make_not_null(&dg_geometry_), logical_to_grid.dg_points,
      grid_to_inertial);
  ::domain::inertial_geometry_on_device<false>(
      exec, cell_geometry, logical_to_grid.cell_points, grid_to_inertial);

  // The mesh velocity and its divergence on the DG grid, projected
  const auto components = [](const auto& view, const size_t first,
                             const size_t last) {
    return Kokkos::subview(view, Kokkos::ALL(), Kokkos::make_pair(first, last),
                           Kokkos::ALL());
  };
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(logical_derivatives_, d)
        .apply(exec,
               components(dg_logical_derivatives_.view(), 3 * d, 3 * d + 3),
               dg_geometry_.view());
  }
  apply_pointwise_on_device<DivMeshVelocity>(
      exec, make_not_null(&dg_div_mesh_velocity_),
      DivMeshVelocity::return_tags{}, DivMeshVelocity::argument_tags{},
      make_device_arguments<::domain::Tags::InverseJacobian<
                                3, Frame::ElementLogical, Frame::Inertial>,
                            ::Tags::TempI<0, 3, Frame::Inertial>,
                            ::Tags::TempI<1, 3, Frame::Inertial>,
                            ::Tags::TempI<2, 3, Frame::Inertial>>(
          exec,
          get<::Tags::BatchView<::domain::Tags::InverseJacobian<
              3, Frame::ElementLogical, Frame::Inertial>>>(dg_geometry_),
          get<::Tags::BatchView<::Tags::TempI<0, 3, Frame::Inertial>>>(
              dg_logical_derivatives_),
          get<::Tags::BatchView<::Tags::TempI<1, 3, Frame::Inertial>>>(
              dg_logical_derivatives_),
          get<::Tags::BatchView<::Tags::TempI<2, 3, Frame::Inertial>>>(
              dg_logical_derivatives_)));
  mesh_velocity_to_cells_.apply(
      exec, components(cell_mesh_velocity->view(), 0, 3), dg_geometry_.view());
  div_mesh_velocity_to_cells_.apply(
      exec, components(cell_mesh_velocity->view(), 3, 4),
      dg_div_mesh_velocity_.view());
}

void TimeDependentGeometry::compute_faces(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<inertial_face_tags>*> faces) {
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::TimeDependentGeometry::compute_faces"};
  ASSERT(faces->number_of_elements() == 3 * number_of_elements_,
         "Expected " << 3 * number_of_elements_ << " face sets, not "
                     << faces->number_of_elements());
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(to_faces_, d)
        .apply(
            exec,
            Kokkos::subview(faces->view(), Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::make_pair(d * number_of_elements_,
                                              (d + 1) * number_of_elements_)),
            dg_geometry_.view());
  }
}
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
