// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/TagsDomain.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
namespace fd_patch = evolution::dg::subcell::fd_patch;
using GridToInertial =
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>;

template <typename View>
DataVector to_data_vector(const View& view) {
  DataVector result(view.extent(0));
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = view(i);
  }
  return result;
}

// The elements of a curved block, so the inverse Jacobians aren't diagonal,
// or of an affine block
std::vector<ElementMap<3, Frame::Grid>> make_element_maps(const bool curved) {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  std::vector<ElementMap<3, Frame::Grid>> result{};
  for (size_t index = 0; index < 4; ++index) {
    const ElementId<3> id{
        0,
        {{SegmentId{1, index % 2}, SegmentId{1, index / 2}, SegmentId{0, 0}}}};
    if (curved) {
      result.emplace_back(
          id,
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Wedge<3>{
                  1.0, 3.0, 0.0, 1.0, OrientationMap<3>::create_aligned(),
                  true}));
    } else {
      result.emplace_back(
          id,
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              Affine3D{Affine{-1.0, 1.0, 2.0, 4.0},
                       Affine{-1.0, 1.0, -3.0, -1.5},
                       Affine{-1.0, 1.0, 0.5, 3.0}}));
    }
  }
  return result;
}

domain::FunctionsOfTimeMap make_functions_of_time(const double time,
                                                  const bool uniform) {
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  domain::FunctionsOfTimeMap result{};
  result["ExpansionA"] = std::make_unique<Polynomial>(
      0.0, std::array<DataVector, 3>{{{1.1}, {0.2}, {0.03}}}, time + 1.0);
  result["ExpansionB"] = std::make_unique<Polynomial>(
      0.0, std::array<DataVector, 3>{{{uniform ? 1.1 : 1.05}, {0.2}, {0.03}}},
      time + 1.0);
  result["Translation"] = std::make_unique<Polynomial>(
      0.0,
      std::array<DataVector, 3>{
          {{0.1, -0.2, 0.3}, {0.4, 0.5, -0.6}, {0.01, 0.02, 0.03}}},
      time + 1.0);
  return result;
}

// A rotation, a uniform expansion and a translation, so the map is affine
// with a matrix that isn't symmetric
std::unique_ptr<GridToInertial> make_grid_to_inertial() {
  return domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
      domain::CoordinateMaps::Rotation<3>{0.3, -0.5, 0.7},
      domain::CoordinateMaps::TimeDependent::CubicScale<3>{10.0, "ExpansionA",
                                                           "ExpansionB"},
      domain::CoordinateMaps::TimeDependent::Translation<3>{"Translation"});
}

// Compares the inertial geometry on the device with the host computation of
// `grmhd::ValenciaDivClean::subcell::TimeDerivative` and the subcell compute
// tags.
void check_inertial_geometry(
    const std::vector<ElementMap<3, Frame::Grid>>& element_maps,
    const Mesh<3>& dg_mesh, const GridToInertial& grid_to_inertial,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const bool curved) {
  const Kokkos::DefaultExecutionSpace exec{};
  const size_t number_of_elements = element_maps.size();
  const auto logical_to_grid =
      fd_patch::logical_to_grid(exec, element_maps, dg_mesh);
  CHECK(get<0>(logical_to_grid.corners).size() == 8 * number_of_elements);
  const auto affine = fd_patch::affine_grid_to_inertial(
      grid_to_inertial, time, functions_of_time, logical_to_grid.corners);
  CHECK(affine.is_moving == not grid_to_inertial.is_identity());

  BatchedVariables<fd_patch::inertial_face_tags> faces{
      exec, logical_to_grid.faces.number_of_grid_points(),
      3 * number_of_elements};
  BatchedVariables<fd_patch::inertial_cell_tags> cells{
      exec, logical_to_grid.cells.number_of_grid_points(), number_of_elements};
  fd_patch::inertial_geometry_on_device(exec, make_not_null(&faces),
                                        make_not_null(&cells), logical_to_grid,
                                        affine);
  BatchedVariables<fd_patch::inertial_face_tags, Kokkos::HostSpace> host_faces{
      Kokkos::DefaultHostExecutionSpace{}, faces.number_of_grid_points(),
      faces.number_of_elements()};
  BatchedVariables<fd_patch::inertial_cell_tags, Kokkos::HostSpace> host_cells{
      Kokkos::DefaultHostExecutionSpace{}, cells.number_of_grid_points(),
      cells.number_of_elements()};
  deep_copy(exec, make_not_null(&host_faces), faces);
  deep_copy(exec, make_not_null(&host_cells), cells);
  exec.fence();

  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto dg_logical_coords = logical_coordinates(dg_mesh);
  const auto cell_logical_coords = logical_coordinates(subcell_mesh);
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const auto& element_map = element_maps[e];
    // The logical to inertial inverse Jacobian as
    // `fd::Tags::InverseJacobianLogicalToInertialCompute` and the domain
    // compute tags
    const auto logical_to_inertial_inv_jacobian =
        [&](const tnsr::I<DataVector, 3, Frame::ElementLogical>& logical) {
          const auto grid_coords = element_map(logical);
          const auto grid_inv_jacobian = element_map.inv_jacobian(logical);
          const auto [inertial_coords, inertial_inv_jacobian, jacobian,
                      mesh_velocity] =
              grid_to_inertial.coords_frame_velocity_jacobians(
                  grid_coords, time, functions_of_time);
          InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
              inv_jacobian{get<0>(logical).size(), 0.0};
          for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
              for (size_t k = 0; k < 3; ++k) {
                inv_jacobian.get(i, j) += grid_inv_jacobian.get(i, k) *
                                          inertial_inv_jacobian.get(k, j);
              }
            }
          }
          return std::tuple{inv_jacobian, mesh_velocity, inertial_coords};
        };

    const auto [cell_inv_jacobian, cell_mesh_velocity, cell_coords] =
        logical_to_inertial_inv_jacobian(cell_logical_coords);
    const auto element_cells = host_cells.element(e);
    CHECK_ITERABLE_CUSTOM_APPROX(
        to_data_vector(
            get(get<::Tags::MirrorView<fd_patch::Tags::CellDetInverseJacobian,
                                       Kokkos::HostSpace>>(element_cells))),
        get(determinant(cell_inv_jacobian)), custom_approx);

    const auto [dg_inv_jacobian, dg_mesh_velocity, dg_inertial_coords] =
        logical_to_inertial_inv_jacobian(dg_logical_coords);
    const auto dg_det_inv_jacobian = determinant(dg_inv_jacobian);
    // The host computes the divergence of the mesh velocity with spectral
    // derivatives, which are only exact for affine element maps
    if (affine.is_moving and not curved) {
      std::optional<Scalar<DataVector>> div_mesh_velocity{};
      evolution::domain::Tags::DivMeshVelocityCompute<3>::function(
          make_not_null(&div_mesh_velocity), std::optional{dg_mesh_velocity},
          dg_mesh, dg_inv_jacobian, dg_inertial_coords);
      CHECK_ITERABLE_CUSTOM_APPROX(
          get(div_mesh_velocity.value()),
          DataVector(dg_mesh.number_of_grid_points(), affine.div_mesh_velocity),
          custom_approx);
    }
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      Index<3> face_mesh_extents = subcell_mesh.extents();
      face_mesh_extents[i] += 1;
      const auto element_faces = host_faces.element(i * number_of_elements + e);
      for (size_t j = 0; j < 3; ++j) {
        CAPTURE(j);
        CHECK_ITERABLE_CUSTOM_APPROX(
            to_data_vector(
                get<::Tags::MirrorView<fd_patch::Tags::FaceConormal,
                                       Kokkos::HostSpace>>(element_faces)
                    .get(j)),
            evolution::dg::subcell::fd::project_to_faces(
                dg_inv_jacobian.get(i, j), dg_mesh, face_mesh_extents, i,
                (i == 0) != (j == 0) ? Spectral::Parity::Odd
                                     : Spectral::Parity::Even),
            custom_approx);
        const DataVector expected_mesh_velocity =
            affine.is_moving
                ? evolution::dg::subcell::fd::project_to_faces(
                      dg_mesh_velocity.get(j), dg_mesh, face_mesh_extents, i,
                      j == 0 ? Spectral::Parity::Odd : Spectral::Parity::Even)
                : DataVector(face_mesh_extents.product(), 0.0);
        CHECK_ITERABLE_CUSTOM_APPROX(
            to_data_vector(
                get<::Tags::MirrorView<fd_patch::Tags::FaceMeshVelocity,
                                       Kokkos::HostSpace>>(element_faces)
                    .get(j)),
            expected_mesh_velocity, custom_approx);
      }
      CHECK_ITERABLE_CUSTOM_APPROX(
          to_data_vector(
              get(get<::Tags::MirrorView<fd_patch::Tags::FaceDetInverseJacobian,
                                         Kokkos::HostSpace>>(element_faces))),
          evolution::dg::subcell::fd::project_to_faces(
              get(dg_det_inv_jacobian), dg_mesh, face_mesh_extents, i,
              Spectral::Parity::Even),
          custom_approx);
    }
  }
}

void test_geometry() {
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const double time = 0.7;
  for (const bool curved : {true, false}) {
    CAPTURE(curved);
    const auto element_maps = make_element_maps(curved);
    check_inertial_geometry(
        element_maps, dg_mesh,
        *domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{}),
        time, {}, curved);
    check_inertial_geometry(element_maps, dg_mesh, *make_grid_to_inertial(),
                            time, make_functions_of_time(time, true), curved);
  }
}

void test_non_affine_map() {
  const Kokkos::DefaultExecutionSpace exec{};
  const Mesh<3> dg_mesh{4, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto logical_to_grid =
      fd_patch::logical_to_grid(exec, make_element_maps(true), dg_mesh);
  const double time = 0.7;
  CHECK_THROWS_WITH(
      fd_patch::affine_grid_to_inertial(*make_grid_to_inertial(), time,
                                        make_functions_of_time(time, false),
                                        logical_to_grid.corners),
      Catch::Matchers::ContainsSubstring(
          "FD patches only support affine grid to inertial maps"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.GeometryKokkos",
                  "[Evolution][Unit]") {
  test_geometry();
  test_non_affine_map();
}
