// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/RotScaleTrans.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/InertialGeometryKokkos.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using FunctionsOfTimeMap = std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
using RotScaleTrans = domain::CoordinateMaps::TimeDependent::RotScaleTrans<3>;

FunctionsOfTimeMap make_functions_of_time() {
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<3>;
  FunctionsOfTimeMap functions_of_time{};
  functions_of_time["ExpansionA"] = std::make_unique<Polynomial>(
      0.0, std::array<DataVector, 4>{{{0.9}, {-0.02}, {0.003}, {0.0}}}, 10.0);
  functions_of_time["ExpansionB"] = std::make_unique<Polynomial>(
      0.0, std::array<DataVector, 4>{{{1.0}, {-0.001}, {0.0}, {0.0}}}, 10.0);
  functions_of_time["Translation"] = std::make_unique<Polynomial>(
      0.0,
      std::array<DataVector, 4>{{{0.3, -0.4, 0.2},
                                 {-0.2, 0.1, 0.05},
                                 {0.01, 0.01, 0.01},
                                 {0.0, 0.0, 0.0}}},
      10.0);
  DataVector axis{{1.0, -2.0, 0.5}};
  axis /= sqrt(5.25);
  const double angle = 0.7;
  functions_of_time["Rotation"] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          0.0,
          std::array<DataVector, 1>{DataVector{
              {cos(0.5 * angle), axis[0] * sin(0.5 * angle),
               axis[1] * sin(0.5 * angle), axis[2] * sin(0.5 * angle)}}},
          std::array<DataVector, 4>{axis * angle, axis * 0.3, axis * -0.05,
                                    axis * 0.0},
          10.0);
  return functions_of_time;
}

// Two elements of a curved block in the transition region of a moving map,
// one with the map and one without, mapped on the device and on the host
template <bool WithMeshVelocity>
void test_inertial_geometry() {
  using Affine = domain::CoordinateMaps::Affine;
  using Equiangular = domain::CoordinateMaps::Equiangular;
  const auto block_map = [] {
    return domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
        domain::CoordinateMaps::ProductOf3Maps<Equiangular, Affine, Affine>{
            Equiangular{-1.0, 1.0, 5.0, 9.0}, Affine{-1.0, 1.0, -2.0, 1.0},
            Affine{-1.0, 1.0, -1.0, 3.0}});
  };
  const std::array element_maps{
      ElementMap<3, Frame::Grid>{ElementId<3>{0, {{{1, 0}, {1, 1}, {0, 0}}}},
                                 block_map()},
      ElementMap<3, Frame::Grid>{ElementId<3>{0, {{{1, 1}, {1, 0}, {0, 0}}}},
                                 block_map()}};
  const domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotScaleTrans>
      grid_to_inertial{RotScaleTrans{
          std::pair<std::string, std::string>{"ExpansionA", "ExpansionB"},
          "Rotation", "Translation", 2.0, 20.0,
          RotScaleTrans::BlockRegion::Transition}};
  const auto functions_of_time = make_functions_of_time();
  const double time = 1.3;

  const Mesh<3> mesh{5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t number_of_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);

  using InputTags = domain::logical_to_grid_geometry_tags<3>;
  using ResultTags = domain::inertial_geometry_tags<3, WithMeshVelocity>;
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<InputTags, Kokkos::HostSpace> host_input{
      Kokkos::DefaultHostExecutionSpace{}, number_of_points, 2};
  for (size_t e = 0; e < 2; ++e) {
    Variables<InputTags> element{};
    set_data_ref_to_element(make_not_null(&element), host_input, e);
    get<domain::Tags::Coordinates<3, Frame::Grid>>(element) =
        gsl::at(element_maps, e)(logical_coords);
    get<domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Grid>>(
        element) = gsl::at(element_maps, e).inv_jacobian(logical_coords);
  }
  BatchedVariables<InputTags> input{exec, number_of_points, 2};
  deep_copy(exec, make_not_null(&input), host_input);

  const Kokkos::View<domain::CoordinateMaps::DeviceMap<3>*> maps{"maps", 2};
  const auto host_maps = Kokkos::create_mirror_view(maps);
  host_maps(0) = grid_to_inertial.device_map(time, functions_of_time).value();
  host_maps(1) = domain::CoordinateMaps::DeviceMap<3>{};
  Kokkos::deep_copy(exec, maps, host_maps);

  BatchedVariables<ResultTags> result{exec, number_of_points, 2};
  domain::inertial_geometry_on_device<WithMeshVelocity>(
      exec, make_not_null(&result), input, maps);
  BatchedVariables<ResultTags, Kokkos::HostSpace> host_result{
      Kokkos::DefaultHostExecutionSpace{}, number_of_points, 2};
  deep_copy(exec, make_not_null(&host_result), result);
  exec.fence();

  for (size_t e = 0; e < 2; ++e) {
    CAPTURE(e);
    Variables<InputTags> element_input{};
    set_data_ref_to_element(make_not_null(&element_input), host_input, e);
    Variables<ResultTags> element_result{};
    set_data_ref_to_element(make_not_null(&element_result), host_result, e);
    const auto& grid_coords =
        get<domain::Tags::Coordinates<3, Frame::Grid>>(element_input);
    const auto& logical_to_grid_inverse_jacobian = get<
        domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Grid>>(
        element_input);

    // The host, as the DG compute tags
    std::optional<
        std::tuple<tnsr::I<DataVector, 3, Frame::Inertial>,
                   InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>,
                   Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>,
                   tnsr::I<DataVector, 3, Frame::Inertial>>>
        grid_to_inertial_quantities{};
    if (e == 0) {
      grid_to_inertial_quantities =
          grid_to_inertial.coords_frame_velocity_jacobians(grid_coords, time,
                                                           functions_of_time);
    }
    tnsr::I<DataVector, 3, Frame::Inertial> expected_coords{};
    domain::Tags::InertialFromGridCoordinatesCompute<3>::function(
        make_not_null(&expected_coords), grid_coords,
        grid_to_inertial_quantities);
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        expected_inverse_jacobian{};
    domain::Tags::ElementToInertialInverseJacobian<3>::function(
        make_not_null(&expected_inverse_jacobian),
        logical_to_grid_inverse_jacobian, grid_to_inertial_quantities);

    CHECK_ITERABLE_APPROX(
        (get<domain::Tags::Coordinates<3, Frame::Inertial>>(element_result)),
        expected_coords);
    CHECK_ITERABLE_APPROX(
        (get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                           Frame::Inertial>>(element_result)),
        expected_inverse_jacobian);
    CHECK_ITERABLE_APPROX(
        (get<domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                          Frame::Inertial>>(element_result)),
        determinant(expected_inverse_jacobian));
    if constexpr (WithMeshVelocity) {
      const auto& mesh_velocity =
          get<domain::Tags::MeshVelocityTensor<3>>(element_result);
      if (e == 0) {
        CHECK_ITERABLE_APPROX(mesh_velocity,
                              std::get<3>(*grid_to_inertial_quantities));
      } else {
        CHECK(mesh_velocity ==
              tnsr::I<DataVector, 3, Frame::Inertial>{number_of_points, 0.0});
      }
    }
    if (e == 1) {
      // Without a map the results are the grid frame values
      for (size_t i = 0; i < 3; ++i) {
        CHECK(get<domain::Tags::Coordinates<3, Frame::Inertial>>(element_result)
                  .get(i) == grid_coords.get(i));
        for (size_t j = 0; j < 3; ++j) {
          CHECK(get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                                  Frame::Inertial>>(
                    element_result)
                    .get(i, j) == logical_to_grid_inverse_jacobian.get(i, j));
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.InertialGeometry.Kokkos", "[Domain][Unit]") {
  test_inertial_geometry<false>();
  test_inertial_geometry<true>();
}
