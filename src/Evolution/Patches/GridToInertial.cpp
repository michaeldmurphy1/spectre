// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/GridToInertial.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::patches {
AffineGridToInertial affine_grid_to_inertial(
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    const double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
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
  const size_t number_of_sample_points = get<0>(sample_points).size();
  if (number_of_sample_points == 0) {
    return result;
  }
  // One evaluation at all sample points, which is much cheaper than one per
  // point
  const auto [all_mapped, all_inv_jac, all_jac, all_velocities] =
      grid_to_inertial.coords_frame_velocity_jacobians(sample_points, time,
                                                       functions_of_time);
  (void)all_jac;
  for (size_t p = 0; p < number_of_sample_points; ++p) {
    tnsr::I<double, 3, Frame::Grid> point{};
    for (size_t d = 0; d < 3; ++d) {
      point.get(d) = sample_points.get(d)[p];
    }
    const double point_scale = 1.0 + max_abs(point);
    for (size_t i = 0; i < 3; ++i) {
      double expected_mapped = offset.get(i);
      double expected_velocity = result.velocity_offset.get(i);
      for (size_t j = 0; j < 3; ++j) {
        expected_mapped += matrix.get(i, j) * point.get(j);
        expected_velocity += result.velocity_matrix.get(i, j) * point.get(j);
        if (std::abs(all_inv_jac.get(j, i)[p] - inverse_jacobian.get(j, i)) >
            tolerance * inverse_scale) {
          ERROR(
              "The grid to inertial map isn't affine: its inverse Jacobian "
              "at the grid point "
              << point << " differs from that at the origin, "
              << inverse_jacobian
              << ". FD patches only support affine grid to inertial maps.");
        }
      }
      if (std::abs(all_mapped.get(i)[p] - expected_mapped) >
              tolerance * matrix_scale * point_scale or
          std::abs(all_velocities.get(i)[p] - expected_velocity) >
              tolerance * velocity_scale * point_scale) {
        tnsr::I<double, 3, Frame::Inertial> mapped{};
        tnsr::I<double, 3, Frame::Inertial> velocity{};
        for (size_t d = 0; d < 3; ++d) {
          mapped.get(d) = all_mapped.get(d)[p];
          velocity.get(d) = all_velocities.get(d)[p];
        }
        ERROR("The grid to inertial map isn't affine at the grid point "
              << point << ", which it maps to " << mapped
              << " with mesh velocity " << velocity
              << ". FD patches only support affine grid to inertial maps.");
      }
    }
  }
  return result;
}

::domain::CoordinateMaps::DeviceMap<3> device_grid_to_inertial(
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    const double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
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

}  // namespace evolution::patches
