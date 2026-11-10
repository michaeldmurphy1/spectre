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
#include "Utilities/ErrorHandling/Error.hpp"

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
}  // namespace evolution::patches
