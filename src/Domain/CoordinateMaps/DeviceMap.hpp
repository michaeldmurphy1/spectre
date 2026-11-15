// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/DeviceRotScaleTrans.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace domain::CoordinateMaps {
/// The kinds of maps a `DeviceMap` can hold
enum class DeviceMapKind { RotScaleTrans };

/// One map of a `DeviceMap`: its kind and its data
template <size_t Dim>
struct DeviceMapStage {
  DeviceMapKind kind = DeviceMapKind::RotScaleTrans;
  TimeDependent::DeviceRotScaleTrans<Dim> rot_scale_trans{};
};

/// The mapped point, the inverse Jacobian, the Jacobian and the frame velocity
/// of a `DeviceMap` at a point, as
/// `CoordinateMap::coords_frame_velocity_jacobians`
template <size_t Dim>
struct DeviceMappedPoint {
  tnsr::I<double, Dim, Frame::NoFrame> coords{};
  tnsr::Ij<double, Dim, Frame::NoFrame> inverse_jacobian{};
  tnsr::Ij<double, Dim, Frame::NoFrame> jacobian{};
  tnsr::I<double, Dim, Frame::NoFrame> frame_velocity{};
};

/*!
 * \brief A `CoordinateMap` at one time, in a form that device kernels can
 * evaluate.
 *
 * \details The host gets this from `CoordinateMapBase::device_map` at a time.
 * It holds the maps that aren't the identity, in the order they are applied,
 * each with its data at that time, so it can be copied to the device as plain
 * data. Since every map is held by value with a tag of its kind (rather than
 * by its type), the maps of the elements of different blocks can be evaluated
 * in one kernel.
 *
 * `operator()` evaluates the maps like
 * `CoordinateMap::coords_frame_velocity_jacobians`, so the two agree to
 * roundoff.
 */
template <size_t Dim>
struct DeviceMap {
  static constexpr size_t maximum_number_of_stages = 4;

  size_t number_of_stages = 0;
  std::array<DeviceMapStage<Dim>, maximum_number_of_stages> stages{};

  /// Append a map, which is applied after the maps already held
  void push_back(const TimeDependent::DeviceRotScaleTrans<Dim>& map) {
    DeviceMapStage<Dim>& stage = next_stage();
    stage.kind = DeviceMapKind::RotScaleTrans;
    stage.rot_scale_trans = map;
  }

  /// Whether the map is the identity, i.e. holds no maps
  bool is_identity() const { return number_of_stages == 0; }

  /// Whether the map is affine, i.e. its Jacobian doesn't depend on the point
  bool is_affine() const {
    for (size_t i = 0; i < number_of_stages; ++i) {
      const DeviceMapStage<Dim>& stage = stages[i];
      switch (stage.kind) {
        case DeviceMapKind::RotScaleTrans:
          if (not stage.rot_scale_trans.is_affine()) {
            return false;
          }
          break;
        default:
          ERROR("Unknown kind of map " << static_cast<int>(stage.kind));
      }
    }
    return true;
  }

  /// Evaluate the map at `source_point`
  KOKKOS_FUNCTION DeviceMappedPoint<Dim> operator()(
      const tnsr::I<double, Dim, Frame::NoFrame>& source_point) const;

 private:
  DeviceMapStage<Dim>& next_stage() {
    if (number_of_stages == maximum_number_of_stages) {
      ERROR("A DeviceMap holds at most " << maximum_number_of_stages
                                         << " maps.");
    }
    return stages[number_of_stages++];
  }
};

namespace DeviceMap_detail {
// The inverse of the Jacobian, as `determinant_and_inverse`
KOKKOS_FUNCTION inline void invert(
    const gsl::not_null<tnsr::Ij<double, 3, Frame::NoFrame>*> inverse,
    const tnsr::Ij<double, 3, Frame::NoFrame>& jacobian) {
  const double t00 = get<0, 0>(jacobian);
  const double t01 = get<0, 1>(jacobian);
  const double t02 = get<0, 2>(jacobian);
  const double t10 = get<1, 0>(jacobian);
  const double t11 = get<1, 1>(jacobian);
  const double t12 = get<1, 2>(jacobian);
  const double t20 = get<2, 0>(jacobian);
  const double t21 = get<2, 1>(jacobian);
  const double t22 = get<2, 2>(jacobian);
  const double a = t11 * t22 - t12 * t21;
  const double b = t12 * t20 - t10 * t22;
  const double c = t10 * t21 - t11 * t20;
  const double det = t00 * a + t01 * b + t02 * c;
  const double one_over_det = 1.0 / det;
  get<0, 0>(*inverse) = a * one_over_det;
  get<0, 1>(*inverse) = (t21 * t02 - t22 * t01) * one_over_det;
  get<0, 2>(*inverse) = (t01 * t12 - t02 * t11) * one_over_det;
  get<1, 0>(*inverse) = b * one_over_det;
  get<1, 1>(*inverse) = (t22 * t00 - t20 * t02) * one_over_det;
  get<1, 2>(*inverse) = (t02 * t10 - t00 * t12) * one_over_det;
  get<2, 0>(*inverse) = c * one_over_det;
  get<2, 1>(*inverse) = (t20 * t01 - t21 * t00) * one_over_det;
  get<2, 2>(*inverse) = (t00 * t11 - t01 * t10) * one_over_det;
}

KOKKOS_FUNCTION inline void invert(
    const gsl::not_null<tnsr::Ij<double, 2, Frame::NoFrame>*> inverse,
    const tnsr::Ij<double, 2, Frame::NoFrame>& jacobian) {
  const double t00 = get<0, 0>(jacobian);
  const double t01 = get<0, 1>(jacobian);
  const double t10 = get<1, 0>(jacobian);
  const double t11 = get<1, 1>(jacobian);
  const double det = t00 * t11 - t01 * t10;
  const double one_over_det = 1.0 / det;
  get<0, 0>(*inverse) = t11 * one_over_det;
  get<0, 1>(*inverse) = -t01 * one_over_det;
  get<1, 0>(*inverse) = -t10 * one_over_det;
  get<1, 1>(*inverse) = t00 * one_over_det;
}
}  // namespace DeviceMap_detail

template <size_t Dim>
KOKKOS_FUNCTION DeviceMappedPoint<Dim> DeviceMap<Dim>::operator()(
    const tnsr::I<double, Dim, Frame::NoFrame>& source_point) const {
  DeviceMappedPoint<Dim> result{};
  std::array<double, Dim> point{};
  for (size_t i = 0; i < Dim; ++i) {
    point[i] = source_point.get(i);
    result.frame_velocity.get(i) = 0.0;
    for (size_t j = 0; j < Dim; ++j) {
      result.jacobian.get(i, j) = i == j ? 1.0 : 0.0;
    }
  }
  tnsr::Ij<double, Dim, Frame::NoFrame> stage_jacobian{};
  std::array<double, Dim> stage_frame_velocity{};
  for (size_t stage_index = 0; stage_index < number_of_stages; ++stage_index) {
    const DeviceMapStage<Dim>& stage = stages[stage_index];
    switch (stage.kind) {
      case DeviceMapKind::RotScaleTrans:
        stage.rot_scale_trans.apply(make_not_null(&point),
                                    make_not_null(&stage_jacobian),
                                    make_not_null(&stage_frame_velocity));
        break;
      default:
        // Unreachable, since `push_back` sets the kind
        break;
    }
    // Compose as `CoordinateMap::coords_frame_velocity_jacobians`. All maps
    // here are time dependent.
    if (stage_index == 0) {
      result.jacobian = stage_jacobian;
      for (size_t i = 0; i < Dim; ++i) {
        result.frame_velocity.get(i) = stage_frame_velocity[i];
      }
    } else {
      std::array<double, Dim> temp{};
      for (size_t source = 0; source < Dim; ++source) {
        for (size_t target = 0; target < Dim; ++target) {
          temp[target] =
              stage_jacobian.get(target, 0) * result.jacobian.get(0, source);
          for (size_t dummy = 1; dummy < Dim; ++dummy) {
            temp[target] += stage_jacobian.get(target, dummy) *
                            result.jacobian.get(dummy, source);
          }
        }
        for (size_t target = 0; target < Dim; ++target) {
          result.jacobian.get(target, source) = temp[target];
        }
      }
      for (size_t target = 0; target < Dim; ++target) {
        for (size_t source = 0; source < Dim; ++source) {
          stage_frame_velocity[target] += stage_jacobian.get(target, source) *
                                          result.frame_velocity.get(source);
        }
      }
      for (size_t i = 0; i < Dim; ++i) {
        result.frame_velocity.get(i) = stage_frame_velocity[i];
      }
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    result.coords.get(i) = point[i];
  }
  DeviceMap_detail::invert(make_not_null(&result.inverse_jacobian),
                           result.jacobian);
  return result;
}
}  // namespace domain::CoordinateMaps
