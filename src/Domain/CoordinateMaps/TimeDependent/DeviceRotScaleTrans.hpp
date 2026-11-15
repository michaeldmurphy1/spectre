// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace domain::CoordinateMaps::TimeDependent {
/*!
 * \brief The `RotScaleTrans` map at one time, in a form that device kernels
 * can evaluate.
 *
 * \details The host fills this with `RotScaleTrans::device_map` from the
 * functions of time at that time. `apply` then evaluates the map, its Jacobian
 * and the frame velocity at a point with the same operations as the host's
 * `RotScaleTrans::operator()`, `RotScaleTrans::jacobian` and
 * `RotScaleTrans::frame_velocity`, so the two agree to roundoff (and are
 * equal on the host unless the compiler contracts operations differently).
 * See `RotScaleTrans` for the formulas.
 */
template <size_t Dim>
struct DeviceRotScaleTrans {
  /// As `RotScaleTrans::BlockRegion`
  enum class Region { Inner, Transition, Outer };

  bool has_rotation = false;
  bool has_expansion = false;
  bool has_translation = false;
  /// The rotation matrix \f$R(t)\f$ and its time derivative
  tnsr::Ij<double, Dim, Frame::NoFrame> rotation{};
  tnsr::Ij<double, Dim, Frame::NoFrame> rotation_derivative{};
  /// The expansion factors \f$E_a(t)\f$ and \f$E_b(t)\f$ and their time
  /// derivatives
  double expansion_a = 1.0;
  double expansion_b = 1.0;
  double expansion_a_derivative = 0.0;
  double expansion_b_derivative = 0.0;
  /// The translation \f$T(t)\f$ and its time derivative
  std::array<double, Dim> translation{};
  std::array<double, Dim> translation_derivative{};
  double inner_radius = 0.0;
  double outer_radius = 0.0;
  Region region = Region::Inner;

  /// Whether the map is affine, i.e. its Jacobian doesn't depend on the point
  bool is_affine() const {
    return region != Region::Transition or
           not(has_expansion or has_translation);
  }

  /*!
   * \brief Compute the Jacobian and the frame velocity of the map at `point`,
   * and then map `point`.
   */
  KOKKOS_FUNCTION void apply(
      gsl::not_null<std::array<double, Dim>*> point,
      gsl::not_null<tnsr::Ij<double, Dim, Frame::NoFrame>*> jacobian,
      gsl::not_null<std::array<double, Dim>*> frame_velocity) const;

 private:
  KOKKOS_FUNCTION bool closer_to_outer_radius(double radius) const {
    return 1.0 - radius / (inner_radius + outer_radius) < .5;
  }
  // The falloff of the expansion in the transition region, w_E in the
  // documentation of `RotScaleTrans`, for the expansion factors
  // `difference = E_a - E_b` or their derivatives
  KOKKOS_FUNCTION double expansion_falloff(double radius,
                                           double difference) const;
  // The falloff of the translation in the transition region, w_T
  KOKKOS_FUNCTION double translation_falloff(double radius) const;
  KOKKOS_FUNCTION void apply_jacobian(
      gsl::not_null<tnsr::Ij<double, Dim, Frame::NoFrame>*> jacobian,
      const std::array<double, Dim>& source, double radius) const;
  KOKKOS_FUNCTION void apply_frame_velocity(
      gsl::not_null<std::array<double, Dim>*> frame_velocity,
      const std::array<double, Dim>& source) const;
  KOKKOS_FUNCTION void apply_map(gsl::not_null<std::array<double, Dim>*> point,
                                 double radius) const;
};

template <size_t Dim>
KOKKOS_FUNCTION double DeviceRotScaleTrans<Dim>::expansion_falloff(
    const double radius, const double difference) const {
  return closer_to_outer_radius(radius)
             ? ((outer_radius - radius) * difference * inner_radius) /
                   ((outer_radius - inner_radius) * radius)
             : ((inner_radius - radius) * difference * outer_radius) /
                   ((outer_radius - inner_radius) * radius);
}

template <size_t Dim>
KOKKOS_FUNCTION double DeviceRotScaleTrans<Dim>::translation_falloff(
    const double radius) const {
  return closer_to_outer_radius(radius)
             ? (outer_radius - radius) / (outer_radius - inner_radius)
             : (inner_radius - radius) / (outer_radius - inner_radius) + 1.0;
}

template <size_t Dim>
KOKKOS_FUNCTION void DeviceRotScaleTrans<Dim>::apply(
    const gsl::not_null<std::array<double, Dim>*> point,
    const gsl::not_null<tnsr::Ij<double, Dim, Frame::NoFrame>*> jacobian,
    const gsl::not_null<std::array<double, Dim>*> frame_velocity) const {
  // As `magnitude`
  double radius_squared = (*point)[0] * (*point)[0];
  for (size_t i = 1; i < Dim; ++i) {
    radius_squared += (*point)[i] * (*point)[i];
  }
  const double radius = std::sqrt(radius_squared);
  apply_jacobian(jacobian, *point, radius);
  apply_frame_velocity(frame_velocity, *point);
  apply_map(point, radius);
}

template <size_t Dim>
KOKKOS_FUNCTION void DeviceRotScaleTrans<Dim>::apply_jacobian(
    const gsl::not_null<tnsr::Ij<double, Dim, Frame::NoFrame>*> jacobian,
    const std::array<double, Dim>& source, const double radius) const {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      jacobian->get(i, j) = i == j ? 1.0 : 0.0;
    }
  }
  const double scale =
      region == Region::Inner
          ? expansion_a
          : (region == Region::Outer or closer_to_outer_radius(radius)
                 ? expansion_b
                 : expansion_a);
  if (has_rotation and not has_expansion) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        jacobian->get(i, j) = rotation.get(i, j);
      }
    }
  } else if (has_expansion and region != Region::Transition) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        jacobian->get(i, j) =
            has_rotation ? scale * rotation.get(i, j) : (i == j ? scale : 0.0);
      }
    }
  } else if (has_expansion) {
    const double alpha = inner_radius * outer_radius /
                         ((radius * radius) * (inner_radius - outer_radius));
    const double radial_scaling_factor =
        expansion_falloff(radius, expansion_a - expansion_b);
    for (size_t i = 0; i < Dim; ++i) {
      if (has_rotation) {
        double rotated_coords = 0.0;
        for (size_t l = 0; l < Dim; ++l) {
          rotated_coords += rotation.get(i, l) * source[l];
        }
        for (size_t j = 0; j < Dim; ++j) {
          jacobian->get(i, j) = scale * rotation.get(i, j) +
                                alpha * rotated_coords *
                                    (expansion_a - expansion_b) * source[j] /
                                    radius +
                                rotation.get(i, j) * radial_scaling_factor;
        }
      } else {
        for (size_t j = 0; j < Dim; ++j) {
          jacobian->get(i, j) = alpha * source[i] *
                                (expansion_a - expansion_b) * source[j] /
                                radius;
        }
        jacobian->get(i, i) += scale + radial_scaling_factor;
      }
    }
  }
  if (has_translation and region == Region::Transition) {
    for (size_t i = 0; i < Dim; ++i) {
      const double deriv_translation_factor =
          (-translation[i] / (outer_radius - inner_radius));
      for (size_t j = 0; j < Dim; ++j) {
        jacobian->get(i, j) += deriv_translation_factor * source[j] / radius;
      }
    }
  }
}

template <size_t Dim>
KOKKOS_FUNCTION void DeviceRotScaleTrans<Dim>::apply_frame_velocity(
    const gsl::not_null<std::array<double, Dim>*> frame_velocity,
    const std::array<double, Dim>& source) const {
  // As `RotScaleTrans::frame_velocity`, which computes the radius differently
  // from the other functions
  double radius = source[0] * source[0];
  for (size_t i = 1; i < Dim; ++i) {
    radius += source[i] * source[i];
  }
  radius = std::sqrt(radius);
  for (size_t i = 0; i < Dim; ++i) {
    (*frame_velocity)[i] = 0.0;
  }
  const bool outer_scale =
      region == Region::Outer or
      (region == Region::Transition and closer_to_outer_radius(radius));
  const double scale = outer_scale ? expansion_b : expansion_a;
  const double dt_scale =
      outer_scale ? expansion_b_derivative : expansion_a_derivative;
  if (has_rotation and not has_expansion) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        (*frame_velocity)[i] += rotation_derivative.get(i, j) * source[j];
      }
    }
  } else if (has_expansion) {
    double radial_scaling_factor = 0.0;
    double deriv_radial_scaling_factor = 0.0;
    if (region == Region::Transition) {
      radial_scaling_factor =
          expansion_falloff(radius, expansion_a - expansion_b);
      deriv_radial_scaling_factor = expansion_falloff(
          radius, expansion_a_derivative - expansion_b_derivative);
    }
    for (size_t i = 0; i < Dim; ++i) {
      if (not has_rotation) {
        (*frame_velocity)[i] +=
            region == Region::Transition
                ? source[i] * (dt_scale + deriv_radial_scaling_factor)
                : source[i] * (dt_scale);
      } else {
        for (size_t j = 0; j < Dim; ++j) {
          (*frame_velocity)[i] +=
              region == Region::Transition
                  ? source[j] * (rotation_derivative.get(i, j) *
                                     (scale + radial_scaling_factor) +
                                 rotation.get(i, j) *
                                     (dt_scale + deriv_radial_scaling_factor))
                  : source[j] * (scale * rotation_derivative.get(i, j) +
                                 dt_scale * rotation.get(i, j));
        }
      }
    }
  }
  if (has_translation) {
    if (region == Region::Inner) {
      for (size_t i = 0; i < Dim; ++i) {
        (*frame_velocity)[i] += translation_derivative[i];
      }
    } else if (region == Region::Transition) {
      const double radial_translation_factor = translation_falloff(radius);
      for (size_t i = 0; i < Dim; ++i) {
        (*frame_velocity)[i] +=
            translation_derivative[i] * radial_translation_factor;
      }
    }
  }
}

template <size_t Dim>
KOKKOS_FUNCTION void DeviceRotScaleTrans<Dim>::apply_map(
    const gsl::not_null<std::array<double, Dim>*> point,
    const double radius) const {
  if (has_rotation) {
    const std::array<double, Dim> source = *point;
    for (size_t i = 0; i < Dim; ++i) {
      (*point)[i] = rotation.get(i, 0) * source[0];
      for (size_t j = 1; j < Dim; ++j) {
        (*point)[i] += rotation.get(i, j) * source[j];
      }
    }
  }
  if (has_expansion) {
    double factor = expansion_b;
    if (region == Region::Inner) {
      factor = expansion_a;
    } else if (region == Region::Transition) {
      factor = (closer_to_outer_radius(radius) ? expansion_b : expansion_a) +
               expansion_falloff(radius, expansion_a - expansion_b);
    }
    for (size_t i = 0; i < Dim; ++i) {
      (*point)[i] *= factor;
    }
  }
  if (has_translation) {
    if (region == Region::Inner) {
      for (size_t i = 0; i < Dim; ++i) {
        (*point)[i] += translation[i];
      }
    } else if (region == Region::Transition) {
      const double radial_translation_factor = translation_falloff(radius);
      for (size_t i = 0; i < Dim; ++i) {
        (*point)[i] += translation[i] * radial_translation_factor;
      }
    }
  }
}
}  // namespace domain::CoordinateMaps::TimeDependent
