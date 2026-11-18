// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/*!
 * \brief The 3+1 quantities of a spacetime metric at one point, for Kokkos
 * kernels.
 *
 * \details These compute the same values as the host functions they are named
 * after, with the same operations in the same order, so that device kernels
 * that use them agree with the host bitwise (up to fused multiply-adds, which
 * the compiler may form differently in vectorized host code). They are written
 * for `double`s in three dimensions and are force-inlined, since they run once
 * per grid point inside larger kernels.
 */
namespace gr::device {
/// The spatial metric, as `gr::spatial_metric`
KOKKOS_FORCEINLINE_FUNCTION tnsr::ii<double, 3> spatial_metric(
    const tnsr::aa<double, 3>& spacetime_metric) {
  tnsr::ii<double, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.get(i, j) = spacetime_metric.get(i + 1, j + 1);
    }
  }
  return result;
}

/// The determinant of the spatial metric and its inverse
struct DeterminantAndInverse {
  double determinant;
  tnsr::II<double, 3> inverse;
};

/// The determinant and inverse of the `spatial_metric`, as
/// `determinant_and_inverse`
KOKKOS_FORCEINLINE_FUNCTION DeterminantAndInverse
determinant_and_inverse(const tnsr::ii<double, 3>& spatial_metric) {
  const double t00 = get<0, 0>(spatial_metric);
  const double t01 = get<0, 1>(spatial_metric);
  const double t02 = get<0, 2>(spatial_metric);
  const double t11 = get<1, 1>(spatial_metric);
  const double t12 = get<1, 2>(spatial_metric);
  const double t22 = get<2, 2>(spatial_metric);
  const double a = t11 * t22 - t12 * t12;
  const double b = t12 * t02 - t01 * t22;
  const double c = t01 * t12 - t11 * t02;
  DeterminantAndInverse result{};
  result.determinant = t00 * a + t01 * b + t02 * c;
  const double one_over_det = 1.0 / result.determinant;
  get<0, 0>(result.inverse) = (t11 * t22 - t12 * t12) * one_over_det;
  get<0, 1>(result.inverse) = (t12 * t02 - t22 * t01) * one_over_det;
  get<0, 2>(result.inverse) = (t01 * t12 - t02 * t11) * one_over_det;
  get<1, 1>(result.inverse) = (t22 * t00 - t02 * t02) * one_over_det;
  get<1, 2>(result.inverse) = (t02 * t01 - t00 * t12) * one_over_det;
  get<2, 2>(result.inverse) = (t00 * t11 - t01 * t01) * one_over_det;
  return result;
}

/// The shift, as `gr::shift`
KOKKOS_FORCEINLINE_FUNCTION tnsr::I<double, 3> shift(
    const tnsr::aa<double, 3>& spacetime_metric,
    const tnsr::II<double, 3>& inverse_spatial_metric) {
  tnsr::I<double, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) =
        inverse_spatial_metric.get(i, 0) * get<1, 0>(spacetime_metric);
    for (size_t j = 1; j < 3; ++j) {
      result.get(i) +=
          inverse_spatial_metric.get(i, j) * spacetime_metric.get(j + 1, 0);
    }
  }
  return result;
}

/// The lapse, as `gr::lapse`
KOKKOS_FORCEINLINE_FUNCTION double lapse(
    const tnsr::aa<double, 3>& spacetime_metric,
    const tnsr::I<double, 3>& shift) {
  double result = -get<0, 0>(spacetime_metric);
  for (size_t i = 0; i < 3; ++i) {
    result += shift.get(i) * spacetime_metric.get(i + 1, 0);
  }
  return sqrt(result);
}

/// The inverse spacetime metric, as `gr::inverse_spacetime_metric`
KOKKOS_FORCEINLINE_FUNCTION tnsr::AA<double, 3> inverse_spacetime_metric(
    const double lapse, const tnsr::I<double, 3>& shift,
    const tnsr::II<double, 3>& inverse_spatial_metric) {
  tnsr::AA<double, 3> result{};
  get<0, 0>(result) = -1.0 / (lapse * lapse);
  const double minus_one_over_lapse_sqrd = get<0, 0>(result);
  for (size_t i = 0; i < 3; ++i) {
    result.get(0, i + 1) = -shift.get(i) * minus_one_over_lapse_sqrd;
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.get(i + 1, j + 1) =
          inverse_spatial_metric.get(i, j) +
          shift.get(i) * shift.get(j) * minus_one_over_lapse_sqrd;
    }
  }
  return result;
}

/// The spacetime normal vector, as `gr::spacetime_normal_vector`
KOKKOS_FORCEINLINE_FUNCTION tnsr::A<double, 3> spacetime_normal_vector(
    const double lapse, const tnsr::I<double, 3>& shift) {
  tnsr::A<double, 3> result{};
  get<0>(result) = 1. / lapse;
  for (size_t i = 0; i < 3; ++i) {
    result.get(i + 1) = -shift.get(i) * get<0>(result);
  }
  return result;
}
}  // namespace gr::device
#endif  // SPECTRE_KOKKOS
