// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Utilities/ForceInline.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace fd::detail {
/*!
 * \brief Centered finite difference stencils for the first derivative of
 * order `Order` on a uniform grid.
 *
 * `pointwise(q, stride, weights)` computes the derivative at `q[0]` from the
 * neighbors `q[i * stride]`, where `weights` are computed once from the grid
 * spacing by `derivative_weights(1 / delta)`. If `UnitStride` is `true` then
 * `stride` must be 1, which lets the compiler use constant offsets. These
 * functions are shared by the host and device implementations of the finite
 * difference derivatives.
 */
template <size_t Order, bool UnitStride>
struct DerivativeStencil;

template <bool UnitStride>
struct DerivativeStencil<2, UnitStride> {
  static constexpr size_t fd_order = 2;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 1>& weights) {
    if constexpr (UnitStride) {
      SPECTRE_KOKKOS_ASSERT(stride == 1,
                            "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[0] * (q[1] - q[-1]);
    } else {
      const auto signed_stride = static_cast<ptrdiff_t>(stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[0] * (q[signed_stride] - q[-signed_stride]);
    }
  }

  KOKKOS_FUNCTION static constexpr std::array<double, 1> derivative_weights(
      const double one_over_delta) {
    return {{0.5 * one_over_delta}};
  }
};

template <bool UnitStride>
struct DerivativeStencil<4, UnitStride> {
  static constexpr size_t fd_order = 4;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 2>& weights) {
    if constexpr (UnitStride) {
      SPECTRE_KOKKOS_ASSERT(stride == 1,
                            "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[1] * (q[-2] - q[2]) + weights[0] * (q[1] - q[-1]);
    } else {
      const auto signed_stride = static_cast<ptrdiff_t>(stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[1] * (q[-2 * signed_stride] - q[2 * signed_stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[signed_stride] - q[-signed_stride]);
    }
  }

  KOKKOS_FUNCTION static constexpr std::array<double, 2> derivative_weights(
      const double one_over_delta) {
    return {{0.6666666666666666 * one_over_delta,
             0.08333333333333333 * one_over_delta}};
  }
};

template <bool UnitStride>
struct DerivativeStencil<6, UnitStride> {
  static constexpr size_t fd_order = 6;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 3>& weights) {
    if constexpr (UnitStride) {
      SPECTRE_KOKKOS_ASSERT(stride == 1,
                            "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[2] * (q[3] - q[-3]) - weights[1] * (q[2] - q[-2]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[1] - q[-1]);
    } else {
      const auto signed_stride = static_cast<ptrdiff_t>(stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[2] * (q[3 * signed_stride] - q[-3 * signed_stride]) -
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[1] * (q[2 * signed_stride] - q[-2 * signed_stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[signed_stride] - q[-signed_stride]);
    }
  }

  KOKKOS_FUNCTION static constexpr std::array<double, 3> derivative_weights(
      const double one_over_delta) {
    return {{0.75 * one_over_delta, 0.15 * one_over_delta,
             0.016666666666666666 * one_over_delta}};
  }
};

template <bool UnitStride>
struct DerivativeStencil<8, UnitStride> {
  static constexpr size_t fd_order = 8;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 4>& weights) {
    if constexpr (UnitStride) {
      SPECTRE_KOKKOS_ASSERT(stride == 1,
                            "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[3] * (q[4] - q[-4]) + weights[2] * (q[3] - q[-3]) -
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[1] * (q[2] - q[-2]) + weights[0] * (q[1] - q[-1]);
    } else {
      const auto signed_stride = static_cast<ptrdiff_t>(stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[3] * (q[4 * signed_stride] - q[-4 * signed_stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[2] * (q[3 * signed_stride] - q[-3 * signed_stride]) -
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[1] * (q[2 * signed_stride] - q[-2 * signed_stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[signed_stride] - q[-signed_stride]);
    }
  }

  KOKKOS_FUNCTION static constexpr std::array<double, 4> derivative_weights(
      const double one_over_delta) {
    return {{0.8 * one_over_delta, 0.2 * one_over_delta,
             0.0380952380952381 * one_over_delta,
             -0.0035714285714285713 * one_over_delta}};
  }
};

template <bool UnitStride>
struct DerivativeStencil<10, UnitStride> {
  static constexpr size_t fd_order = 10;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 5>& weights) {
    if constexpr (UnitStride) {
      SPECTRE_KOKKOS_ASSERT(stride == 1,
                            "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[4] * (q[5] - q[-5]) + weights[3] * (q[4] - q[-4]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[2] * (q[3] - q[-3]) - weights[1] * (q[2] - q[-2]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[1] - q[-1]);
    } else {
      const auto signed_stride = static_cast<ptrdiff_t>(stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[4] * (q[5 * signed_stride] - q[-5 * signed_stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[3] * (q[4 * signed_stride] - q[-4 * signed_stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[2] * (q[3 * signed_stride] - q[-3 * signed_stride]) -
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[1] * (q[2 * signed_stride] - q[-2 * signed_stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[signed_stride] - q[-signed_stride]);
    }
  }

  KOKKOS_FUNCTION static constexpr std::array<double, 5> derivative_weights(
      const double one_over_delta) {
    return {{0.8333333333333334 * one_over_delta,
             0.2380952380952381 * one_over_delta,
             0.05952380952380952 * one_over_delta,
             -0.009920634920634921 * one_over_delta,
             0.0007936507936507937 * one_over_delta}};
  }
};
}  // namespace fd::detail
