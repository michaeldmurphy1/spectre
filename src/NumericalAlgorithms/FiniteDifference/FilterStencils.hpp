// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace fd::detail {
/*!
 * \brief Kreiss-Oliger dissipation stencils of order `Order` on a uniform
 * grid (see `fd::kreiss_oliger_filter`).
 *
 * `pointwise(q, stride, epsilon)` computes the dissipation at `q[0]` from the
 * neighbors `q[i * stride]`. If `UnitStride` is `true` then `stride` must be
 * 1, which lets the compiler use constant offsets. These functions are shared
 * by the host and device implementations of the filter.
 */
template <size_t Order, bool UnitStride>
struct KoDissipationStencil;

template <bool UnitStride>
struct KoDissipationStencil<2, UnitStride> {
  static constexpr size_t fd_order = 2;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride, const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return epsilon *
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             ((q[-1] + q[1]) - 2.0 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * ((q[-stride] + q[stride]) - 2.0 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationStencil<4, UnitStride> {
  static constexpr size_t fd_order = 4;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride, const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return -epsilon *
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             (0.0625 * (q[-2] + q[2]) - 0.25 * (q[-1] + q[1]) + 0.375 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return -epsilon * (0.0625 * (q[-2 * stride] + q[2 * stride]) -
                         0.25 * (q[-stride] + q[stride]) + 0.375 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationStencil<6, UnitStride> {
  static constexpr size_t fd_order = 6;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride, const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (0.015625 * (q[-3] + q[3]) - 0.09375 * (q[-2] + q[2]) +
                        0.234375 * (q[-1] + q[1]) - 0.3125 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (0.015625 * (q[-3 * stride] + q[3 * stride]) -
                        0.09375 * (q[-2 * stride] + q[2 * stride]) +
                        0.234375 * (q[-stride] + q[stride]) - 0.3125 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationStencil<8, UnitStride> {
  static constexpr size_t fd_order = 8;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride, const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (-0.00390625 * (q[-4] + q[4]) +
                        0.03125 * (q[-3] + q[3]) - 0.109375 * (q[-2] + q[2]) +
                        0.21875 * (q[-1] + q[1]) - 0.2734375 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (-0.00390625 * (q[-4 * stride] + q[4 * stride]) +
                        0.03125 * (q[-3 * stride] + q[3 * stride]) -
                        0.109375 * (q[-2 * stride] + q[2 * stride]) +
                        0.21875 * (q[-stride] + q[stride]) - 0.2734375 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationStencil<10, UnitStride> {
  static constexpr size_t fd_order = 10;

  KOKKOS_FUNCTION SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride, const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon *
             (0.0009765625 * (q[-5] + q[5]) - 0.009765625 * (q[-4] + q[4]) +
              0.0439453125 * (q[-3] + q[3]) - 0.1171875 * (q[-2] + q[2]) +
              0.205078125 * (q[-1] + q[1]) - 0.24609375 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon *
             (0.0009765625 * (q[-5 * stride] + q[5 * stride]) -
              0.009765625 * (q[-4 * stride] + q[4 * stride]) +
              0.0439453125 * (q[-3 * stride] + q[3 * stride]) -
              0.1171875 * (q[-2 * stride] + q[2 * stride]) +
              0.205078125 * (q[-stride] + q[stride]) - 0.24609375 * q[0]);
    }
  }
};
}  // namespace fd::detail
