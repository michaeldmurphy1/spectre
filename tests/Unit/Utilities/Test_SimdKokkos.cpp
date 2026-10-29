// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Simd/Simd.hpp"

namespace {
// Calls every scalar `simd` function in a kernel.
// These are only used without xsimd, which is required for GPU backends.
void test_on_device() {
#ifndef SPECTRE_USE_XSIMD
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    std::array<double, 25> r{};
    r[0] = simd::all(true) ? 2.0 : 1.0;
    r[1] = simd::any(false) ? 2.0 : 1.0;
    r[2] = simd::none(false) ? 2.0 : 1.0;
    r[3] = simd::clip(3.0, -1.0, 2.0);
    r[4] = simd::clip(-3.0, -1.0, 2.0);
    r[5] = simd::exp10(1.5);
    r[6] = static_cast<double>(simd::exp10(0.5f));
    r[7] = simd::sign(true);
    r[8] = simd::sign(-2.5);
    r[9] = simd::sign(0.0);
    r[10] = simd::select(false, 1.0, 2.0);
    const auto [sin_double, cos_double] = simd::sincos(0.3);
    r[11] = sin_double;
    r[12] = cos_double;
    const auto [sin_float, cos_float] = simd::sincos(0.3f);
    r[13] = static_cast<double>(sin_float);
    r[14] = static_cast<double>(cos_float);
    r[15] = simd::fma(2.0, 3.0, 4.0);
    r[16] = static_cast<double>(simd::fma(2, 3, 4));
    r[17] = simd::fms(2.0, 3.0, 4.0);
    r[18] = static_cast<double>(simd::fms(2, 3, 4));
    r[19] = simd::fnma(2.0, 3.0, 4.0);
    r[20] = static_cast<double>(simd::fnma(2, 3, 4));
    r[21] = simd::fnms(2.0, 3.0, 4.0);
    r[22] = static_cast<double>(simd::fnms(2, 3, 4));
    double memory = 5.0;
    simd::store(&memory, simd::load(&memory) + 1.0);
    r[23] = memory;
    simd::store_unaligned(&memory, simd::load_unaligned(&memory) * 2.0);
    r[24] = memory;
    return r;
  });
#endif  // SPECTRE_USE_XSIMD
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Simd.Kokkos", "[Unit][Utilities]") {
  test_on_device();
}
