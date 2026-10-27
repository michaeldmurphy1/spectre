// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "Utilities/Simd/Simd.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Simd", "[Unit][Utilities]") {
  const auto [sin_double, cos_double] = simd::sincos(0.3);
  CHECK(sin_double == approx(std::sin(0.3)));
  CHECK(cos_double == approx(std::cos(0.3)));
  const auto [sin_float, cos_float] = simd::sincos(0.3f);
  CHECK(sin_float == approx(std::sin(0.3f)));
  CHECK(cos_float == approx(std::cos(0.3f)));
}
