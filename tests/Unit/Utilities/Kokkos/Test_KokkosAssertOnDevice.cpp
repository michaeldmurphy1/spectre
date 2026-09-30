// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
KOKKOS_FUNCTION double checked_sqrt(const double x) {
  SPECTRE_KOKKOS_ASSERT(x >= 0.0, "Can't take the square root of " << x);
  return sqrt(x);
}
}  // namespace

// Only built with SPECTRE_DEBUG and a device backend, see CMakeLists.txt.
// [[OutputRegex, SPECTRE_KOKKOS_ASSERT failed on the device: x >= 0.0]]
SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.AssertOnDevice", "[Utilities][Unit]") {
  OUTPUT_TEST();
  Kokkos::parallel_for(
      "TestAssertOnDevice", 1, KOKKOS_LAMBDA(const int /*i*/) {
        static_cast<void>(checked_sqrt(-1.0));
      });
  Kokkos::fence();
}
