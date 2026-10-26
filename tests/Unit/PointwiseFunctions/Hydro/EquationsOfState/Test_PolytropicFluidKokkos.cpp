// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/DeviceView.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.EquationsOfState.PolytropicFluid.Kokkos",
    "[Unit][EquationsOfState]") {
  // Specific enthalpies must be larger than 1 for the relativistic EOS
  TestHelpers::EquationsOfState::check_device_view(
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}, {{{1.1, 2.0}}});
  TestHelpers::EquationsOfState::check_device_view(
      EquationsOfState::PolytropicFluid<false>{100.0, 2.0}, {{{0.1, 2.0}}});
  // An exponent above 2 gives finite upper bounds
  for (const double polytropic_exponent : {2.0, 2.5}) {
    TestHelpers::EquationsOfState::check_device_view_bounds(
        EquationsOfState::PolytropicFluid<true>{100.0, polytropic_exponent},
        {{{0.1, 2.0}}});
  }
  TestHelpers::EquationsOfState::check_device_view_bounds(
      EquationsOfState::PolytropicFluid<false>{100.0, 2.5}, {{{0.1, 2.0}}});
}
