// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/DeviceView.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.IdealFluid.Kokkos",
                  "[Unit][EquationsOfState]") {
  TestHelpers::EquationsOfState::check_device_view(
      EquationsOfState::IdealFluid<true>{5.0 / 3.0, 0.1},
      {{{0.1, 2.0}, {0.1, 2.0}}});
  TestHelpers::EquationsOfState::check_device_view(
      EquationsOfState::IdealFluid<false>{1.4, 0.1},
      {{{0.1, 2.0}, {0.1, 2.0}}});
  // An adiabatic index above 2 gives a finite upper bound of the specific
  // internal energy
  for (const double adiabatic_index : {5.0 / 3.0, 2.5}) {
    TestHelpers::EquationsOfState::check_device_view_bounds(
        EquationsOfState::IdealFluid<true>{adiabatic_index, 0.1},
        {{{0.1, 2.0}, {0.1, 2.0}}});
  }
  TestHelpers::EquationsOfState::check_device_view_bounds(
      EquationsOfState::IdealFluid<false>{1.4, 0.1},
      {{{0.1, 2.0}, {0.1, 2.0}}});
}
