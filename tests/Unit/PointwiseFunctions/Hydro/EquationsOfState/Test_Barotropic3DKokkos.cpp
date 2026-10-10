// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/DeviceView.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.EquationsOfState.Barotropic3D.Kokkos",
    "[Unit][EquationsOfState]") {
  TestHelpers::EquationsOfState::check_device_view(
      EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>{
          EquationsOfState::PolytropicFluid<true>{100.0, 2.0}},
      {{{0.1, 2.0}, {0.1, 2.0}, {0.05, 0.5}}});
}
