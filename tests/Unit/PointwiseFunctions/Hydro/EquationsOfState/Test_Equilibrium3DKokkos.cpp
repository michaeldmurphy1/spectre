// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/DeviceView.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.EquationsOfState.Equilibrium3D.Kokkos",
    "[Unit][EquationsOfState]") {
  TestHelpers::EquationsOfState::check_device_view(
      EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>{
          EquationsOfState::IdealFluid<true>{5.0 / 3.0, 0.1}},
      {{{0.1, 2.0}, {0.1, 2.0}, {0.05, 0.5}}});
  TestHelpers::EquationsOfState::check_device_view_bounds(
      EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>{
          EquationsOfState::IdealFluid<true>{2.5, 0.1}},
      {{{0.1, 2.0}, {0.1, 2.0}, {0.05, 0.5}}});
}
