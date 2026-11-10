// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/DeviceView.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Spectral.Kokkos",
                  "[Unit][EquationsOfState]") {
  // The arguments are both densities and specific enthalpies. As densities
  // they are below the reference density (1.5), between it and the upper
  // density (10) and above, and as specific enthalpies they are in all three
  // branches of `rest_mass_density_from_enthalpy`, including the root find.
  const EquationsOfState::Spectral eos{1.5, 0.1, {2.0, 0.0, 0.4, -0.1}, 10.0};
  TestHelpers::EquationsOfState::check_device_view(eos, {{{1.01, 20.0}}});
  TestHelpers::EquationsOfState::check_device_view_bounds(eos,
                                                          {{{1.01, 20.0}}});

  // Too many tabulated values for the `DeviceView`
  CHECK_THROWS_WITH(
      (EquationsOfState::Spectral{1.0, 0.1, {2.0, 0.0, 0.4, -0.1}, 1.0e30}
           .device_view()),
      Catch::Matchers::ContainsSubstring(
          "The DeviceView of the Spectral EOS holds at most"));
}
