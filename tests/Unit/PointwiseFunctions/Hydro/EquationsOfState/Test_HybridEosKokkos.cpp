// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/DeviceView.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.HybridEos.Kokkos",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  // The cold EOS of the BNS runs, with densities across its reference and
  // upper densities. The minimum temperature is positive, so the entropy is
  // defined everywhere.
  const EoS::Spectral spectral{
      1.0118e-4, 3.3625e-7, {2.0, 0.0, 0.4029, -0.1008}, 5.0e-3};
  const EoS::HybridEos<EoS::Spectral> hybrid_spectral{spectral, 1.75, 1.0e-3};
  TestHelpers::EquationsOfState::check_device_view(
      hybrid_spectral, {{{1.0e-5, 1.0e-2}, {1.0e-3, 0.5}}});
  TestHelpers::EquationsOfState::check_device_view_bounds(
      hybrid_spectral, {{{1.0e-5, 1.0e-2}, {1.0e-3, 0.5}}});
  // As in the BNS runs
  const EoS::Equilibrium3D<EoS::HybridEos<EoS::Spectral>> equilibrium{
      hybrid_spectral};
  TestHelpers::EquationsOfState::check_device_view(
      equilibrium, {{{1.0e-5, 1.0e-2}, {1.0e-3, 0.5}, {0.05, 0.5}}});
  TestHelpers::EquationsOfState::check_device_view_bounds(
      equilibrium, {{{1.0e-5, 1.0e-2}, {1.0e-3, 0.5}, {0.05, 0.5}}});

  const EoS::HybridEos<EoS::PolytropicFluid<true>> hybrid_polytrope{
      EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5, 1.0e-3};
  TestHelpers::EquationsOfState::check_device_view(hybrid_polytrope,
                                                   {{{0.1, 2.0}, {0.1, 2.0}}});
  TestHelpers::EquationsOfState::check_device_view_bounds(
      hybrid_polytrope, {{{0.1, 2.0}, {0.1, 2.0}}});
}
