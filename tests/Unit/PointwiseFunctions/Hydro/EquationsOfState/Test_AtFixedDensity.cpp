// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/AtFixedDensity.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"

namespace {
// The functions of the device view of `equation_of_state` at random rest
// mass densities and temperatures in the `ranges` agree with those of the
// view at that density
template <typename EquationOfState>
void check(const EquationOfState& equation_of_state,
           const std::array<std::pair<double, double>, 2>& ranges) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> log_density_distribution{
      log(ranges[0].first), log(ranges[0].second)};
  std::uniform_real_distribution<double> temperature_distribution{
      ranges[1].first, ranges[1].second};
  const auto view = equation_of_state.device_view();
  const Scalar<double> electron_fraction{0.1};
  for (size_t i = 0; i < 1000; ++i) {
    const Scalar<double> density{exp(log_density_distribution(generator))};
    const Scalar<double> temperature{temperature_distribution(generator)};
    CAPTURE(density);
    CAPTURE(temperature);
    const auto& at_density =
        EquationsOfState::at_fixed_density(view, get(density));
    const auto specific_internal_energy =
        view.specific_internal_energy_from_density_and_temperature(
            density, temperature, electron_fraction);
    CHECK(get(at_density.specific_internal_energy_from_density_and_temperature(
              density, temperature, electron_fraction)) ==
          approx(get(specific_internal_energy)));
    CHECK(get(at_density.pressure_from_density_and_temperature(
              density, temperature, electron_fraction)) ==
          approx(get(view.pressure_from_density_and_temperature(
              density, temperature, electron_fraction))));
    CHECK(get(at_density.pressure_from_density_and_energy(
              density, specific_internal_energy, electron_fraction)) ==
          approx(get(view.pressure_from_density_and_energy(
              density, specific_internal_energy, electron_fraction))));
    CHECK(get(at_density.temperature_from_density_and_energy(
              density, specific_internal_energy, electron_fraction)) ==
          approx(get(temperature)));
    CHECK(get(at_density.sound_speed_squared_from_density_and_temperature(
              density, temperature, electron_fraction)) ==
          approx(get(view.sound_speed_squared_from_density_and_temperature(
              density, temperature, electron_fraction))));
    CHECK(get(at_density.kappa_from_density_and_temperature(
              density, temperature, electron_fraction)) ==
          approx(get(view.kappa_from_density_and_temperature(
              density, temperature, electron_fraction))));
  }
}

void test_hybrid() {
  namespace EoS = EquationsOfState;
  // The cold EOS of the BNS runs, with densities below its reference density,
  // between it and its upper density, and above
  const EoS::Spectral spectral{
      1.0118e-4, 3.3625e-7, {2.0, 0.0, 0.4029, -0.1008}, 5.0e-3};
  const EoS::Equilibrium3D<EoS::HybridEos<EoS::Spectral>> hybrid_spectral{
      EoS::HybridEos<EoS::Spectral>{spectral, 1.75, 0.0}};
  check(hybrid_spectral, {{{1.0e-10, 1.0e-2}, {0.0, 0.5}}});
  using AtDensity = std::remove_cvref_t<decltype(EoS::at_fixed_density(
      hybrid_spectral.device_view(), 1.0))>;
  static_assert(
      not std::is_same_v<AtDensity, decltype(hybrid_spectral.device_view())>);

  const EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>>
      hybrid_polytrope{EoS::HybridEos<EoS::PolytropicFluid<true>>{
          EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5, 1.0e-3}};
  check(hybrid_polytrope, {{{0.1, 2.0}, {0.1, 2.0}}});

#ifdef SPECTRE_DEBUG
  const auto view = hybrid_spectral.device_view();
  const auto at_density = EoS::at_fixed_density(view, 1.0e-3);
  CHECK_THROWS_WITH(
      at_density.pressure_from_density_and_temperature(
          Scalar<double>{2.0e-3}, Scalar<double>{0.1}, Scalar<double>{0.1}),
      Catch::Matchers::ContainsSubstring(
          "The cold EOS was evaluated at the rest mass density"));
#endif  // SPECTRE_DEBUG
}

// Other equations of state are returned as they are
void test_other() {
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
      ideal_fluid{EquationsOfState::IdealFluid<true>{5.0 / 3.0}};
  const auto view = ideal_fluid.device_view();
  const auto& at_density = EquationsOfState::at_fixed_density(view, 1.0);
  CHECK(&at_density == &view);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.AtFixedDensity",
                  "[Unit][EquationsOfState]") {
  test_hybrid();
  test_other();
}
