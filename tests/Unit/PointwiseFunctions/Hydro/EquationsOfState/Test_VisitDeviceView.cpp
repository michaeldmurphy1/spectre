// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PiecewisePolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"

namespace {
namespace EoS = EquationsOfState;

static_assert(EoS::has_device_view_v<EoS::IdealFluid<true>>);
static_assert(EoS::has_device_view_v<EoS::PolytropicFluid<false>>);
static_assert(
    EoS::has_device_view_v<EoS::Equilibrium3D<EoS::IdealFluid<true>>>);
static_assert(
    EoS::has_device_view_v<EoS::Barotropic3D<EoS::PolytropicFluid<true>>>);
static_assert(not EoS::has_device_view_v<EoS::PiecewisePolytropicFluid<true>>);
// The 3D wrappers only have a `device_view()` if the wrapped EOS has one
static_assert(not EoS::has_device_view_v<
              EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>>>);

// Checks that `visit_device_view` calls the function with the `DeviceView` of
// the concrete type of `eos`, and compares it with `check`
template <typename ExpectedEosType, bool IsRelativistic,
          size_t ThermodynamicDim, typename F>
void check_visited(
    const EoS::EquationOfState<IsRelativistic, ThermodynamicDim>& eos,
    const F& check) {
  using ExpectedDeviceView =
      decltype(std::declval<const ExpectedEosType&>().device_view());
  size_t number_of_calls = 0;
  EoS::visit_device_view(eos, [&](const auto& device_view) {
    ++number_of_calls;
    using DeviceView = std::decay_t<decltype(device_view)>;
    CHECK(std::is_same_v<DeviceView, ExpectedDeviceView>);
    // `visit_device_view` instantiates this for all EOSs with a `DeviceView`
    if constexpr (std::is_same_v<DeviceView, ExpectedDeviceView>) {
      check(device_view);
    }
  });
  CHECK(number_of_calls == 1);
}

// Checks that `visit_device_view` is an error for EOSs without a `DeviceView`
template <bool IsRelativistic, size_t ThermodynamicDim>
void check_error(
    const EoS::EquationOfState<IsRelativistic, ThermodynamicDim>& eos,
    const std::string& eos_name) {
  bool called = false;
  CHECK_THROWS_WITH(
      EoS::visit_device_view(
          eos, [&called](const auto& /*device_view*/) { called = true; }),
      Catch::Matchers::ContainsSubstring(eos_name) and
          Catch::Matchers::ContainsSubstring(" has no DeviceView, so it can't "
                                             "be evaluated on the device"));
  CHECK_FALSE(called);
}

void test() {
  check_visited<EoS::PolytropicFluid<true>>(
      EoS::PolytropicFluid<true>{100.0, 2.0}, [](const auto& device_view) {
        CHECK(device_view.polytropic_constant == 100.0);
        CHECK(device_view.polytropic_exponent == 2.0);
      });
  check_visited<EoS::IdealFluid<false>>(
      EoS::IdealFluid<false>{1.4, 0.1}, [](const auto& device_view) {
        CHECK(device_view.adiabatic_index == 1.4);
        CHECK(device_view.min_temperature == 0.1);
      });
  check_visited<EoS::Equilibrium3D<EoS::IdealFluid<true>>>(
      EoS::Equilibrium3D<EoS::IdealFluid<true>>{EoS::IdealFluid<true>{1.4}},
      [](const auto& device_view) {
        CHECK(device_view.underlying.adiabatic_index == 1.4);
      });
  check_visited<EoS::Barotropic3D<EoS::PolytropicFluid<true>>>(
      EoS::Barotropic3D<EoS::PolytropicFluid<true>>{
          EoS::PolytropicFluid<true>{100.0, 2.0}},
      [](const auto& device_view) {
        CHECK(device_view.underlying.polytropic_constant == 100.0);
      });

  check_error(EoS::PiecewisePolytropicFluid<true>{1.0e-4, 100.0, 1.5, 2.0},
              "EquationsOfState::PiecewisePolytropicFluid<true>");
  check_error(
      EoS::HybridEos<EoS::PolytropicFluid<true>>{
          EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5},
      "EquationsOfState::HybridEos<");
  check_error(
      EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>>{
          EoS::HybridEos<EoS::PolytropicFluid<true>>{
              EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5}},
      "EquationsOfState::Equilibrium3D<EquationsOfState::HybridEos<");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.VisitDeviceView",
                  "[Unit][EquationsOfState]") {
  test();
}
