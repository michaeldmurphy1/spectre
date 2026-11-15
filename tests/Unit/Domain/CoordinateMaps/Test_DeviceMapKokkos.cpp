// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/CoordinateMaps/TimeDependent/DeviceRotScaleTrans.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
using DeviceRotScaleTrans =
    domain::CoordinateMaps::TimeDependent::DeviceRotScaleTrans<3>;

// A rotation by `angle` about z, and its time derivative for angular velocity
// `omega`
DeviceRotScaleTrans rotation(const double angle, const double omega) {
  DeviceRotScaleTrans map{};
  map.has_rotation = true;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      map.rotation.get(i, j) = 0.0;
      map.rotation_derivative.get(i, j) = 0.0;
    }
  }
  map.rotation.get(0, 0) = cos(angle);
  map.rotation.get(0, 1) = -sin(angle);
  map.rotation.get(1, 0) = sin(angle);
  map.rotation.get(1, 1) = cos(angle);
  map.rotation.get(2, 2) = 1.0;
  map.rotation_derivative.get(0, 0) = -omega * sin(angle);
  map.rotation_derivative.get(0, 1) = -omega * cos(angle);
  map.rotation_derivative.get(1, 0) = omega * cos(angle);
  map.rotation_derivative.get(1, 1) = -omega * sin(angle);
  return map;
}

// Every kind of map evaluated in a kernel gives the same as on the host
void test_on_device() {
  domain::CoordinateMaps::DeviceMap<3> map{};
  DeviceRotScaleTrans transition = rotation(0.4, 0.2);
  transition.has_expansion = true;
  transition.has_translation = true;
  transition.expansion_a = 0.9;
  transition.expansion_b = 0.99;
  transition.expansion_a_derivative = -0.01;
  transition.expansion_b_derivative = -0.002;
  transition.translation = {{0.3, -0.2, 0.1}};
  transition.translation_derivative = {{-0.05, 0.02, 0.01}};
  transition.inner_radius = 2.0;
  transition.outer_radius = 20.0;
  transition.region = DeviceRotScaleTrans::Region::Transition;
  map.push_back(transition);
  map.push_back(rotation(-1.1, 0.5));
  for (const double radius : {3.0, 15.0}) {
    CAPTURE(radius);
    const tnsr::I<double, 3, Frame::NoFrame> point{
        {{0.6 * radius, -0.64 * radius, 0.48 * radius}}};
    TestHelpers::check_on_device(KOKKOS_LAMBDA() {
      const auto result = map(point);
      return TestHelpers::flatten(result.coords, result.inverse_jacobian,
                                  result.jacobian, result.frame_velocity);
    });
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.DeviceMap.Kokkos",
                  "[Domain][Unit]") {
  test_on_device();
}
