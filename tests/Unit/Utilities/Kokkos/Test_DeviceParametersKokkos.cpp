// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <type_traits>

#include "Utilities/Kokkos/DeviceParametersKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
struct Parameters {
  double a;
  int b;
};

// Reads the parameters in a kernel, as kernels that capture `device()` do
Parameters read_on_device(const Kokkos::DefaultExecutionSpace& exec,
                          const Kokkos::View<const Parameters*>& device) {
  const Kokkos::View<Parameters*> result{"result", 1};
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec, 0, 1),
      KOKKOS_LAMBDA(const int /*i*/) { result(0) = device(0); });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  return host(0);
}

void test_set() {
  const Kokkos::DefaultExecutionSpace exec{};
  DeviceParameters<Parameters> parameters{"Test"};
  parameters.set(exec, Parameters{1.5, 2});
  // The kernel sees the parameters, and the address doesn't change when they
  // are set again (after waiting for the device, as `set()` requires)
  const auto device = parameters.device();
  auto read = read_on_device(exec, device);
  CHECK(read.a == 1.5);
  CHECK(read.b == 2);
  exec.fence();
  parameters.set(exec, Parameters{-3.0, 4});
  CHECK(parameters.device().data() == device.data());
  read = read_on_device(exec, device);
  CHECK(read.a == -3.0);
  CHECK(read.b == 4);

  // Moving keeps the device buffer
  DeviceParameters<Parameters> moved{std::move(parameters)};
  CHECK(moved.device().data() == device.data());
  static_assert(not std::is_copy_constructible_v<DeviceParameters<Parameters>>);
  static_assert(not std::is_copy_assignable_v<DeviceParameters<Parameters>>);
}

void test_array() {
  const Kokkos::DefaultExecutionSpace exec{};
  DeviceParameters<std::array<double, 3>> parameters{};
  parameters.set(exec, std::array<double, 3>{{1.0, 2.0, 3.0}});
  const Kokkos::View<const double*> elements = parameters.device_array();
  CHECK(elements.extent(0) == 3);
  CHECK(static_cast<const void*>(elements.data()) ==
        static_cast<const void*>(parameters.device().data()));
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, elements);
  CHECK(host(0) == 1.0);
  CHECK(host(1) == 2.0);
  CHECK(host(2) == 3.0);
}
}  // namespace

// The debug check that the device copied the parameters before the next
// `set()` can only fail on asynchronous backends, so it isn't tested here.
SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.DeviceParameters",
                  "[Unit][Utilities]") {
  test_set();
  test_array();
}
