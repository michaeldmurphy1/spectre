// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/VariableFixing/LimitLorentzFactor.hpp"
#include "Evolution/VariableFixing/LimitLorentzFactorOnDeviceKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
using primitive_tags = hydro::grmhd_tags<DataVector>;

// The same as `LimitLorentzFactor` on each element, bitwise, with points
// above and below the density cutoff and the Lorentz factor cap
void test(const gsl::not_null<std::mt19937*> generator, const bool enable) {
  CAPTURE(enable);
  const VariableFixing::LimitLorentzFactor limit{1.0e-6, 5.0, enable};
  constexpr size_t number_of_points = 50;
  constexpr size_t number_of_elements = 3;
  std::uniform_real_distribution<> density(0.0, 2.0e-6);
  std::uniform_real_distribution<> lorentz_factor(1.0, 10.0);
  std::uniform_real_distribution<> velocity(-0.5, 0.5);
  std::vector<Variables<primitive_tags>> elements{};
  BatchedVariables<primitive_tags, Kokkos::HostSpace> host_batch{
      number_of_points, number_of_elements};
  for (size_t e = 0; e < number_of_elements; ++e) {
    Variables<primitive_tags> element{number_of_points};
    fill_with_random_values(make_not_null(&element), generator,
                            make_not_null(&velocity));
    get(get<hydro::Tags::RestMassDensity<DataVector>>(element)) =
        make_with_random_values<DataVector>(generator, density,
                                            number_of_points);
    get(get<hydro::Tags::LorentzFactor<DataVector>>(element)) =
        make_with_random_values<DataVector>(generator, lorentz_factor,
                                            number_of_points);
    Variables<primitive_tags> batch_element{};
    set_data_ref_to_element(make_not_null(&batch_element), host_batch, e);
    batch_element = element;
    elements.push_back(std::move(element));
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<primitive_tags> device_batch{number_of_points,
                                                number_of_elements};
  deep_copy(exec, make_not_null(&device_batch), host_batch);
  VariableFixing::limit_lorentz_factor_on_device(
      exec, make_not_null(&device_batch), limit);
  deep_copy(exec, make_not_null(&host_batch), device_batch);
  exec.fence();

  size_t limited_points = 0;
  for (size_t e = 0; e < number_of_elements; ++e) {
    Variables<primitive_tags> expected = elements[e];
    limit(make_not_null(&get<hydro::Tags::LorentzFactor<DataVector>>(expected)),
          make_not_null(
              &get<hydro::Tags::SpatialVelocity<DataVector, 3>>(expected)),
          get<hydro::Tags::RestMassDensity<DataVector>>(expected));
    Variables<primitive_tags> result{};
    set_data_ref_to_element(make_not_null(&result), host_batch, e);
    CHECK(result == expected);
    for (size_t i = 0; i < number_of_points; ++i) {
      if (get(get<hydro::Tags::LorentzFactor<DataVector>>(expected))[i] !=
          get(get<hydro::Tags::LorentzFactor<DataVector>>(elements[e]))[i]) {
        ++limited_points;
      }
    }
  }
  // Some points are limited if and only if the limiting is enabled
  CHECK((limited_points > 0) == enable);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.LimitLorentzFactor.Kokkos",
                  "[Unit][Evolution][VariableFixing]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator), true);
  test(make_not_null(&generator), false);
}
