// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"

namespace {
template <typename DataType>
void test(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &grmhd::GhValenciaDivClean::trace_reversed_stress_energy<DataType>,
      "StressEnergy",
      {"trace_reversed_stress_energy", "four_velocity_one_form",
       "comoving_magnetic_field_one_form"},
      {{{1.0e-2, 0.5}}}, used_for_size);

  pypp::check_with_random_values<1>(
      &grmhd::GhValenciaDivClean::add_stress_energy_term_to_dt_pi<DataType>,
      "StressEnergy", {"add_stress_energy_term_to_dt_pi"}, {{{1.0e-2, 0.5}}},
      used_for_size, 1.0e-12, std::random_device{}(), 0.1234);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.GhValenciaDivClean.StressEnergy",
                  "[Unit][GrMhd]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/GhValenciaDivClean"};

  test(DataVector{5});
  test(0.0);
}
