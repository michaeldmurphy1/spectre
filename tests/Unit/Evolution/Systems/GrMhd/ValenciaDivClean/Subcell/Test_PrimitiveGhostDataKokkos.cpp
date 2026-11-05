// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// The ghost variables of each element of a batch are the same as those that
// `PrimitiveGhostVariables::apply` computes on the host
void test(const gsl::not_null<std::mt19937*> generator) {
  using primitive_tags = hydro::grmhd_tags<DataVector>;
  using ghost_tags = grmhd::ValenciaDivClean::subcell::PrimitiveGhostVariables::
      ghost_variables_tag_list;
  constexpr size_t number_of_points = 27;
  constexpr size_t number_of_elements = 3;
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<Variables<primitive_tags>> prims{};
  BatchedVariables<primitive_tags, Kokkos::HostSpace> host_prims{
      number_of_points, number_of_elements};
  for (size_t e = 0; e < number_of_elements; ++e) {
    prims.push_back(make_with_random_values<Variables<primitive_tags>>(
        generator, make_not_null(&dist), number_of_points));
    Variables<primitive_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_prims, e);
    element = prims.back();
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<primitive_tags> device_prims{exec, number_of_points,
                                                number_of_elements};
  deep_copy(exec, make_not_null(&device_prims), host_prims);
  BatchedVariables<ghost_tags> ghost_variables{exec, number_of_points,
                                               number_of_elements};
  grmhd::ValenciaDivClean::subcell::primitive_ghost_variables_on_device(
      exec, make_not_null(&ghost_variables), device_prims);
  BatchedVariables<ghost_tags, Kokkos::HostSpace> host_ghost_variables{
      number_of_points, number_of_elements};
  deep_copy(exec, make_not_null(&host_ghost_variables), ghost_variables);
  exec.fence();

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const DataVector expected =
        grmhd::ValenciaDivClean::subcell::PrimitiveGhostVariables::apply(
            prims[e], 0);
    Variables<ghost_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_ghost_variables, e);
    REQUIRE(element.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      CAPTURE(i);
      // Computed in the same way as on the host
      CHECK(element.data()[i] == expected[i]);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.PrimitiveGhostData.Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
}
