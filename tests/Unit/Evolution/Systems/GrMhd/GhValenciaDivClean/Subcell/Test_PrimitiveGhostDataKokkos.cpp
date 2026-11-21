// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// The hydro ghost variables of each element of a batch are the same as those
// that `PrimitiveGhostVariables::apply` computes on the host, where they come
// first
void test(const gsl::not_null<std::mt19937*> generator) {
  using primitive_tags = hydro::grmhd_tags<DataVector>;
  using evolved_tags =
      grmhd::GhValenciaDivClean::subcell::evolved_tags_on_device;
  using ghost_tags =
      grmhd::GhValenciaDivClean::Tags::primitive_grmhd_reconstruction_tags;
  constexpr size_t number_of_points = 27;
  constexpr size_t number_of_elements = 3;
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<Variables<primitive_tags>> prims{};
  std::vector<Variables<evolved_tags>> evolved{};
  BatchedVariables<primitive_tags, Kokkos::HostSpace> host_prims{
      number_of_points, number_of_elements};
  BatchedVariables<evolved_tags, Kokkos::HostSpace> host_evolved{
      number_of_points, number_of_elements};
  for (size_t e = 0; e < number_of_elements; ++e) {
    prims.push_back(make_with_random_values<Variables<primitive_tags>>(
        generator, make_not_null(&dist), number_of_points));
    Variables<primitive_tags> prims_element{};
    set_data_ref_to_element(make_not_null(&prims_element), host_prims, e);
    prims_element = prims.back();
    evolved.push_back(make_with_random_values<Variables<evolved_tags>>(
        generator, make_not_null(&dist), number_of_points));
    Variables<evolved_tags> evolved_element{};
    set_data_ref_to_element(make_not_null(&evolved_element), host_evolved, e);
    evolved_element = evolved.back();
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<primitive_tags> device_prims{exec, number_of_points,
                                                number_of_elements};
  deep_copy(exec, make_not_null(&device_prims), host_prims);
  BatchedVariables<ghost_tags> ghost_variables{exec, number_of_points,
                                               number_of_elements};
  grmhd::GhValenciaDivClean::subcell::primitive_ghost_variables_on_device(
      exec, make_not_null(&ghost_variables), device_prims);
  BatchedVariables<ghost_tags, Kokkos::HostSpace> host_ghost_variables{
      number_of_points, number_of_elements};
  deep_copy(exec, make_not_null(&host_ghost_variables), ghost_variables);
  exec.fence();

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const DataVector expected =
        grmhd::GhValenciaDivClean::subcell::PrimitiveGhostVariables::apply(
            prims[e], get<gr::Tags::SpacetimeMetric<DataVector, 3>>(evolved[e]),
            get<gh::Tags::Phi<DataVector, 3>>(evolved[e]),
            get<gh::Tags::Pi<DataVector, 3>>(evolved[e]), 0);
    Variables<ghost_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_ghost_variables, e);
    REQUIRE(element.size() < expected.size());
    for (size_t i = 0; i < element.size(); ++i) {
      CAPTURE(i);
      // Computed in the same way as on the host
      CHECK(element.data()[i] == expected[i]);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.PrimitiveGhostData."
    "Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
}
