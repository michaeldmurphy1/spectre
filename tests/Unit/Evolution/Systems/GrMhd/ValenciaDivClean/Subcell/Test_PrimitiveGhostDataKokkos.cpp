// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// `apply_on_device` gives the same ghost variables as `apply`, and leaves
// the rest of the buffer unchanged
void test(const gsl::not_null<std::mt19937*> generator) {
  using Mutator =
      grmhd::ValenciaDivClean::subcell::PrimitiveGhostVariablesOnDevice;
  constexpr size_t number_of_points = 27;
  constexpr size_t additional_buffer = 5;
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  const auto prims =
      make_with_random_values<Variables<hydro::grmhd_tags<DataVector>>>(
          generator, make_not_null(&dist), number_of_points);
  auto box = db::create<
      db::AddSimpleTags<::Tags::Variables<hydro::grmhd_tags<DataVector>>>>(
      prims);
  const DataVector expected =
      db::mutate_apply<Mutator>(make_not_null(&box), additional_buffer);

  const Kokkos::DefaultExecutionSpace exec{};
  const Kokkos::View<double*> ghost_variables("ghost variables",
                                              expected.size());
  Kokkos::deep_copy(exec, ghost_variables, -1.0);
  Mutator::apply_on_device(exec, ghost_variables, prims);
  const auto host_ghost_variables =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ghost_variables);
  const size_t number_of_ghost_values = expected.size() - additional_buffer;
  for (size_t i = 0; i < number_of_ghost_values; ++i) {
    CAPTURE(i);
    // Computed in the same order as on the host
    CHECK(host_ghost_variables(i) == expected[i]);
  }
  for (size_t i = number_of_ghost_values; i < expected.size(); ++i) {
    CHECK(host_ghost_variables(i) == -1.0);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.PrimitiveGhostData.Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
}
