// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/RdmpTciDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags = grmhd::ValenciaDivClean::Tags;
using conserved_tags =
    tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
               Tags::TildeS<Frame::Inertial>, Tags::TildeB<Frame::Inertial>,
               Tags::TildePhi>;

// The RDMP data of each element of a batch is the same as `TciOnFdGrid`
// computes on the host
void test(const gsl::not_null<std::mt19937*> generator) {
  constexpr size_t number_of_points = 343;
  constexpr size_t number_of_elements = 5;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Variables<conserved_tags>> conserved{};
  BatchedVariables<conserved_tags, Kokkos::HostSpace> host_conserved{
      number_of_points, number_of_elements};
  for (size_t e = 0; e < number_of_elements; ++e) {
    conserved.push_back(make_with_random_values<Variables<conserved_tags>>(
        generator, make_not_null(&dist), number_of_points));
    Variables<conserved_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_conserved, e);
    element = conserved.back();
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<conserved_tags> device_conserved{exec, number_of_points,
                                                    number_of_elements};
  deep_copy(exec, make_not_null(&device_conserved), host_conserved);
  const auto rdmp_tci_data =
      grmhd::ValenciaDivClean::subcell::rdmp_tci_data_on_device(
          exec, get<::Tags::BatchView<Tags::TildeD>>(device_conserved),
          get<::Tags::BatchView<Tags::TildeYe>>(device_conserved),
          get<::Tags::BatchView<Tags::TildeTau>>(device_conserved),
          get<::Tags::BatchView<Tags::TildeB<Frame::Inertial>>>(
              device_conserved));
  REQUIRE(rdmp_tci_data.size() == number_of_elements);

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    const auto& vars = conserved[e];
    const DataVector magnitude_tilde_b =
        get(magnitude(get<Tags::TildeB<Frame::Inertial>>(vars)));
    // As in `TciOnFdGrid::apply`
    const evolution::dg::subcell::RdmpTciData expected{
        .max_variables_values = DataVector{max(get(get<Tags::TildeD>(vars))),
                                           max(get(get<Tags::TildeYe>(vars))),
                                           max(get(get<Tags::TildeTau>(vars))),
                                           max(magnitude_tilde_b)},
        .min_variables_values = DataVector{min(get(get<Tags::TildeD>(vars))),
                                           min(get(get<Tags::TildeYe>(vars))),
                                           min(get(get<Tags::TildeTau>(vars))),
                                           min(magnitude_tilde_b)}};
    CHECK_ITERABLE_APPROX(rdmp_tci_data[e].max_variables_values,
                          expected.max_variables_values);
    CHECK_ITERABLE_APPROX(rdmp_tci_data[e].min_variables_values,
                          expected.min_variables_values);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.RdmpTciData.Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
}
