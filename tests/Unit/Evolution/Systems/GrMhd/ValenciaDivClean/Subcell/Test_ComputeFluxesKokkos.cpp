// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxesKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.ComputeFluxes.Kokkos",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const size_t num_pts = 5;
  // A batch of elements, with a tag that isn't used by `ComputeFluxes` and
  // must be left unchanged
  using batch_tags =
      tmpl::append<ComputeFluxes::return_tags, ComputeFluxes::argument_tags,
                   tmpl::list<hydro::Tags::Temperature<DataVector>>>;
  const size_t num_elements = 3;
  BatchedVariables<batch_tags, Kokkos::HostSpace> host_batch{num_pts,
                                                             num_elements};
  std::array<Variables<batch_tags>, num_elements> expected{};
  for (size_t e = 0; e < num_elements; ++e) {
    Variables<batch_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    fill_with_random_values(make_not_null(&element), make_not_null(&gen),
                            make_not_null(&dist));
    gsl::at(expected, e) = element;
    subcell::compute_fluxes(make_not_null(&gsl::at(expected, e)));
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<batch_tags> batch{exec, num_pts, num_elements};
  deep_copy(exec, make_not_null(&batch), host_batch);
  subcell::compute_fluxes(exec, make_not_null(&batch));
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();
  for (size_t e = 0; e < num_elements; ++e) {
    CAPTURE(e);
    Variables<batch_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    CHECK_VARIABLES_APPROX(element, gsl::at(expected, e));
  }
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
