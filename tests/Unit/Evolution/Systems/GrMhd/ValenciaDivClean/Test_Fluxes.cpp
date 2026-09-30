// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDevice.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Fluxes", "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::ComputeFluxes::apply<DataVector>, "Fluxes",
      {"tilde_d_flux", "tilde_ye_flux", "tilde_tau_flux", "tilde_s_flux",
       "tilde_b_flux", "tilde_phi_flux"},
      {{{0.0, 1.0}}}, DataVector{5});

#ifdef SPECTRE_KOKKOS
  TestHelpers::check_pointwise_on_device<
      grmhd::ValenciaDivClean::ComputeFluxes>(
      &grmhd::ValenciaDivClean::ComputeFluxes::apply<DataVector>,
      grmhd::ValenciaDivClean::ComputeFluxes::return_tags{},
      grmhd::ValenciaDivClean::ComputeFluxes::argument_tags{}, 5, 0.1, 1.0);
#endif  // SPECTRE_KOKKOS
}
