// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDevice.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.ConservativeFromPrimitive",
                  "[Unit][GrMhd]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply<DataVector>,
      "ConservativeFromPrimitive",
      {"tilde_d", "tilde_ye", "tilde_tau", "tilde_s", "tilde_b", "tilde_phi"},
      {{{0.0, 1.0}}}, DataVector{5});

#ifdef SPECTRE_KOKKOS
  TestHelpers::check_pointwise_on_device<
      grmhd::ValenciaDivClean::ConservativeFromPrimitive>(
      &grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply<DataVector>,
      grmhd::ValenciaDivClean::ConservativeFromPrimitive::return_tags{},
      grmhd::ValenciaDivClean::ConservativeFromPrimitive::argument_tags{}, 5,
      0.0, 1.0);
#endif  // SPECTRE_KOKKOS
}
