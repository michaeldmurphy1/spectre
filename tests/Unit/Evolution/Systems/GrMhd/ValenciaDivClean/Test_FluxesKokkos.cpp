// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDevice.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Fluxes.Kokkos",
                  "[Unit][GrMhd]") {
  TestHelpers::check_pointwise_on_device<
      grmhd::ValenciaDivClean::ComputeFluxes>(
      &grmhd::ValenciaDivClean::ComputeFluxes::apply<DataVector>,
      grmhd::ValenciaDivClean::ComputeFluxes::return_tags{},
      grmhd::ValenciaDivClean::ComputeFluxes::argument_tags{}, 5, 0.1, 1.0);
}
