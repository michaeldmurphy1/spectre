// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDevice.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

SPECTRE_TEST_CASE(
    "Unit.GrMhd.ValenciaDivClean.ConservativeFromPrimitive.Kokkos",
    "[Unit][GrMhd]") {
  TestHelpers::check_pointwise_on_device<
      grmhd::ValenciaDivClean::ConservativeFromPrimitive>(
      &grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply<DataVector>,
      grmhd::ValenciaDivClean::ConservativeFromPrimitive::return_tags{},
      grmhd::ValenciaDivClean::ConservativeFromPrimitive::argument_tags{}, 5,
      0.0, 1.0);
}
