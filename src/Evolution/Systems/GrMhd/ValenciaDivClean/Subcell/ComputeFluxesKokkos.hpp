// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Calls `ComputeFluxes` at every grid point of every element of the
 * batch `vars` in device memory in a single Kokkos kernel, e.g. on a batch of
 * face variables (see `fd::face_tags`).
 *
 * The fluxes are written to `vars`, which must also hold the arguments of
 * `ComputeFluxes`. Nothing is copied between the host and the device, and the
 * kernel is only enqueued on `exec`.
 */
template <typename TagsList>
void compute_fluxes(const Kokkos::DefaultExecutionSpace& exec,
                    const gsl::not_null<BatchedVariables<TagsList>*> vars) {
  apply_pointwise_on_device<ComputeFluxes>(
      exec, vars, typename ComputeFluxes::return_tags{},
      typename ComputeFluxes::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, get<::Tags::BatchView<ArgumentTags>>(*vars)...);
      }(typename ComputeFluxes::argument_tags{}));
}
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
