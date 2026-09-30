// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#endif  // SPECTRE_KOKKOS

namespace grmhd::ValenciaDivClean::subcell {
namespace detail {
template <typename TagsList, typename... ReturnTags, typename... ArgumentTags>
void compute_fluxes_impl(const gsl::not_null<Variables<TagsList>*> vars,
                         tmpl::list<ReturnTags...> return_tags,
                         tmpl::list<ArgumentTags...> argument_tags) {
#ifdef SPECTRE_KOKKOS
  apply_pointwise_on_device<grmhd::ValenciaDivClean::ComputeFluxes>(
      vars, return_tags, argument_tags, get<ArgumentTags>(*vars)...);
#else
  (void)return_tags;
  (void)argument_tags;
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&get<ReturnTags>(*vars))..., get<ArgumentTags>(*vars)...);
#endif  // SPECTRE_KOKKOS
}
}  // namespace detail

/*!
 * \brief Helper function that calls `ComputeFluxes` by retrieving the return
 * and argument tags from `vars`.
 *
 * With Kokkos, the fluxes are computed pointwise in a Kokkos kernel.
 */
template <typename TagsList>
void compute_fluxes(const gsl::not_null<Variables<TagsList>*> vars) {
  detail::compute_fluxes_impl(
      vars, typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags{},
      typename grmhd::ValenciaDivClean::ComputeFluxes::argument_tags{});
}
}  // namespace grmhd::ValenciaDivClean::subcell
