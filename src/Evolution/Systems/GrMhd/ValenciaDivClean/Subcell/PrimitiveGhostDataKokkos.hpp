// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Computes the ghost variables of `PrimitiveGhostVariables` (e.g.
 * \f$W v^i\f$ from the Lorentz factor and the spatial velocity) of a batch of
 * elements whose data is in device memory.
 *
 * The data of each element has the layout of the ghost variables that
 * `PrimitiveGhostVariables::apply` returns (without the RDMP data), and the
 * values are computed in the same way.
 *
 * The kernel is enqueued on `exec`, which is not fenced.
 */
void primitive_ghost_variables_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<
        BatchedVariables<PrimitiveGhostVariables::ghost_variables_tag_list>*>
        ghost_variables,
    const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives);
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
