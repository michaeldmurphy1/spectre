// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief Computes the hydro part of the ghost variables of
 * `PrimitiveGhostVariables` (the primitive variables to reconstruct, e.g.
 * \f$W v^i\f$) of a batch of elements whose data is in device memory.
 *
 * The values are computed as in `PrimitiveGhostVariables::apply`. The rest of
 * the ghost variables are the GH variables, which FD patches read from the
 * evolved variables (see `evolution::dg::subcell::fd_patch::GhostZones`).
 *
 * The kernel is enqueued on `exec`, which is not fenced.
 */
void primitive_ghost_variables_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<
        GhValenciaDivClean::Tags::primitive_grmhd_reconstruction_tags>*>
        ghost_variables,
    const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives);
}  // namespace grmhd::GhValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
