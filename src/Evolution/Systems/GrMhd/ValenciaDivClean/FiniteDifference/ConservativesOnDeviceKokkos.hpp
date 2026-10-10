// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief Computes the variables on one side of the faces of a batch of FD
 * grids in device memory from the reconstructed primitive variables, like
 * `compute_conservatives_for_reconstruction` does on the host with a 3D EOS,
 * in a single Kokkos kernel.
 *
 * At every grid point of the `faces` this computes, in order:
 * 1. the Lorentz factor from \f$W v^i\f$ and the spatial metric, and \f$v^i\f$,
 * 2. the spatial velocity one-form,
 * 3. the specific internal energy and pressure from the rest mass density,
 *    temperature and electron fraction with the `eos` (through its
 *    `device_view()`, see `EquationsOfState::visit_device_view`), and
 * 4. the conserved variables (see `ConservativeFromPrimitive`).
 *
 * The `faces` are updated in place. Nothing is copied between the host and
 * the device, and the kernel is only enqueued on `exec`.
 */
void compute_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<face_tags>*> faces,
    const EquationsOfState::EquationOfState<true, 3>& eos);
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
