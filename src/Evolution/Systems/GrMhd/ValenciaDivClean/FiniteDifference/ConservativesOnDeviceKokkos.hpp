// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
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
 * 2. if `fix_to_atmosphere` is given, the fix to the atmosphere,
 * 3. the spatial velocity one-form,
 * 4. the specific internal energy and pressure from the rest mass density,
 *    temperature and electron fraction with the `eos` (through its
 *    `device_view()`, see `EquationsOfState::visit_device_view`), and
 * 5. the conserved variables (see `ConservativeFromPrimitive`).
 *
 * A failed root find of the kappa limiting of the fix to the atmosphere is
 * never an error. The variables at such grid points are left as they were
 * before the fix, and this returns the number of failed grid points and the
 * first one (see `PointwiseStatusSummary`). Report a failure with
 * `VariableFixing::error_fixing_to_atmosphere_on_device`.
 *
 * The `faces` are updated in place. Nothing is copied between the host and
 * the device, but since the failures are needed on the host, this function
 * waits for the kernel on `exec` to complete.
 *
 * `FaceTags` is `face_tags` or `device_face_tags`.
 */
template <typename FaceTags>
PointwiseStatusSummary compute_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<FaceTags>*> faces,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* fix_to_atmosphere);
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
