// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
namespace VariableFixing {
class LimitLorentzFactor;
}  // namespace VariableFixing
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace VariableFixing {
/*!
 * \brief Limits the Lorentz factor of a batch of elements whose data is in
 * device memory, like `LimitLorentzFactor` does for each element, in a single
 * Kokkos kernel on `exec`.
 *
 * The Lorentz factor and the spatial velocity of the `primitives` are changed
 * in place, with the same operations as `LimitLorentzFactor` (see
 * `LimitLorentzFactor::velocity_rescaling`). Nothing is done if the limiting
 * isn't enabled.
 */
void limit_lorentz_factor_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*> primitives,
    const LimitLorentzFactor& limit_lorentz_factor);
}  // namespace VariableFixing
#endif  // SPECTRE_KOKKOS
