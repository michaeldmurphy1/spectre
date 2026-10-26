// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace grmhd::ValenciaDivClean {
class PrimitiveFromConservativeOptions;
}  // namespace grmhd::ValenciaDivClean
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean {
/*!
 * \brief Computes the primitive variables from the conservative variables
 * like `PrimitiveFromConservative`, but in a single Kokkos kernel.
 *
 * The EOS is evaluated through its `device_view()` (see
 * `EquationsOfState::visit_device_view`). It is an error if the `eos` has no
 * `device_view()`.
 *
 * The `pressure` is used as the initial guess of the recovery schemes that use
 * one. Returns `false` if all recovery schemes failed at any grid point. Unlike
 * `PrimitiveFromConservative` it doesn't stop at the first grid point where
 * recovery fails, and the primitive variables at the failed grid points are
 * NaN. If `ErrorOnFailure` is `true`, a failure is an error that reports the
 * number of failed grid points and the variables at the first one, like the
 * error of `PrimitiveFromConservative`.
 *
 * The arguments are copied to the device and the results are copied back, so
 * all data is on the host. All copies and the kernel run on the execution space
 * instance `exec`, which is fenced before returning.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes, bool ErrorOnFailure,
          bool EnforcePhysicality = true>
bool primitive_from_conservative_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    gsl::not_null<Scalar<DataVector>*> electron_fraction,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
    gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const PrimitiveFromConservativeOptions&
        primitive_from_conservative_options);
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
