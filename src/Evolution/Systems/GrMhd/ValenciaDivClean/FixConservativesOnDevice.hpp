// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
namespace grmhd::ValenciaDivClean {
class FixConservatives;
}  // namespace grmhd::ValenciaDivClean
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean {
/// @{
/*!
 * \brief Fix the conservative variables like `FixConservatives`, but in a
 * Kokkos kernel.
 *
 * Copies the variables to the device, applies `FixConservatives::AtGridPoint`
 * at every grid point, and copies the fixed variables back. Returns `true` if
 * any variables were fixed. If the root find for the Lorentz factor fails at
 * any grid point, this ERRORs with the variables at the first such grid point.
 *
 * The copies and the kernel are enqueued on the execution space instance
 * `exec` (the default instance if not given), and this function blocks until
 * they are complete.
 */
bool fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<Scalar<DataVector>*> tilde_d,
    gsl::not_null<Scalar<DataVector>*> tilde_ye,
    gsl::not_null<Scalar<DataVector>*> tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives);

bool fix_conservatives_on_device(
    gsl::not_null<Scalar<DataVector>*> tilde_d,
    gsl::not_null<Scalar<DataVector>*> tilde_ye,
    gsl::not_null<Scalar<DataVector>*> tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives);
/// @}
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
