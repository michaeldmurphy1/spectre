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
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief Computes the specific internal energy and pressure from the rest mass
 * density, temperature and electron fraction with the `eos`, and then the
 * conserved variables (see `ConservativeFromPrimitive`), in a single Kokkos
 * kernel.
 *
 * The EOS is evaluated through its `device_view()` (see
 * `EquationsOfState::visit_device_view`). It is an error if the `eos` has no
 * `device_view()`.
 *
 * The arguments are copied to the device and the results are copied back, so
 * all data is on the host. All copies and the kernel run on the execution space
 * instance `exec`, which is fenced before returning.
 */
void compute_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> tilde_d,
    gsl::not_null<Scalar<DataVector>*> tilde_ye,
    gsl::not_null<Scalar<DataVector>*> tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Scalar<DataVector>& divergence_cleaning_field,
    const EquationsOfState::EquationOfState<true, 3>& eos);
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
