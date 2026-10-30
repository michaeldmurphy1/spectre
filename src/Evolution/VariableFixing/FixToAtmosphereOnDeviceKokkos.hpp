// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace VariableFixing {
/*!
 * \brief Fix the primitive variables to the atmosphere like
 * `FixToAtmosphere`, but in a Kokkos kernel.
 *
 * Copies the variables to the device, applies
 * `FixToAtmosphere::DeviceView::fix_at_point` at every grid point with the
 * `DeviceView` of the `equation_of_state`, and copies the fixed variables back.
 * It is an error if the equation of state has no `DeviceView`. If the root find
 * of the kappa limiting fails at any grid point, this ERRORs with the variables
 * at the first such grid point.
 *
 * The copies and the kernel are enqueued on the execution space instance
 * `exec` (the default instance if not given), and this function blocks until
 * they are complete.
 */
void fix_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
    gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>& fix_to_atmosphere);

void fix_to_atmosphere_on_device(
    gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
    gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>& fix_to_atmosphere);
}  // namespace VariableFixing
#endif  // SPECTRE_KOKKOS

namespace VariableFixing {
/*!
 * \brief Fix the primitive variables to the atmosphere with
 * `fix_to_atmosphere_on_device` in Kokkos builds and with `FixToAtmosphere`
 * otherwise.
 *
 * With Kokkos, the kernel runs on the default execution space instance and this
 * function blocks until the variables are copied back.
 */
void fix_to_atmosphere_on_host_or_device(
    gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
    gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>& fix_to_atmosphere);
}  // namespace VariableFixing
