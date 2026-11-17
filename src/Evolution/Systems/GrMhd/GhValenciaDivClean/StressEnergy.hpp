// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace grmhd::GhValenciaDivClean {

/*!
 * \brief Add in the trace-reversed stress-energy source term to the \f$\Pi\f$
 * evolved variable of the Generalized Harmonic system.
 *
 * \details The only stress energy source term in the Generalized Harmonic
 * evolution equations is in the equation for \f$\Pi_{a b}\f$:
 *
 * \f[
 * \partial_t \Pi_{ab} + \text{(spatial derivative terms)} =
 * \text{(GH source terms)}
 * - 16 \pi \alpha (T_{ab} - \frac{1}{2} g_{a b} T^c{}_c)
 * \f]
 *
 * (note that this function takes as argument the trace-reversed stress energy
 * tensor)
 *
 * This function adds that contribution to the existing value of `dt_pi`. The
 * spacetime terms in the GH equation should be computed before passing the
 * `dt_pi` to this function for updating.
 *
 * \see `gh::TimeDerivative` for details about the spacetime
 * part of the time derivative calculation.
 */
template <typename DataType>
KOKKOS_FUNCTION void add_stress_energy_term_to_dt_pi(
    gsl::not_null<tnsr::aa<DataType, 3>*> dt_pi,
    const tnsr::aa<DataType, 3>& trace_reversed_stress_energy,
    const Scalar<DataType>& lapse);

/*!
 * \brief Calculate the trace-reversed stress-energy tensor \f$(T_{\mu \nu} -
 * 1/2 g_{\mu \nu} g^{\lambda \sigma} T_{\lambda \sigma}) \f$ associated with
 * the matter part of the GRMHD system.
 *
 * \details The stress energy tensor is needed to compute the backreaction of
 * the matter to the spacetime degrees of freedom. The stress energy is
 * calculated as described in \cite Moesta2013dna :
 *
 * \f[
 * T_{\mu \nu} = \rho h^* u_\mu u_\nu + p^* g_{\mu \nu} - b_\mu b_\nu,
 * \f]
 *
 * where \f$u_\mu\f$ is the four-velocity, \f$\rho\f$ is the rest mass density,
 * \f$h^*\f$ is the magnetically modified enthalpy, \f$p^*\f$ is the
 * magnetically modified pressure, and \f$b_{\mu}\f$ is the comoving magnetic
 * field (note that we deviate from the notation of \cite Moesta2013dna by
 * denoting the pressure with a lower-case \f$p\f$ instead of an upper-case
 * \f$P\f$).
 *
 * The spatial components of the four velocity \f$u_\mu\f$ are
 *
 * \f[
 * u_i = W v_i,
 * \f]
 *
 * and the time component is
 *
 * \f[
 * u_0 = - \alpha W + \beta^i u_i.
 * \f].
 *
 * The magnetically modified enthalpy is
 *
 * \f[
 * h^* = 1 + \epsilon + (p + b^2) / \rho.
 * \f]
 *
 * The magnetically modified pressure is
 *
 * \f[
 * p^* = p + b^2 / 2.
 * \f]
 *
 *
 * The comoving magnetic field is computed via
 *
 * \f{align}{
 * b_i &= B_i / W + v_i W v^k B_k\\
 * b_0 &= - \alpha W v^i B_i + \beta^i b_i
 * \f}
 *
 * Therefore, the trace-reversed stress energy simplifies to
 *
 * \f[
 * (T_{\mu \nu} - \frac{1}{2} g_{\mu \nu} g^{\lambda \sigma} T_{\lambda \sigma})
 * = \rho h^* u_\mu u_\nu  + \left(\frac{1}{2} \rho h^* - p\right) g_{\mu \nu}
 *  - b_\mu b_\nu
 * \f]
 */
template <typename DataType>
KOKKOS_FUNCTION void trace_reversed_stress_energy(
    gsl::not_null<tnsr::aa<DataType, 3>*> stress_energy,
    gsl::not_null<tnsr::a<DataType, 3>*> four_velocity_one_form_buffer,
    gsl::not_null<tnsr::a<DataType, 3>*>
        comoving_magnetic_field_one_form_buffer,
    const Scalar<DataType>& rest_mass_density,
    const tnsr::i<DataType, 3, Frame::Inertial>& spatial_velocity_one_form,
    const tnsr::i<DataType, 3, Frame::Inertial>& magnetic_field_one_form,
    const Scalar<DataType>& magnetic_field_squared,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& one_over_w_squared,
    const Scalar<DataType>& pressure,
    const Scalar<DataType>& specific_internal_energy,
    const tnsr::aa<DataType, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const Scalar<DataType>& lapse);
}  // namespace grmhd::GhValenciaDivClean
