// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean {
/*!
 * \brief Computes the conserved variables from the primitive variables like
 * `ConservativeFromPrimitive`, but in a Kokkos kernel.
 *
 * The kernel runs on the instance of the calling thread
 * (`Parallel::device_execution_space()`). The primitive variables are copied to
 * the device and the conserved variables back, since the data is still stored
 * on the host.
 */
struct ConservativeFromPrimitiveOnDevice {
  using return_tags = ConservativeFromPrimitive::return_tags;
  using argument_tags = ConservativeFromPrimitive::argument_tags;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_ye,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      gsl::not_null<Scalar<DataVector>*> tilde_phi,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& specific_internal_energy,
      const Scalar<DataVector>& pressure,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const Scalar<DataVector>& divergence_cleaning_field);
};
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
