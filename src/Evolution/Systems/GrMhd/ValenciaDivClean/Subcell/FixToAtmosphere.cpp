// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/FixToAtmosphere.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::subcell {
void FixToAtmosphere::apply(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const fixer& fix_to_atmosphere) {
  VariableFixing::fix_to_atmosphere_on_host_or_device(
      rest_mass_density, specific_internal_energy, spatial_velocity,
      lorentz_factor, pressure, temperature, electron_fraction, spatial_metric,
      equation_of_state, fix_to_atmosphere);
}
}  // namespace grmhd::ValenciaDivClean::subcell
