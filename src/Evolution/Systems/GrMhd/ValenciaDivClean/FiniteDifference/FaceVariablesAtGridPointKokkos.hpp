// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd {
/// The `FixToAtmosphere::DeviceView` as an argument of a kernel
struct FixToAtmosphereView : db::SimpleTag {
  using type = VariableFixing::FixToAtmosphere<3>::DeviceView;
};

/*!
 * \brief The variables on a face at a grid point from the reconstructed
 * primitive variables, computed like `compute_conservatives_for_reconstruction`
 * (see `compute_conservatives_on_device` for the steps).
 *
 * \details The previous specific internal energy and pressure are only used
 * as the result if the root find of the fix to the atmosphere fails (the
 * status is then `failed`); otherwise they are recomputed.
 */
template <typename DeviceView, bool FixAtmosphere>
struct FaceVariablesAtGridPoint {
  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                 Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
                 Tags::TildeB<>, Tags::TildePhi>;
  using argument_tags = tmpl::append<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 EquationsOfState::Tags::DeviceView<DeviceView>>,
      tmpl::conditional_t<FixAtmosphere, tmpl::list<FixToAtmosphereView>,
                          tmpl::list<>>>;

  template <typename... FixToAtmosphere>
  KOKKOS_FUNCTION static PointwiseStatus apply(
      const gsl::not_null<Scalar<double>*> rest_mass_density,
      const gsl::not_null<Scalar<double>*> specific_internal_energy,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          spatial_velocity,
      const gsl::not_null<Scalar<double>*> lorentz_factor,
      const gsl::not_null<Scalar<double>*> pressure,
      const gsl::not_null<Scalar<double>*> temperature,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*>
          spatial_velocity_one_form,
      const gsl::not_null<Scalar<double>*> tilde_d,
      const gsl::not_null<Scalar<double>*> tilde_ye,
      const gsl::not_null<Scalar<double>*> tilde_tau,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*> tilde_s,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> tilde_b,
      const gsl::not_null<Scalar<double>*> tilde_phi,
      const Scalar<double>& reconstructed_rest_mass_density,
      const Scalar<double>& previous_specific_internal_energy,
      const Scalar<double>& previous_pressure,
      const Scalar<double>& reconstructed_temperature,
      const Scalar<double>& electron_fraction,
      const tnsr::I<double, 3, Frame::Inertial>&
          lorentz_factor_times_spatial_velocity,
      const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
      const Scalar<double>& divergence_cleaning_field,
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
      const Scalar<double>& sqrt_det_spatial_metric, const DeviceView& eos,
      const FixToAtmosphere&... fix_to_atmosphere) {
    // 1. The Lorentz factor and the spatial velocity
    get(*lorentz_factor) = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      get(*lorentz_factor) +=
          spatial_metric.get(i, i) *
          square(lorentz_factor_times_spatial_velocity.get(i));
      for (size_t j = i + 1; j < 3; ++j) {
        get(*lorentz_factor) += 2.0 * spatial_metric.get(i, j) *
                                lorentz_factor_times_spatial_velocity.get(i) *
                                lorentz_factor_times_spatial_velocity.get(j);
      }
    }
    get(*lorentz_factor) = sqrt(1.0 + get(*lorentz_factor));
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity->get(i) =
          lorentz_factor_times_spatial_velocity.get(i) / get(*lorentz_factor);
    }
    *rest_mass_density = reconstructed_rest_mass_density;
    *specific_internal_energy = previous_specific_internal_energy;
    *pressure = previous_pressure;
    *temperature = reconstructed_temperature;

    // 2. The fix to the atmosphere
    if constexpr (sizeof...(FixToAtmosphere) == 1) {
      const tnsr::I<double, 3, Frame::Inertial> unfixed_spatial_velocity =
          *spatial_velocity;
      const Scalar<double> unfixed_lorentz_factor = *lorentz_factor;
      bool root_find_failed = false;
      (fix_to_atmosphere.template fix_at_point<3>(
           make_not_null(&get(*rest_mass_density)),
           make_not_null(&get(*specific_internal_energy)), spatial_velocity,
           make_not_null(&get(*lorentz_factor)), make_not_null(&get(*pressure)),
           make_not_null(&get(*temperature)), get(electron_fraction),
           spatial_metric, eos, &root_find_failed),
       ...);
      if (root_find_failed) {
        // `fix_at_point` may have partially fixed the variables
        *rest_mass_density = reconstructed_rest_mass_density;
        *specific_internal_energy = previous_specific_internal_energy;
        *spatial_velocity = unfixed_spatial_velocity;
        *lorentz_factor = unfixed_lorentz_factor;
        *pressure = previous_pressure;
        *temperature = reconstructed_temperature;
        return {false, true};
      }
    }

    // 3. The spatial velocity one-form
    raise_or_lower_index(spatial_velocity_one_form, *spatial_velocity,
                         spatial_metric);

    // 4. The specific internal energy and pressure
    *specific_internal_energy =
        eos.specific_internal_energy_from_density_and_temperature(
            *rest_mass_density, *temperature, electron_fraction);
    *pressure = eos.pressure_from_density_and_temperature(
        *rest_mass_density, *temperature, electron_fraction);

    // 5. The conserved variables
    ConservativeFromPrimitive::apply(
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
        *rest_mass_density, electron_fraction, *specific_internal_energy,
        *pressure, *spatial_velocity, *lorentz_factor, magnetic_field,
        sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field);
    return {false, false};
  }
};
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
