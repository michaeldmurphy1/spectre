// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ConservativesOnDeviceKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
namespace {
// The variables on a face at a grid point, computed like
// `compute_conservatives_for_reconstruction`
template <typename DeviceView>
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
  using argument_tags =
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
                 EquationsOfState::Tags::DeviceView<DeviceView>>;

  KOKKOS_FUNCTION static void apply(
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
      const Scalar<double>& sqrt_det_spatial_metric, const DeviceView& eos) {
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

    // 2. The spatial velocity one-form
    raise_or_lower_index(spatial_velocity_one_form, *spatial_velocity,
                         spatial_metric);

    // 3. The specific internal energy and pressure
    *specific_internal_energy =
        eos.specific_internal_energy_from_density_and_temperature(
            *rest_mass_density, *temperature, electron_fraction);
    *pressure = eos.pressure_from_density_and_temperature(
        *rest_mass_density, *temperature, electron_fraction);

    // 4. The conserved variables
    ConservativeFromPrimitive::apply(
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
        *rest_mass_density, electron_fraction, *specific_internal_energy,
        *pressure, *spatial_velocity, *lorentz_factor, magnetic_field,
        sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field);
  }
};
}  // namespace

void compute_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<face_tags>*> faces,
    const EquationsOfState::EquationOfState<true, 3>& eos) {
  EquationsOfState::visit_device_view(eos, [&](const auto& device_view) {
    using Function =
        FaceVariablesAtGridPoint<std::decay_t<decltype(device_view)>>;
    // The variables are both arguments and results. This is safe because the
    // arguments at a grid point are read before the results there are
    // written.
    apply_pointwise_on_device<Function>(
        exec, faces, typename Function::return_tags{},
        typename Function::argument_tags{},
        [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
          // The last argument is the EOS
          return make_device_arguments<
              ArgumentTags..., tmpl::back<typename Function::argument_tags>>(
              exec, get<::Tags::BatchView<ArgumentTags>>(*faces)...,
              device_view);
        }(tmpl::pop_back<typename Function::argument_tags>{}));
  });
}
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
