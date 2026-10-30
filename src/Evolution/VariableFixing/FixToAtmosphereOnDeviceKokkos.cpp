// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#endif  // SPECTRE_KOKKOS

namespace VariableFixing {
#ifdef SPECTRE_KOKKOS
namespace {
// The `FixToAtmosphere::DeviceView` as an argument of the kernel
struct FixToAtmosphereView : db::SimpleTag {
  using type = FixToAtmosphere<3>::DeviceView;
};

// Fixes the primitive variables at a grid point with
// `FixToAtmosphere::DeviceView::fix_at_point`. The `return_tags` are the fixed
// variables and the first `argument_tags` are the same variables before fixing.
// If the root find of the kappa limiting fails, the status is `failed`.
template <typename DeviceView>
struct FixToAtmosphereAtGridPoint {
  using return_tags = typename FixToAtmosphere<3>::return_tags;
  using argument_tags =
      tmpl::append<return_tags,
                   tmpl::list<hydro::Tags::ElectronFraction<DataVector>,
                              gr::Tags::SpatialMetric<DataVector, 3>,
                              EquationsOfState::Tags::DeviceView<DeviceView>,
                              FixToAtmosphereView>>;

  KOKKOS_FUNCTION static PointwiseStatus apply(
      const gsl::not_null<Scalar<double>*> rest_mass_density,
      const gsl::not_null<Scalar<double>*> specific_internal_energy,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          spatial_velocity,
      const gsl::not_null<Scalar<double>*> lorentz_factor,
      const gsl::not_null<Scalar<double>*> pressure,
      const gsl::not_null<Scalar<double>*> temperature,
      const Scalar<double>& unfixed_rest_mass_density,
      const Scalar<double>& unfixed_specific_internal_energy,
      const tnsr::I<double, 3, Frame::Inertial>& unfixed_spatial_velocity,
      const Scalar<double>& unfixed_lorentz_factor,
      const Scalar<double>& unfixed_pressure,
      const Scalar<double>& unfixed_temperature,
      const Scalar<double>& electron_fraction,
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
      const DeviceView& equation_of_state,
      const FixToAtmosphere<3>::DeviceView& fix_to_atmosphere) {
    *rest_mass_density = unfixed_rest_mass_density;
    *specific_internal_energy = unfixed_specific_internal_energy;
    *spatial_velocity = unfixed_spatial_velocity;
    *lorentz_factor = unfixed_lorentz_factor;
    *pressure = unfixed_pressure;
    *temperature = unfixed_temperature;
    bool root_find_failed = false;
    fix_to_atmosphere.template fix_at_point<3>(
        make_not_null(&get(*rest_mass_density)),
        make_not_null(&get(*specific_internal_energy)), spatial_velocity,
        make_not_null(&get(*lorentz_factor)), make_not_null(&get(*pressure)),
        make_not_null(&get(*temperature)), get(electron_fraction),
        spatial_metric, equation_of_state, &root_find_failed);
    return {false, root_find_failed};
  }
};
}  // namespace

void fix_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
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
    const FixToAtmosphere<3>& fix_to_atmosphere) {
  using return_tags = typename FixToAtmosphere<3>::return_tags;
  const size_t number_of_points = get(*rest_mass_density).size();
  Variables<return_tags> fixed{number_of_points};
  PointwiseStatusSummary status{};
  EquationsOfState::visit_device_view(
      equation_of_state, [&](const auto& device_view) {
        using Function =
            FixToAtmosphereAtGridPoint<std::decay_t<decltype(device_view)>>;
        status = copy_and_apply_pointwise_on_device<Function>(
            exec, make_not_null(&fixed), return_tags{},
            typename Function::argument_tags{}, *rest_mass_density,
            *specific_internal_energy, *spatial_velocity, *lorentz_factor,
            *pressure, *temperature, electron_fraction, spatial_metric,
            device_view, fix_to_atmosphere.device_view());
      });
  if (status.number_of_failures > 0) {
    // Only the inputs at the failed grid point are needed for the error, so
    // this also works once the data stays on the device
    const size_t i = status.first_failed_grid_point;
    ERROR(
        "The root find of the kappa limiting in FixToAtmosphere failed on the "
        "device at "
        << status.number_of_failures << " of " << number_of_points
        << " grid points. At the first one, grid point " << i
        << ", before fixing:\n  rest_mass_density = "
        << extract_point(*rest_mass_density, i)
        << "\n  specific_internal_energy = "
        << extract_point(*specific_internal_energy, i)
        << "\n  spatial_velocity =\n"
        << extract_point(*spatial_velocity, i)
        << "\n  lorentz_factor = " << extract_point(*lorentz_factor, i)
        << "\n  pressure = " << extract_point(*pressure, i)
        << "\n  temperature = " << extract_point(*temperature, i)
        << "\n  electron_fraction = " << extract_point(electron_fraction, i)
        << "\nRun on the host for the details of the root find.");
  }
  *rest_mass_density = get<hydro::Tags::RestMassDensity<DataVector>>(fixed);
  *specific_internal_energy =
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(fixed);
  *spatial_velocity = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(fixed);
  *lorentz_factor = get<hydro::Tags::LorentzFactor<DataVector>>(fixed);
  *pressure = get<hydro::Tags::Pressure<DataVector>>(fixed);
  *temperature = get<hydro::Tags::Temperature<DataVector>>(fixed);
}

void fix_to_atmosphere_on_device(
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
    const FixToAtmosphere<3>& fix_to_atmosphere) {
  fix_to_atmosphere_on_device(Kokkos::DefaultExecutionSpace{},
                              rest_mass_density, specific_internal_energy,
                              spatial_velocity, lorentz_factor, pressure,
                              temperature, electron_fraction, spatial_metric,
                              equation_of_state, fix_to_atmosphere);
}
#endif  // SPECTRE_KOKKOS

void fix_to_atmosphere_on_host_or_device(
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
    const FixToAtmosphere<3>& fix_to_atmosphere) {
#ifdef SPECTRE_KOKKOS
  fix_to_atmosphere_on_device(rest_mass_density, specific_internal_energy,
                              spatial_velocity, lorentz_factor, pressure,
                              temperature, electron_fraction, spatial_metric,
                              equation_of_state, fix_to_atmosphere);
#else
  fix_to_atmosphere(rest_mass_density, specific_internal_energy,
                    spatial_velocity, lorentz_factor, pressure, temperature,
                    electron_fraction, spatial_metric, equation_of_state);
#endif  // SPECTRE_KOKKOS
}
}  // namespace VariableFixing
