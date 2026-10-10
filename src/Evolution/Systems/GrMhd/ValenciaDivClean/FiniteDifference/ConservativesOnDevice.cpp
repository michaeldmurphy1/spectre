// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ConservativesOnDevice.hpp"

#ifdef SPECTRE_KOKKOS
#include <cstddef>
#include <type_traits>

#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArguments.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
namespace {
// The `DeviceView` of an EOS as an argument of a pointwise function
template <typename DeviceView>
struct EosDeviceView : db::SimpleTag {
  using type = DeviceView;
};

// Computes the specific internal energy and pressure with the 3D EOS, and then
// the conserved variables, at a grid point
template <typename DeviceView>
struct ConservativesFromDensityAndTemperature {
  using return_tags =
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>, Tags::TildeD, Tags::TildeYe,
                 Tags::TildeTau, Tags::TildeS<>, Tags::TildeB<>,
                 Tags::TildePhi>;
  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 EosDeviceView<DeviceView>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> specific_internal_energy,
      const gsl::not_null<Scalar<double>*> pressure,
      const gsl::not_null<Scalar<double>*> tilde_d,
      const gsl::not_null<Scalar<double>*> tilde_ye,
      const gsl::not_null<Scalar<double>*> tilde_tau,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*> tilde_s,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> tilde_b,
      const gsl::not_null<Scalar<double>*> tilde_phi,
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& electron_fraction,
      const Scalar<double>& temperature,
      const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<double>& lorentz_factor,
      const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
      const Scalar<double>& sqrt_det_spatial_metric,
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
      const Scalar<double>& divergence_cleaning_field, const DeviceView& eos) {
    *specific_internal_energy =
        eos.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction);
    *pressure = eos.pressure_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction);
    ConservativeFromPrimitive::apply(
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
        rest_mass_density, electron_fraction, *specific_internal_energy,
        *pressure, spatial_velocity, lorentz_factor, magnetic_field,
        sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field);
  }
};

// Enqueues the copy of the result `Tag` in `device_results` into `host_tensor`
template <typename Tag, typename DeviceVars>
void copy_result_to_host(const Kokkos::DefaultExecutionSpace& exec,
                         const gsl::not_null<typename Tag::type*> host_tensor,
                         const DeviceVars& device_results) {
  using HostUnmanaged =
      Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  const auto& device_tensor = get<::Tags::MirrorView<Tag>>(device_results);
  for (size_t i = 0; i < host_tensor->size(); ++i) {
    const HostUnmanaged host_view((*host_tensor)[i].data(),
                                  (*host_tensor)[i].size());
    Kokkos::deep_copy(exec, host_view, device_tensor[i]);
  }
}
}  // namespace

void compute_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Scalar<DataVector>& divergence_cleaning_field,
    const EquationsOfState::EquationOfState<true, 3>& eos) {
  EquationsOfState::visit_device_view(eos, [&](const auto& device_view) {
    using Function = ConservativesFromDensityAndTemperature<
        std::decay_t<decltype(device_view)>>;
    using return_tags = typename Function::return_tags;
    using argument_tags = typename Function::argument_tags;

    // The results are written directly into the host tensors rather than
    // into a `Variables`, so this doesn't use
    // `copy_and_apply_pointwise_on_device`
    using DeviceResults = Variables<
        tmpl::transform<return_tags, tmpl::bind<::Tags::MirrorView, tmpl::_1>>>;
    DeviceResults device_results{typename DeviceResults::storage_type{
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "results"),
        get(rest_mass_density).size()}};
    const auto device_args = tmpl::as_pack<argument_tags>([&](auto... tags) {
      return make_device_arguments<tmpl::type_from<decltype(tags)>...>(
          exec, rest_mass_density, electron_fraction, temperature,
          spatial_velocity, lorentz_factor, magnetic_field,
          sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field,
          device_view);
    });
    apply_pointwise_on_device<Function>(exec, make_not_null(&device_results),
                                        return_tags{}, argument_tags{},
                                        device_args);

    copy_result_to_host<hydro::Tags::SpecificInternalEnergy<DataVector>>(
        exec, specific_internal_energy, device_results);
    copy_result_to_host<hydro::Tags::Pressure<DataVector>>(exec, pressure,
                                                           device_results);
    copy_result_to_host<Tags::TildeD>(exec, tilde_d, device_results);
    copy_result_to_host<Tags::TildeYe>(exec, tilde_ye, device_results);
    copy_result_to_host<Tags::TildeTau>(exec, tilde_tau, device_results);
    copy_result_to_host<Tags::TildeS<>>(exec, tilde_s, device_results);
    copy_result_to_host<Tags::TildeB<>>(exec, tilde_b, device_results);
    copy_result_to_host<Tags::TildePhi>(exec, tilde_phi, device_results);
    // The host data and device views must stay alive until all work on
    // `exec` is complete
    exec.fence("compute_conservatives_on_device");
  });
}
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
