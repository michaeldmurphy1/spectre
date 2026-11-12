// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace VariableFixing {
namespace {
// The `FixToAtmosphere::DeviceView` as an argument of the kernel
struct FixToAtmosphereView : db::SimpleTag {
  using type = FixToAtmosphere<3>::DeviceView;
};

// Fixes the primitive variables at a grid point with
// `FixToAtmosphere::DeviceView::fix_at_point`. The `return_tags` are the fixed
// variables and the first `argument_tags` are the same variables before fixing.
// If the root find of the kappa limiting fails, the status is `failed` and the
// returned variables are the unfixed ones.
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
    if (root_find_failed) {
      // `fix_at_point` may have partially fixed the variables
      *rest_mass_density = unfixed_rest_mass_density;
      *specific_internal_energy = unfixed_specific_internal_energy;
      *spatial_velocity = unfixed_spatial_velocity;
      *lorentz_factor = unfixed_lorentz_factor;
      *pressure = unfixed_pressure;
      *temperature = unfixed_temperature;
    }
    return {false, root_find_failed};
  }
};

// Fixes with a `PointwiseStatusSummary` result if `status` is `nullptr`, and
// else records the status in `*status`
template <typename StatusPointer>
PointwiseStatusSummary fix_to_atmosphere_impl(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*>
        primitives,
    const StatusPointer status,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>::DeviceView& fix_to_atmosphere) {
  using return_tags = typename FixToAtmosphere<3>::return_tags;
  PointwiseStatusSummary result{};
  EquationsOfState::visit_device_view(
      equation_of_state, [&](const auto& device_view) {
        using Function =
            FixToAtmosphereAtGridPoint<std::decay_t<decltype(device_view)>>;
        // The variables are both arguments and results. This is safe because
        // the arguments at a grid point are read before the results there are
        // written.
        const auto arguments = [&]<typename... ReturnTags>(
                                   tmpl::list<ReturnTags...> /*meta*/) {
          return make_device_arguments<
              ReturnTags..., hydro::Tags::ElectronFraction<DataVector>,
              gr::Tags::SpatialMetric<DataVector, 3>,
              EquationsOfState::Tags::DeviceView<
                  std::decay_t<decltype(device_view)>>,
              FixToAtmosphereView>(
              exec, get<::Tags::BatchView<ReturnTags>>(*primitives)...,
              get<::Tags::BatchView<hydro::Tags::ElectronFraction<DataVector>>>(
                  *primitives),
              spatial_metric, device_view, fix_to_atmosphere);
        }(return_tags{});
        result = ApplyPointwiseOnDevice_detail::apply_with_status<Function>(
            exec, primitives, status, return_tags{},
            typename Function::argument_tags{}, arguments);
      });
  return result;
}
}  // namespace

PointwiseStatusSummary fix_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*>
        primitives,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>::DeviceView& fix_to_atmosphere) {
  return fix_to_atmosphere_impl(exec, primitives, nullptr, spatial_metric,
                                equation_of_state, fix_to_atmosphere);
}

void fix_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*>
        primitives,
    const gsl::not_null<
        DeferredPointwiseStatus<fix_to_atmosphere_snapshot_tags>*>
        status,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>::DeviceView& fix_to_atmosphere) {
  fix_to_atmosphere_impl(exec, primitives, status.get(), spatial_metric,
                         equation_of_state, fix_to_atmosphere);
}

}  // namespace VariableFixing
#endif  // SPECTRE_KOKKOS
