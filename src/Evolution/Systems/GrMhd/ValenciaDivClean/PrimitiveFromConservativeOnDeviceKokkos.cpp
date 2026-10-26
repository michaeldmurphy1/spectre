// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDeviceKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <boost/preprocessor/tuple/elem.hpp>
#include <cstddef>
#include <iomanip>
#include <string>
#include <type_traits>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAlHydro.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAlHydro.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
// Whether the `DeviceView` of the EOS has the bounds that the recovery uses
template <typename DeviceView>
concept has_bounds = requires(const DeviceView& eos) {
  eos.electron_fraction_upper_bound();
  eos.specific_enthalpy_lower_bound();
};

// Whether `tilde_d` is below the cutoff for the inversion on the whole grid of
// an element
struct WholeGridBelowCutoff : db::SimpleTag {
  using type = bool;
};

// Whether `tilde_d` is below `cutoff` on the whole grid of each element, as
// `PrimitiveFromConservative` decides for its grid
Kokkos::View<const bool*> whole_grid_below_cutoff_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const typename ::Tags::BatchView<Tags::TildeD>::type& tilde_d,
    const double cutoff) {
  const auto& tilde_d_view = get(tilde_d);
  const size_t number_of_grid_points = tilde_d_view.extent(0);
  const size_t number_of_elements = tilde_d_view.extent(1);
  const Kokkos::View<bool*> result{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "WholeGridBelowCutoff"),
      number_of_elements};
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  Kokkos::parallel_for(
      "WholeGridBelowCutoff",
      TeamPolicy(exec, static_cast<int>(number_of_elements), Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const auto e = static_cast<size_t>(team.league_rank());
        double max_tilde_d = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, number_of_grid_points),
            [&](const size_t i, double& local_max) {
              local_max = Kokkos::max(local_max, tilde_d_view(i, e));
            },
            Kokkos::Max<double>(max_tilde_d));
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { result(e) = max_tilde_d < cutoff; });
      });
  return result;
}

// Recovers the primitive variables at a grid point, like
// `PrimitiveFromConservative`. The primitive variables before the recovery are
// arguments, and the pressure among them is the initial guess. If all schemes
// fail, the status is `failed` and the primitive variables keep their values
// from before the recovery, so the caller can report them.
template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool EnforcePhysicality, typename DeviceView>
struct PrimitiveFromConservativeAtGridPoint {
  using return_tags =
      typename PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes,
                                         false>::return_tags;
  using argument_tags = tmpl::list<
      Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
      Tags::TildeB<>, Tags::TildePhi, gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::ElectronFraction<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::MagneticField<DataVector, 3>,
      hydro::Tags::DivergenceCleaningField<DataVector>,
      hydro::Tags::LorentzFactor<DataVector>, hydro::Tags::Pressure<DataVector>,
      hydro::Tags::Temperature<DataVector>,
      EquationsOfState::Tags::DeviceView<DeviceView>,
      Tags::PrimitiveFromConservativeOptions,
      ::Tags::PerElement<WholeGridBelowCutoff>>;

  KOKKOS_FUNCTION static PointwiseStatus apply(
      const gsl::not_null<Scalar<double>*> rest_mass_density,
      const gsl::not_null<Scalar<double>*> electron_fraction,
      const gsl::not_null<Scalar<double>*> specific_internal_energy,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          spatial_velocity,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> magnetic_field,
      const gsl::not_null<Scalar<double>*> divergence_cleaning_field,
      const gsl::not_null<Scalar<double>*> lorentz_factor,
      const gsl::not_null<Scalar<double>*> pressure,
      const gsl::not_null<Scalar<double>*> temperature,
      const Scalar<double>& tilde_d, const Scalar<double>& tilde_ye,
      const Scalar<double>& tilde_tau,
      const tnsr::i<double, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<double, 3, Frame::Inertial>& tilde_b,
      const Scalar<double>& tilde_phi,
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<double, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<double>& sqrt_det_spatial_metric,
      const Scalar<double>& previous_rest_mass_density,
      const Scalar<double>& previous_electron_fraction,
      const Scalar<double>& previous_specific_internal_energy,
      const tnsr::I<double, 3, Frame::Inertial>& previous_spatial_velocity,
      const tnsr::I<double, 3, Frame::Inertial>& previous_magnetic_field,
      const Scalar<double>& previous_divergence_cleaning_field,
      const Scalar<double>& previous_lorentz_factor,
      const Scalar<double>& initial_pressure,
      const Scalar<double>& previous_temperature, const DeviceView& eos,
      const PrimitiveFromConservativeOptions& options,
      const bool whole_grid_below_cutoff) {
    using std::max;
    using std::min;
    constexpr bool is_barotropic =
        PrimitiveFromConservative_detail::eos_is_barotropic<DeviceView>;
    const double sqrt_det_g = get(sqrt_det_spatial_metric);
    get(*divergence_cleaning_field) = get(tilde_phi) / sqrt_det_g;
    for (size_t i = 0; i < 3; ++i) {
      magnetic_field->get(i) = tilde_b.get(i) / sqrt_det_g;
    }
    *pressure = initial_pressure;

    if (whole_grid_below_cutoff) {
      // Like `PrimitiveFromConservative`, set the atmosphere first. This also
      // changes the initial guess of the pressure.
      get(*rest_mass_density) = options.density_when_skipping_inversion();
      get(*electron_fraction) = min(eos.electron_fraction_upper_bound(),
                                    max(get(tilde_ye) / get(tilde_d),
                                        eos.electron_fraction_lower_bound()));
      if constexpr (is_barotropic) {
        *specific_internal_energy =
            eos.specific_internal_energy_from_density_and_temperature(
                *rest_mass_density, Scalar<double>{0.0}, Scalar<double>{0.0});
      } else {
        get(*specific_internal_energy) =
            eos.specific_internal_energy_lower_bound(get(*rest_mass_density),
                                                     get(*electron_fraction));
      }
      *temperature = eos.temperature_from_density_and_energy(
          *rest_mass_density, *specific_internal_energy, *electron_fraction);
      *pressure = eos.pressure_from_density_and_temperature(
          *rest_mass_density, *temperature, *electron_fraction);
      get(*lorentz_factor) = 1.0;
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity->get(i) = 0.0;
      }
    }

    tnsr::I<double, 3, Frame::Inertial> tilde_s_upper{};
    raise_or_lower_index(make_not_null(&tilde_s_upper), tilde_s,
                         inv_spatial_metric);
    Scalar<double> momentum_density_squared{};
    dot_product(make_not_null(&momentum_density_squared), tilde_s,
                tilde_s_upper);
    Scalar<double> momentum_density_dot_magnetic_field{};
    dot_product(make_not_null(&momentum_density_dot_magnetic_field), tilde_s,
                *magnetic_field);
    Scalar<double> magnetic_field_squared{};
    dot_product(make_not_null(&magnetic_field_squared), *magnetic_field,
                *magnetic_field, spatial_metric);

    const bool recovered = PrimitiveFromConservative_detail::recover_at_point<
        OrderedListOfPrimitiveRecoverySchemes, EnforcePhysicality,
        PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes,
                                  false>::use_hydro_optimization>(
        make_not_null(&get(*rest_mass_density)),
        make_not_null(&get(*electron_fraction)),
        make_not_null(&get(*specific_internal_energy)), spatial_velocity,
        make_not_null(&get(*lorentz_factor)), make_not_null(&get(*pressure)),
        get(tilde_d), get(tilde_ye), get(tilde_tau) / sqrt_det_g,
        get(momentum_density_squared) / square(sqrt_det_g),
        get(momentum_density_dot_magnetic_field) / sqrt_det_g,
        get(magnetic_field_squared), get(tilde_d) / sqrt_det_g, sqrt_det_g,
        *magnetic_field, tilde_s_upper, eos, options);
    if (not recovered) {
      *rest_mass_density = previous_rest_mass_density;
      *electron_fraction = previous_electron_fraction;
      *specific_internal_energy = previous_specific_internal_energy;
      *spatial_velocity = previous_spatial_velocity;
      *magnetic_field = previous_magnetic_field;
      *divergence_cleaning_field = previous_divergence_cleaning_field;
      *lorentz_factor = previous_lorentz_factor;
      *pressure = initial_pressure;
      *temperature = previous_temperature;
      return {false, true};
    }

    if constexpr (is_barotropic) {
      // Project back to the sub-manifold that satisfies the EOS, see
      // `PrimitiveFromConservative`
      *specific_internal_energy =
          eos.specific_internal_energy_from_density_and_temperature(
              *rest_mass_density, Scalar<double>{0.0}, Scalar<double>{0.0});
    }
    *temperature = eos.temperature_from_density_and_energy(
        *rest_mass_density, *specific_internal_energy, *electron_fraction);
    return {false, false};
  }
};

}  // namespace

template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool EnforcePhysicality>
PointwiseStatusSummary primitive_from_conservative_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*>
        primitives,
    const typename ::Tags::BatchView<Tags::TildeD>::type& tilde_d,
    const typename ::Tags::BatchView<Tags::TildeYe>::type& tilde_ye,
    const typename ::Tags::BatchView<Tags::TildeTau>::type& tilde_tau,
    const typename ::Tags::BatchView<Tags::TildeS<>>::type& tilde_s,
    const typename ::Tags::BatchView<Tags::TildeB<>>::type& tilde_b,
    const typename ::Tags::BatchView<Tags::TildePhi>::type& tilde_phi,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const PrimitiveFromConservativeOptions&
        primitive_from_conservative_options) {
  const Kokkos::View<const bool*> whole_grid_below_cutoff =
      whole_grid_below_cutoff_on_device(
          exec, tilde_d,
          primitive_from_conservative_options.cutoff_d_for_inversion());
  PointwiseStatusSummary status{};
  EquationsOfState::visit_device_view(equation_of_state, [&](const auto&
                                                                 device_view) {
    using DeviceView = std::decay_t<decltype(device_view)>;
    static_assert(has_bounds<DeviceView>,
                  "The DeviceView has no bounds for primitive recovery.");
    using Function = PrimitiveFromConservativeAtGridPoint<
        OrderedListOfPrimitiveRecoverySchemes, EnforcePhysicality, DeviceView>;
    // The primitive variables are both arguments and results. This is safe
    // because the arguments at a grid point are read before the results
    // there are written.
    const auto arguments =
        [&]<typename... ArgumentTags, typename... PrimitiveTags>(
            tmpl::list<ArgumentTags...> /*meta*/,
            tmpl::list<PrimitiveTags...> /*meta*/) {
          return make_device_arguments<ArgumentTags...>(
              exec, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
              spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
              get<::Tags::BatchView<PrimitiveTags>>(*primitives)...,
              device_view, primitive_from_conservative_options,
              whole_grid_below_cutoff);
        }(typename Function::argument_tags{}, hydro::grmhd_tags<DataVector>{});
    status = apply_pointwise_on_device<Function>(
        exec, primitives, typename Function::return_tags{},
        typename Function::argument_tags{}, arguments);
  });
  return status;
}

void error_recovering_primitives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const PointwiseStatusSummary& status,
    const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives,
    const typename ::Tags::BatchView<Tags::TildeD>::type& tilde_d,
    const typename ::Tags::BatchView<Tags::TildeTau>::type& tilde_tau,
    const typename ::Tags::BatchView<Tags::TildeS<>>::type& tilde_s,
    const typename ::Tags::BatchView<Tags::TildeB<>>::type& tilde_b,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const std::string& element_description) {
  ASSERT(status.number_of_failures > 0,
         "There is no failure to report: primitive_from_conservative_on_device "
         "succeeded at all grid points.");
  const size_t s = status.first_failed_grid_point;
  const size_t e = status.first_failed_element;
  const auto point = [&exec, s, e](const auto& batch_tensor) {
    return extract_point_to_host(exec, batch_tensor, s, e);
  };
  const double sqrt_det_g = get(point(sqrt_det_spatial_metric));
  const auto tilde_s_at_s = point(tilde_s);
  const auto tilde_b_at_s = point(tilde_b);
  const auto spatial_metric_at_s = point(spatial_metric);
  const auto inv_spatial_metric_at_s = point(inv_spatial_metric);
  double momentum_density_squared = 0.0;
  double momentum_density_dot_magnetic_field = 0.0;
  double magnetic_field_squared = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    momentum_density_dot_magnetic_field +=
        tilde_s_at_s.get(i) * tilde_b_at_s.get(i);
    for (size_t j = 0; j < 3; ++j) {
      momentum_density_squared += inv_spatial_metric_at_s.get(i, j) *
                                  tilde_s_at_s.get(i) * tilde_s_at_s.get(j);
      magnetic_field_squared += spatial_metric_at_s.get(i, j) *
                                tilde_b_at_s.get(i) * tilde_b_at_s.get(j);
    }
  }
  // The primitive variables at failed grid points hold the values before the
  // recovery
  ERROR("All primitive inversion schemes failed on the device at "
        << status.number_of_failures << " grid points. The first is s = " << s
        << " of " << element_description << ".\n"
        << std::setprecision(17)
        << "tau = " << get(point(tilde_tau)) / sqrt_det_g << "\n"
        << "rest_mass_density_times_lorentz_factor = "
        << get(point(tilde_d)) / sqrt_det_g << "\n"
        << "momentum_density_squared = "
        << momentum_density_squared / square(sqrt_det_g) << "\n"
        << "momentum_density_dot_magnetic_field = "
        << momentum_density_dot_magnetic_field / square(sqrt_det_g) << "\n"
        << "magnetic_field_squared = "
        << magnetic_field_squared / square(sqrt_det_g) << "\n"
        << "previous_rest_mass_density = "
        << get(point(
               get<::Tags::BatchView<hydro::Tags::RestMassDensity<DataVector>>>(
                   primitives)))
        << "\n"
        << "previous_pressure = "
        << get(point(get<::Tags::BatchView<hydro::Tags::Pressure<DataVector>>>(
               primitives)))
        << "\n"
        << "previous_lorentz_factor = "
        << get(point(
               get<::Tags::BatchView<hydro::Tags::LorentzFactor<DataVector>>>(
                   primitives)))
        << "\n");
}

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define PHYSICALITY(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                              \
  template PointwiseStatusSummary                                           \
  primitive_from_conservative_on_device<RECOVERY(data), PHYSICALITY(data)>( \
      const Kokkos::DefaultExecutionSpace& exec,                            \
      gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*>       \
          primitives,                                                       \
      const typename ::Tags::BatchView<Tags::TildeD>::type& tilde_d,        \
      const typename ::Tags::BatchView<Tags::TildeYe>::type& tilde_ye,      \
      const typename ::Tags::BatchView<Tags::TildeTau>::type& tilde_tau,    \
      const typename ::Tags::BatchView<Tags::TildeS<>>::type& tilde_s,      \
      const typename ::Tags::BatchView<Tags::TildeB<>>::type& tilde_b,      \
      const typename ::Tags::BatchView<Tags::TildePhi>::type& tilde_phi,    \
      const typename ::Tags::BatchView<                                     \
          gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,    \
      const typename ::Tags::BatchView<                                     \
          gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&             \
          inv_spatial_metric,                                               \
      const typename ::Tags::BatchView<                                     \
          gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&                \
          sqrt_det_spatial_metric,                                          \
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,  \
      const PrimitiveFromConservativeOptions&                               \
          primitive_from_conservative_options);

namespace {
using NewmanHamlinThenPalenzuelaEtAl =
    tmpl::list<PrimitiveRecoverySchemes::NewmanHamlin,
               PrimitiveRecoverySchemes::PalenzuelaEtAl>;
using KastaunThenNewmanThenPalenzuela =
    tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl,
               PrimitiveRecoverySchemes::NewmanHamlin,
               PrimitiveRecoverySchemes::PalenzuelaEtAl>;
}  // namespace

// The recovery schemes used in the subcell actions
GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl>,
                         tmpl::list<PrimitiveRecoverySchemes::NewmanHamlin>,
                         tmpl::list<PrimitiveRecoverySchemes::PalenzuelaEtAl>,
                         NewmanHamlinThenPalenzuelaEtAl,
                         KastaunThenNewmanThenPalenzuela),
                        (true, false))

#undef INSTANTIATION
#undef PHYSICALITY
#undef RECOVERY
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
