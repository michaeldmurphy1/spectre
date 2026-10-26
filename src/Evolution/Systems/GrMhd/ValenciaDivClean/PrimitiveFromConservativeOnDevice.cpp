// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDevice.hpp"

#ifdef SPECTRE_KOKKOS
#include <boost/preprocessor/tuple/elem.hpp>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <type_traits>

#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
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
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
// Whether the `DeviceView` of the EOS has the bounds that the recovery uses.
// Not all `DeviceView`s have them yet.
template <typename DeviceView>
concept has_bounds = requires(const DeviceView& eos) {
  eos.electron_fraction_upper_bound();
  eos.specific_enthalpy_lower_bound();
};

// Whether `tilde_d` is below the cutoff for the inversion on the whole grid
struct WholeGridBelowCutoff : db::SimpleTag {
  using type = bool;
};

// Recovers the primitive variables at a grid point, like
// `PrimitiveFromConservative`. If all schemes fail, the status is `failed` and
// the primitive variables are NaN.
template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool EnforcePhysicality, typename DeviceView>
struct PrimitiveFromConservativeAtGridPoint {
  using return_tags =
      typename PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes,
                                         false>::return_tags;
  using argument_tags =
      tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
                 Tags::TildeB<>, Tags::TildePhi,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 EquationsOfState::Tags::DeviceView<DeviceView>,
                 Tags::PrimitiveFromConservativeOptions, WholeGridBelowCutoff>;

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
      const Scalar<double>& initial_pressure, const DeviceView& eos,
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
      const double nan = std::numeric_limits<double>::quiet_NaN();
      get(*rest_mass_density) = nan;
      get(*electron_fraction) = nan;
      get(*specific_internal_energy) = nan;
      get(*lorentz_factor) = nan;
      get(*pressure) = nan;
      get(*temperature) = nan;
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity->get(i) = nan;
      }
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

// The same error as `PrimitiveFromConservative` gives at the grid point `s`,
// where `rest_mass_density`, `pressure` and `lorentz_factor` still hold the
// values before the recovery
[[noreturn]] void error_at_grid_point(
    const size_t s, const size_t number_of_failures,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& pressure, const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  const double sqrt_det_g = get(sqrt_det_spatial_metric)[s];
  double momentum_density_squared = 0.0;
  double momentum_density_dot_magnetic_field = 0.0;
  double magnetic_field_squared = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    momentum_density_dot_magnetic_field +=
        tilde_s.get(i)[s] * tilde_b.get(i)[s];
    for (size_t j = 0; j < 3; ++j) {
      momentum_density_squared += inv_spatial_metric.get(i, j)[s] *
                                  tilde_s.get(i)[s] * tilde_s.get(j)[s];
      magnetic_field_squared +=
          spatial_metric.get(i, j)[s] * tilde_b.get(i)[s] * tilde_b.get(j)[s];
    }
  }
  const double rest_mass_density_times_lorentz_factor =
      get(tilde_d)[s] / sqrt_det_g;
  ERROR("All primitive inversion schemes failed on the device at "
        << number_of_failures << " grid points. The first is s = " << s << ".\n"
        << std::setprecision(17) << "tau = " << get(tilde_tau)[s] / sqrt_det_g
        << "\n"
        << "rest_mass_density_times_lorentz_factor = "
        << rest_mass_density_times_lorentz_factor << "\n"
        << "momentum_density_squared = "
        << momentum_density_squared / square(sqrt_det_g) << "\n"
        << "momentum_density_dot_magnetic_field = "
        << momentum_density_dot_magnetic_field / square(sqrt_det_g) << "\n"
        << "magnetic_field_squared = "
        << magnetic_field_squared / square(sqrt_det_g) << "\n"
        << "previous_rest_mass_density = " << get(rest_mass_density)[s] << "\n"
        << "previous_pressure = " << get(pressure)[s] << "\n"
        << "previous_lorentz_factor = " << get(lorentz_factor)[s] << "\n");
}
}  // namespace

template <typename OrderedListOfPrimitiveRecoverySchemes, bool ErrorOnFailure,
          bool EnforcePhysicality>
bool primitive_from_conservative_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const PrimitiveFromConservativeOptions&
        primitive_from_conservative_options) {
  const size_t number_of_points = get(tilde_d).size();
  ASSERT(get(*pressure).size() == number_of_points,
         "The initial guess of the pressure must have "
             << number_of_points << " grid points, but has "
             << get(*pressure).size());
  const bool whole_grid_below_cutoff =
      max(get(tilde_d)) <
      primitive_from_conservative_options.cutoff_d_for_inversion();
  PointwiseStatusSummary status{};
  EquationsOfState::visit_device_view(
      equation_of_state, [&](const auto& device_view) {
        using DeviceView = std::decay_t<decltype(device_view)>;
        if constexpr (not has_bounds<DeviceView>) {
          ERROR(
              "Primitive recovery on the device isn't implemented yet for "
              "the equation of state "
              << pretty_type::get_runtime_type_name(equation_of_state)
              << ", because its DeviceView has no bounds.");
        } else {
          using Function = PrimitiveFromConservativeAtGridPoint<
              OrderedListOfPrimitiveRecoverySchemes, EnforcePhysicality,
              DeviceView>;
          using return_tags = typename Function::return_tags;
          Variables<return_tags> results{number_of_points};
          status = copy_and_apply_pointwise_on_device<Function>(
              exec, make_not_null(&results), return_tags{},
              typename Function::argument_tags{}, tilde_d, tilde_ye, tilde_tau,
              tilde_s, tilde_b, tilde_phi, spatial_metric, inv_spatial_metric,
              sqrt_det_spatial_metric, *pressure, device_view,
              primitive_from_conservative_options, whole_grid_below_cutoff);
          if constexpr (ErrorOnFailure) {
            if (status.number_of_failures > 0) {
              // Only the inputs at the failed grid point are needed for the
              // error, so this also works once the data stays on the device
              error_at_grid_point(status.first_failed_grid_point,
                                  status.number_of_failures, *rest_mass_density,
                                  *lorentz_factor, *pressure, tilde_d,
                                  tilde_tau, tilde_s, tilde_b, spatial_metric,
                                  inv_spatial_metric, sqrt_det_spatial_metric);
            }
          }
          *rest_mass_density =
              get<hydro::Tags::RestMassDensity<DataVector>>(results);
          *electron_fraction =
              get<hydro::Tags::ElectronFraction<DataVector>>(results);
          *specific_internal_energy =
              get<hydro::Tags::SpecificInternalEnergy<DataVector>>(results);
          *spatial_velocity =
              get<hydro::Tags::SpatialVelocity<DataVector, 3>>(results);
          *magnetic_field =
              get<hydro::Tags::MagneticField<DataVector, 3>>(results);
          *divergence_cleaning_field =
              get<hydro::Tags::DivergenceCleaningField<DataVector>>(results);
          *lorentz_factor =
              get<hydro::Tags::LorentzFactor<DataVector>>(results);
          *pressure = get<hydro::Tags::Pressure<DataVector>>(results);
          *temperature = get<hydro::Tags::Temperature<DataVector>>(results);
        }
      });
  return status.number_of_failures == 0;
}

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define ERROR_ON_FAILURE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define PHYSICALITY(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(_, data)                                                \
  template bool primitive_from_conservative_on_device<                        \
      RECOVERY(data), ERROR_ON_FAILURE(data), PHYSICALITY(data)>(             \
      const Kokkos::DefaultExecutionSpace& exec,                              \
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,                   \
      gsl::not_null<Scalar<DataVector>*> electron_fraction,                   \
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,            \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>                 \
          spatial_velocity,                                                   \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field, \
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,           \
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,                      \
      gsl::not_null<Scalar<DataVector>*> pressure,                            \
      gsl::not_null<Scalar<DataVector>*> temperature,                         \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,  \
      const Scalar<DataVector>& tilde_tau,                                    \
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DataVector>& tilde_phi,                                    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                      \
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,    \
      const PrimitiveFromConservativeOptions&                                 \
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
                        (true), (true))
// Used in tests
GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl>,
                         tmpl::list<PrimitiveRecoverySchemes::NewmanHamlin>,
                         tmpl::list<PrimitiveRecoverySchemes::PalenzuelaEtAl>,
                         NewmanHamlinThenPalenzuelaEtAl,
                         KastaunThenNewmanThenPalenzuela),
                        (false), (true, false))

#undef INSTANTIATION
#undef PHYSICALITY
#undef ERROR_ON_FAILURE
#undef RECOVERY
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
