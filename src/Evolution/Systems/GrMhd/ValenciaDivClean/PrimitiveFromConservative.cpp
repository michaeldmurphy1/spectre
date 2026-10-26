// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"

#include <iomanip>
#include <limits>
#include <optional>
#include <ostream>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean {
template <typename OrderedListOfPrimitiveRecoverySchemes, bool ErrorOnFailure>
template <bool EnforcePhysicality>
bool PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes,
                               ErrorOnFailure>::
    apply(const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
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
          const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
              primitive_from_conservative_options) {
  return call_with_dynamic_type<
      bool, typename EquationsOfState::detail::DerivedClasses<true, 3>::type>(
      &equation_of_state, [&](const auto* const derived_eos) {
        return impl<EnforcePhysicality>(
            rest_mass_density, electron_fraction, specific_internal_energy,
            spatial_velocity, magnetic_field, divergence_cleaning_field,
            lorentz_factor, pressure, temperature, tilde_d, tilde_ye, tilde_tau,
            tilde_s, tilde_b, tilde_phi, spatial_metric, inv_spatial_metric,
            sqrt_det_spatial_metric, *derived_eos,
            primitive_from_conservative_options);
      });
}

// If EnforceBarotropic then we assume the EOS is barotropic and enforce that
// condition.
template <typename OrderedListOfPrimitiveRecoverySchemes, bool ErrorOnFailure>
template <bool EnforcePhysicality, typename EosType>
bool PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes,
                               ErrorOnFailure>::
    impl(const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
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
         const EosType& equation_of_state,
         const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
             primitive_from_conservative_options) {
  static_assert(EosType::thermodynamic_dim == 3);
  static_assert(EosType::is_relativistic);
  constexpr bool eos_is_barotropic =
      tt::is_a_v<EquationsOfState::Barotropic3D, EosType>;

  get(*divergence_cleaning_field) =
      get(tilde_phi) / get(sqrt_det_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field->get(i) = tilde_b.get(i) / get(sqrt_det_spatial_metric);
  }

  // Parameters for quick exit from inversion
  const double cutoffD =
      primitive_from_conservative_options.cutoff_d_for_inversion();
  const double floorD =
      primitive_from_conservative_options.density_when_skipping_inversion();

  // If the max over the grid is below the cutoff, then just don't do any
  // work because everything will get reset to atmosphere.
  if (max(get(tilde_d)) < cutoffD) {
    get(*rest_mass_density) = floorD;
    get(*electron_fraction) =
        min(equation_of_state.electron_fraction_upper_bound(),
            max(get(tilde_ye) / get(tilde_d),
                equation_of_state.electron_fraction_lower_bound()));
    if constexpr (eos_is_barotropic) {
      // Note: default construction for T and Y_e must be okay since the EOS is
      // barotropic.
      *specific_internal_energy =
          equation_of_state
              .specific_internal_energy_from_density_and_temperature(
                  *rest_mass_density, Scalar<DataVector>{},
                  Scalar<DataVector>{});
    } else {
      const auto num_points = get(*rest_mass_density).size();
      for (size_t i = 0; i < num_points; ++i) {
        get(*specific_internal_energy)[i] =
            equation_of_state.specific_internal_energy_lower_bound(
                get(*rest_mass_density)[i], get(*electron_fraction)[i]);
      }
    }
    *temperature = equation_of_state.temperature_from_density_and_energy(
        *rest_mass_density, *specific_internal_energy, *electron_fraction);
    *pressure = equation_of_state.pressure_from_density_and_temperature(
        *rest_mass_density, *temperature, *electron_fraction);
    get(*lorentz_factor) = 1.0;
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity->get(i) = 0.0;
    }
  }

  const size_t number_of_points = get<0>(tilde_b).size();
  Variables<
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                 ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                 ::Tags::TempScalar<4>, ::Tags::TempI<5, 3, Frame::Inertial>>>
      temp_buffer(number_of_points);

  DataVector& tau = get(get<::Tags::TempScalar<0>>(temp_buffer));
  tau = get(tilde_tau) / get(sqrt_det_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial>& tilde_s_upper =
      get<::Tags::TempI<5, 3, Frame::Inertial>>(temp_buffer);
  raise_or_lower_index(make_not_null(&tilde_s_upper), tilde_s,
                       inv_spatial_metric);

  Scalar<DataVector>& momentum_density_squared =
      get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&momentum_density_squared), tilde_s, tilde_s_upper);
  get(momentum_density_squared) /= square(get(sqrt_det_spatial_metric));

  Scalar<DataVector>& momentum_density_dot_magnetic_field =
      get<::Tags::TempScalar<2>>(temp_buffer);
  dot_product(make_not_null(&momentum_density_dot_magnetic_field), tilde_s,
              *magnetic_field);
  get(momentum_density_dot_magnetic_field) /= get(sqrt_det_spatial_metric);

  Scalar<DataVector>& magnetic_field_squared =
      get<::Tags::TempScalar<3>>(temp_buffer);
  dot_product(make_not_null(&magnetic_field_squared), *magnetic_field,
              *magnetic_field, spatial_metric);

  DataVector& rest_mass_density_times_lorentz_factor =
      get(get<::Tags::TempScalar<4>>(temp_buffer));
  rest_mass_density_times_lorentz_factor =
      get(tilde_d) / get(sqrt_det_spatial_metric);

  for (size_t s = 0; s < number_of_points; ++s) {
    double rest_mass_density_at_point = get(*rest_mass_density)[s];
    double electron_fraction_at_point = get(*electron_fraction)[s];
    double specific_internal_energy_at_point =
        get(*specific_internal_energy)[s];
    tnsr::I<double, 3, Frame::Inertial> spatial_velocity_at_point{};
    tnsr::I<double, 3, Frame::Inertial> magnetic_field_at_point{};
    tnsr::I<double, 3, Frame::Inertial> tilde_s_upper_at_point{};
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity_at_point.get(i) = spatial_velocity->get(i)[s];
      magnetic_field_at_point.get(i) = magnetic_field->get(i)[s];
      tilde_s_upper_at_point.get(i) = tilde_s_upper.get(i)[s];
    }
    double lorentz_factor_at_point = get(*lorentz_factor)[s];
    double pressure_at_point = get(*pressure)[s];
    const bool recovered = PrimitiveFromConservative_detail::recover_at_point<
        OrderedListOfPrimitiveRecoverySchemes, EnforcePhysicality,
        use_hydro_optimization>(
        make_not_null(&rest_mass_density_at_point),
        make_not_null(&electron_fraction_at_point),
        make_not_null(&specific_internal_energy_at_point),
        make_not_null(&spatial_velocity_at_point),
        make_not_null(&lorentz_factor_at_point),
        make_not_null(&pressure_at_point), get(tilde_d)[s], get(tilde_ye)[s],
        tau[s], get(momentum_density_squared)[s],
        get(momentum_density_dot_magnetic_field)[s],
        get(magnetic_field_squared)[s],
        rest_mass_density_times_lorentz_factor[s],
        get(sqrt_det_spatial_metric)[s], magnetic_field_at_point,
        tilde_s_upper_at_point, equation_of_state,
        primitive_from_conservative_options);
    get(*electron_fraction)[s] = electron_fraction_at_point;

    if (recovered) {
      get(*rest_mass_density)[s] = rest_mass_density_at_point;
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity->get(i)[s] = spatial_velocity_at_point.get(i);
      }
      get(*lorentz_factor)[s] = lorentz_factor_at_point;
      get(*pressure)[s] = pressure_at_point;
      get(*specific_internal_energy)[s] = specific_internal_energy_at_point;
    } else {
      if constexpr (ErrorOnFailure) {
        ERROR("All primitive inversion schemes failed at s = "
              << s << ".\n"
              << std::setprecision(17) << "tau = " << tau[s] << "\n"
              << "rest_mass_density_times_lorentz_factor = "
              << rest_mass_density_times_lorentz_factor[s] << "\n"
              << "momentum_density_squared = "
              << get(momentum_density_squared)[s] << "\n"
              << "momentum_density_dot_magnetic_field = "
              << get(momentum_density_dot_magnetic_field)[s] << "\n"
              << "magnetic_field_squared = " << get(magnetic_field_squared)[s]
              << "\n"
              << "rest_mass_density_times_lorentz_factor = "
              << rest_mass_density_times_lorentz_factor[s] << "\n"
              << "previous_rest_mass_density = " << get(*rest_mass_density)[s]
              << "\n"
              << "previous_pressure = " << get(*pressure)[s] << "\n"
              << "previous_lorentz_factor = " << get(*lorentz_factor)[s]
              << "\n");
      } else {
        return false;
      }
    }
  }
  if constexpr (eos_is_barotropic) {
    // Since the primitive recovery scheme is not restricted to lie on the
    // EOS-satisfying sub-manifold, we project back to the sub-manifold by
    // recomputing the specific internal energy from the EOS.
    //
    // Note: default construction for T and Y_e must be okay since the EOS is
    // barotropic.
    *specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density_and_temperature(
            *rest_mass_density, Scalar<DataVector>{}, Scalar<DataVector>{});
  }
  *temperature = equation_of_state.temperature_from_density_and_energy(
      *rest_mass_density, *specific_internal_energy, *electron_fraction);
  return true;
}
}  // namespace grmhd::ValenciaDivClean

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define ERROR_ON_FAILURE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                        \
  template struct grmhd::ValenciaDivClean::PrimitiveFromConservative< \
      RECOVERY(data), ERROR_ON_FAILURE(data)>;

using NewmanHamlinThenPalenzuelaEtAl = tmpl::list<
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;
using KastaunThenNewmanThenPalenzuela = tmpl::list<
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
    grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;

GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
     tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
     NewmanHamlinThenPalenzuelaEtAl),
    (true, false))

#undef INSTANTIATION

#define PHYSICALITY(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(_, data)                                                 \
  template bool grmhd::ValenciaDivClean::PrimitiveFromConservative<            \
      RECOVERY(data), ERROR_ON_FAILURE(data)>::                                \
      apply<PHYSICALITY(data)>(                                                \
          const gsl::not_null<Scalar<DataVector>*> rest_mass_density,          \
          const gsl::not_null<Scalar<DataVector>*> electron_fraction,          \
          const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,   \
          const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>        \
              spatial_velocity,                                                \
          const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>        \
              magnetic_field,                                                  \
          const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,  \
          const gsl::not_null<Scalar<DataVector>*> lorentz_factor,             \
          const gsl::not_null<Scalar<DataVector>*> pressure,                   \
          const gsl::not_null<Scalar<DataVector>*> temperature,                \
          const Scalar<DataVector>& tilde_d,                                   \
          const Scalar<DataVector>& tilde_ye,                                  \
          const Scalar<DataVector>& tilde_tau,                                 \
          const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,              \
          const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,              \
          const Scalar<DataVector>& tilde_phi,                                 \
          const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,      \
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,  \
          const Scalar<DataVector>& sqrt_det_spatial_metric,                   \
          const EquationsOfState::EquationOfState<true, 3>& equation_of_state, \
          const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&     \
              primitive_from_conservative_options);

GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (tmpl::list<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
     tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro>,
     tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
     tmpl::list<
         grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
     NewmanHamlinThenPalenzuelaEtAl, KastaunThenNewmanThenPalenzuela),
    (true, false), (true, false))

#undef INSTANTIATION
#undef THERMODIM
#undef PHYSICALITY
#undef RECOVERY
