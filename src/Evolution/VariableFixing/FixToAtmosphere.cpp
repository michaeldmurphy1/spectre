// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/FixToAtmosphere.hpp"

#include <limits>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.tpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

// The `DeviceView` functions are `KOKKOS_FUNCTION` templates, and here they are
// instantiated with the EOS on the host, which can't be used on the device. So
// this file is only compiled in the host pass of GPU builds.
#ifndef SPECTRE_KOKKOS_DEVICE_PASS
namespace VariableFixing {
std::ostream& operator<<(std::ostream& os,
                         const FixReconstructedStateToAtmosphere& t) {
  switch (t) {
    case FixReconstructedStateToAtmosphere::Always:
      return os << "Always";
    case FixReconstructedStateToAtmosphere::AtDgFdInterfaceOnly:
      return os << "AtDgFdInterfaceOnly";
    case FixReconstructedStateToAtmosphere::OnFdOnly:
      return os << "OnFdOnly";
    case FixReconstructedStateToAtmosphere::Never:
      return os << "Never";
    default:
      ERROR("Unknown floating point type, must be Float or Double");
  }
}
}  // namespace VariableFixing

template <>
VariableFixing::FixReconstructedStateToAtmosphere
Options::create_from_yaml<VariableFixing::FixReconstructedStateToAtmosphere>::
    create<void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto t :
       {VariableFixing::FixReconstructedStateToAtmosphere::Always,
        VariableFixing::FixReconstructedStateToAtmosphere::AtDgFdInterfaceOnly,
        VariableFixing::FixReconstructedStateToAtmosphere::OnFdOnly,
        VariableFixing::FixReconstructedStateToAtmosphere::Never}) {
    if (type_read == get_output(t)) {
      return t;
    }
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to FixReconstructedStateToAtmosphere.");
}

namespace VariableFixing {
template <size_t Dim>
FixToAtmosphere<Dim>::FixToAtmosphere(
    const double density_of_atmosphere, const double density_cutoff,
    const std::optional<VelocityLimitingOptions> velocity_limiting,
    const std::optional<KappaLimitingOptions> kappa_limiting,
    const Options::Context& context)
    : density_of_atmosphere_(density_of_atmosphere),
      density_cutoff_(density_cutoff),
      velocity_limiting_(velocity_limiting),
      kappa_limiting_(kappa_limiting) {
  if (density_of_atmosphere_ > density_cutoff_) {
    PARSE_ERROR(context, "The cutoff density ("
                             << density_cutoff_
                             << ") must be greater than or equal to the "
                                "density value in the atmosphere ("
                             << density_of_atmosphere_ << ')');
  }

  if (velocity_limiting_.has_value()) {
    if (velocity_limiting_->atmosphere_max_velocity < 0.0) {
      PARSE_ERROR(context,
                  "The AtmosphereMaxVelocity must be non-negative but is "
                      << velocity_limiting_->atmosphere_max_velocity);
    }
    if (velocity_limiting_->near_atmosphere_max_velocity < 0.0) {
      PARSE_ERROR(context,
                  "The NearAtmosphereMaxVelocity must be non-negative but is "
                      << velocity_limiting_->near_atmosphere_max_velocity);
    }
    if (velocity_limiting_->atmosphere_max_velocity >
        velocity_limiting_->near_atmosphere_max_velocity) {
      PARSE_ERROR(context,
                  "The AtmosphereMaxVelocity ("
                      << velocity_limiting_->atmosphere_max_velocity
                      << ") must be smaller NearAtmosphereMaxVelocity ("
                      << velocity_limiting_->near_atmosphere_max_velocity
                      << ").");
    }
    if (velocity_limiting_->atmosphere_density_cutoff < 0.0) {
      PARSE_ERROR(context,
                  "The AtmosphereDensityCutoff must be non-negative but is "
                      << velocity_limiting_->atmosphere_density_cutoff);
    }
    if (velocity_limiting_->transition_density_bound < 0.0) {
      PARSE_ERROR(context,
                  "The TransitionDensityBound must be non-negative but is "
                      << velocity_limiting_->transition_density_bound);
    }
    if (velocity_limiting_->atmosphere_density_cutoff <
        density_of_atmosphere_) {
      PARSE_ERROR(
          context,
          "The AtmosphereDensityCutoff ("
              << velocity_limiting_->atmosphere_density_cutoff
              << ") must be greater than or equal to the DensityOfAtmosphere ("
              << density_of_atmosphere_ << ").");
    }
    if (velocity_limiting_->transition_density_bound <
        velocity_limiting_->atmosphere_density_cutoff) {
      PARSE_ERROR(context, "The TransitionDensityBound ("
                               << velocity_limiting_->transition_density_bound
                               << ") must be greater than or equal to the "
                                  "AtmosphereDensityCutoff ("
                               << velocity_limiting_->atmosphere_density_cutoff
                               << ").");
    }
  }

  if (kappa_limiting_.has_value()) {
    if (kappa_limiting_->density_lower_bound >
        kappa_limiting_->density_upper_bound) {
      PARSE_ERROR(context,
                  "The DensityLowerBound ("
                      << kappa_limiting_->density_lower_bound
                      << ") must be less than or equal to DensityUpperBound("
                      << kappa_limiting_->density_upper_bound << ").");
    }
    if (kappa_limiting_->min_temperature.has_value() and
        kappa_limiting_->min_temperature.value() < 0.0) {
      PARSE_ERROR(context, "The MinTemperature must be non-negative but is "
                               << kappa_limiting_->min_temperature.value());
    }
  }
}

template <size_t Dim>
// NOLINTNEXTLINE(google-runtime-references)
void FixToAtmosphere<Dim>::pup(PUP::er& p) {
  p | density_of_atmosphere_;
  p | density_cutoff_;
  p | velocity_limiting_;
  p | kappa_limiting_;
}

template <size_t Dim>
typename FixToAtmosphere<Dim>::DeviceView FixToAtmosphere<Dim>::device_view()
    const {
  DeviceView view{};
  view.density_of_atmosphere = density_of_atmosphere_;
  view.density_cutoff = density_cutoff_;
  view.limit_velocity = velocity_limiting_.has_value();
  if (velocity_limiting_.has_value()) {
    view.velocity_limiting = *velocity_limiting_;
  }
  view.limit_kappa = kappa_limiting_.has_value();
  if (kappa_limiting_.has_value()) {
    view.kappa_limiting.density_lower_bound =
        kappa_limiting_->density_lower_bound;
    view.kappa_limiting.eplison_kappa_minus =
        kappa_limiting_->eplison_kappa_minus;
    view.kappa_limiting.density_upper_bound =
        kappa_limiting_->density_upper_bound;
    view.kappa_limiting.epsilon_kappa_max = kappa_limiting_->epsilon_kappa_max;
    view.kappa_limiting.min_temperature_from_eos =
        not kappa_limiting_->min_temperature.has_value();
    if (kappa_limiting_->min_temperature.has_value()) {
      view.kappa_limiting.min_temperature = *kappa_limiting_->min_temperature;
    }
    view.kappa_limiting.limit_above_density_upper_bound =
        kappa_limiting_->limit_above_density_upper_bound;
  }
  return view;
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void FixToAtmosphere<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const Scalar<DataVector>& electron_fraction,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const {
  const DeviceView view = device_view();
  for (size_t i = 0; i < rest_mass_density->get().size(); i++) {
    auto spatial_velocity_at_point = extract_point(*spatial_velocity, i);
    view.template fix_at_point<ThermodynamicDim>(
        make_not_null(&get(*rest_mass_density)[i]),
        make_not_null(&get(*specific_internal_energy)[i]),
        make_not_null(&spatial_velocity_at_point),
        make_not_null(&get(*lorentz_factor)[i]),
        make_not_null(&get(*pressure)[i]), make_not_null(&get(*temperature)[i]),
        get(electron_fraction)[i], extract_point(spatial_metric, i),
        equation_of_state, nullptr);
    overwrite_point(spatial_velocity, spatial_velocity_at_point, i);
  }
}

template <size_t Dim>
bool operator==(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return lhs.density_of_atmosphere_ == rhs.density_of_atmosphere_ and
         lhs.density_cutoff_ == rhs.density_cutoff_ and
         lhs.velocity_limiting_ == rhs.velocity_limiting_ and
         lhs.kappa_limiting_ == rhs.kappa_limiting_;
}

template <size_t Dim>
bool operator!=(const FixToAtmosphere<Dim>& lhs,
                const FixToAtmosphere<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
void FixToAtmosphere<Dim>::VelocityLimitingOptions::pup(PUP::er& p) {
  p | atmosphere_max_velocity;
  p | near_atmosphere_max_velocity;
  p | atmosphere_density_cutoff;
  p | transition_density_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::VelocityLimitingOptions::operator==(
    const VelocityLimitingOptions& rhs) const {
  return atmosphere_max_velocity == rhs.atmosphere_max_velocity and
         near_atmosphere_max_velocity == rhs.near_atmosphere_max_velocity and
         atmosphere_density_cutoff == rhs.atmosphere_density_cutoff and
         transition_density_bound == rhs.transition_density_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::VelocityLimitingOptions::operator!=(
    const VelocityLimitingOptions& rhs) const {
  return not(*this == rhs);
}

template <size_t Dim>
void FixToAtmosphere<Dim>::KappaLimitingOptions::pup(PUP::er& p) {
  p | density_lower_bound;
  p | eplison_kappa_minus;
  p | density_upper_bound;
  p | epsilon_kappa_max;
  p | min_temperature;
  p | limit_above_density_upper_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::KappaLimitingOptions::operator==(
    const KappaLimitingOptions& rhs) const {
  return density_lower_bound == rhs.density_lower_bound and
         eplison_kappa_minus == rhs.eplison_kappa_minus and
         density_upper_bound == rhs.density_upper_bound and
         epsilon_kappa_max == rhs.epsilon_kappa_max and
         min_temperature == rhs.min_temperature and
         limit_above_density_upper_bound == rhs.limit_above_density_upper_bound;
}

template <size_t Dim>
bool FixToAtmosphere<Dim>::KappaLimitingOptions::operator!=(
    const KappaLimitingOptions& rhs) const {
  return not(*this == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                     \
  template class FixToAtmosphere<DIM(data)>;                       \
  template bool operator==(const FixToAtmosphere<DIM(data)>& lhs,  \
                           const FixToAtmosphere<DIM(data)>& rhs); \
  template bool operator!=(const FixToAtmosphere<DIM(data)>& lhs,  \
                           const FixToAtmosphere<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define INSTANTIATION(r, data)                                                \
  template void FixToAtmosphere<DIM(data)>::operator()(                       \
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,             \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,      \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>   \
          spatial_velocity,                                                   \
      const gsl::not_null<Scalar<DataVector>*> lorentz_factor,                \
      const gsl::not_null<Scalar<DataVector>*> pressure,                      \
      const gsl::not_null<Scalar<DataVector>*> temperature,                   \
      const Scalar<DataVector>& electron_fraction,                            \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric, \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&        \
          equation_of_state) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2, 3))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
#endif  // SPECTRE_KOKKOS_DEVICE_PASS
