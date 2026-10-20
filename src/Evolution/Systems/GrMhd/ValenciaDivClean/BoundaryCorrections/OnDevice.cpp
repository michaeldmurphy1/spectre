// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDevice.hpp"

#ifdef SPECTRE_KOKKOS
#include <algorithm>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
namespace {
// Tags for the arguments of the pointwise functions that aren't in a
// `Variables`
struct NormalCovector : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};
struct NormalDotMeshVelocity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct DgFormulation : db::SimpleTag {
  using type = dg::Formulation;
};
struct UseHydroSpeeds : db::SimpleTag {
  using type = bool;
};
struct LightSpeedDensityCutoff : db::SimpleTag {
  using type = double;
};
// The packaged data on the exterior side of the face, so that it has a
// different tag than the interior one
template <typename Tag>
struct Exterior : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = typename Tag::type;
};

using evolved_tags = tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
                                Tags::TildeS<Frame::Inertial>,
                                Tags::TildeB<Frame::Inertial>, Tags::TildePhi>;
using flux_tags = db::wrap_tags_in<::Tags::Flux, evolved_tags, tmpl::size_t<3>,
                                   Frame::Inertial>;
template <bool HasMeshVelocity>
using mesh_velocity_tags =
    tmpl::conditional_t<HasMeshVelocity, tmpl::list<NormalDotMeshVelocity>,
                        tmpl::list<>>;

// The normal dot mesh velocity at a grid point, or `nullptr` if the mesh isn't
// moving
template <typename... NormalDotMeshVelocity>
KOKKOS_FUNCTION const Scalar<double>* mesh_velocity_or_nullptr(
    const NormalDotMeshVelocity&... normal_dot_mesh_velocity) {
  static_assert(sizeof...(NormalDotMeshVelocity) <= 1);
  if constexpr (sizeof...(NormalDotMeshVelocity) == 1) {
    return &(normal_dot_mesh_velocity, ...);
  } else {
    return nullptr;
  }
}

// `Rusanov::dg_package_data_impl` at a grid point
template <bool HasMeshVelocity>
struct RusanovPackageData {
  using return_tags = Rusanov::dg_package_field_tags;
  using argument_tags =
      tmpl::append<evolved_tags, flux_tags,
                   tmpl::list<gr::Tags::Lapse<DataVector>,
                              gr::Tags::Shift<DataVector, 3>, NormalCovector>,
                   mesh_velocity_tags<HasMeshVelocity>>;

  template <typename... NormalDotMeshVelocity>
  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> packaged_tilde_d,
      const gsl::not_null<Scalar<double>*> packaged_tilde_ye,
      const gsl::not_null<Scalar<double>*> packaged_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*>
          packaged_tilde_s,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          packaged_tilde_b,
      const gsl::not_null<Scalar<double>*> packaged_tilde_phi,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_d,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_ye,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_s,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_b,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_phi,
      const gsl::not_null<Scalar<double>*> packaged_abs_char_speed,
      const Scalar<double>& tilde_d, const Scalar<double>& tilde_ye,
      const Scalar<double>& tilde_tau,
      const tnsr::i<double, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<double, 3, Frame::Inertial>& tilde_b,
      const Scalar<double>& tilde_phi,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_d,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_ye,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_tau,
      const tnsr::Ij<double, 3, Frame::Inertial>& flux_tilde_s,
      const tnsr::IJ<double, 3, Frame::Inertial>& flux_tilde_b,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_phi,
      const Scalar<double>& lapse,
      const tnsr::I<double, 3, Frame::Inertial>& shift,
      const tnsr::i<double, 3, Frame::Inertial>& normal_covector,
      const NormalDotMeshVelocity&... normal_dot_mesh_velocity) {
    Rusanov::dg_package_data_impl(
        packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
        packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
        packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
        packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
        packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
        packaged_abs_char_speed, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
        tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau, flux_tilde_s,
        flux_tilde_b, flux_tilde_phi, lapse, shift, normal_covector,
        mesh_velocity_or_nullptr(normal_dot_mesh_velocity...));
  }
};

// `Hll::dg_package_data_impl` at a grid point
template <bool HasMeshVelocity, typename DeviceView>
struct HllPackageData {
  using return_tags = Hll::dg_package_field_tags;
  using argument_tags = tmpl::append<
      evolved_tags, flux_tags,
      tmpl::list<
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
          hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
          hydro::Tags::RestMassDensity<DataVector>,
          hydro::Tags::ElectronFraction<DataVector>,
          hydro::Tags::Temperature<DataVector>,
          hydro::Tags::SpatialVelocity<DataVector, 3>, NormalCovector,
          EquationsOfState::Tags::DeviceView<DeviceView>, UseHydroSpeeds,
          LightSpeedDensityCutoff>,
      mesh_velocity_tags<HasMeshVelocity>>;

  template <typename... NormalDotMeshVelocity>
  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> packaged_tilde_d,
      const gsl::not_null<Scalar<double>*> packaged_tilde_ye,
      const gsl::not_null<Scalar<double>*> packaged_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*>
          packaged_tilde_s,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          packaged_tilde_b,
      const gsl::not_null<Scalar<double>*> packaged_tilde_phi,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_d,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_ye,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_s,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_b,
      const gsl::not_null<Scalar<double>*> packaged_normal_dot_flux_tilde_phi,
      const gsl::not_null<Scalar<double>*> packaged_largest_outgoing_char_speed,
      const gsl::not_null<Scalar<double>*> packaged_largest_ingoing_char_speed,
      const Scalar<double>& tilde_d, const Scalar<double>& tilde_ye,
      const Scalar<double>& tilde_tau,
      const tnsr::i<double, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<double, 3, Frame::Inertial>& tilde_b,
      const Scalar<double>& tilde_phi,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_d,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_ye,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_tau,
      const tnsr::Ij<double, 3, Frame::Inertial>& flux_tilde_s,
      const tnsr::IJ<double, 3, Frame::Inertial>& flux_tilde_b,
      const tnsr::I<double, 3, Frame::Inertial>& flux_tilde_phi,
      const Scalar<double>& lapse,
      const tnsr::I<double, 3, Frame::Inertial>& shift,
      const tnsr::i<double, 3, Frame::Inertial>& spatial_velocity_one_form,
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& electron_fraction,
      const Scalar<double>& temperature,
      const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,
      const tnsr::i<double, 3, Frame::Inertial>& normal_covector,
      const DeviceView& equation_of_state, const bool use_hydro_speeds,
      const double light_speed_density_cutoff,
      const NormalDotMeshVelocity&... normal_dot_mesh_velocity) {
    Hll::dg_package_data_impl(
        packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
        packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
        packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
        packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
        packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
        packaged_largest_outgoing_char_speed,
        packaged_largest_ingoing_char_speed, tilde_d, tilde_ye, tilde_tau,
        tilde_s, tilde_b, tilde_phi, flux_tilde_d, flux_tilde_ye,
        flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse,
        shift, spatial_velocity_one_form, rest_mass_density, electron_fraction,
        temperature, spatial_velocity, normal_covector,
        mesh_velocity_or_nullptr(normal_dot_mesh_velocity...),
        equation_of_state, use_hydro_speeds, light_speed_density_cutoff);
  }
};

// `Correction::dg_boundary_terms_impl` at a grid point. The arguments are the
// interior packaged data, the exterior packaged data and the DG formulation.
template <typename Correction>
struct BoundaryTerms {
  using return_tags = evolved_tags;
  using argument_tags = tmpl::append<
      typename Correction::dg_package_field_tags,
      db::wrap_tags_in<Exterior, typename Correction::dg_package_field_tags>,
      tmpl::list<DgFormulation>>;

  template <typename... Args>
  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> boundary_correction_tilde_d,
      const gsl::not_null<Scalar<double>*> boundary_correction_tilde_ye,
      const gsl::not_null<Scalar<double>*> boundary_correction_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3, Frame::Inertial>*>
          boundary_correction_tilde_s,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          boundary_correction_tilde_b,
      const gsl::not_null<Scalar<double>*> boundary_correction_tilde_phi,
      const Args&... args) {
    Correction::dg_boundary_terms_impl(
        boundary_correction_tilde_d, boundary_correction_tilde_ye,
        boundary_correction_tilde_tau, boundary_correction_tilde_s,
        boundary_correction_tilde_b, boundary_correction_tilde_phi, args...);
  }
};

// Calls `copy_and_apply_pointwise_on_device` with the tags of `Function`
template <typename Function, typename Results, typename... Args>
void apply_on_device(const Kokkos::DefaultExecutionSpace& exec,
                     const Results& results, const Args&... args) {
  copy_and_apply_pointwise_on_device<Function>(
      exec, results, typename Function::return_tags{},
      typename Function::argument_tags{}, args...);
}
}  // namespace

OnDevice<Rusanov>::OnDevice(Kokkos::DefaultExecutionSpace exec,
                            const Rusanov& /*rusanov*/)
    : exec_(std::move(exec)) {}

OnDevice<Rusanov>::OnDevice(const Rusanov& rusanov)
    : OnDevice(Kokkos::DefaultExecutionSpace{}, rusanov) {}

double OnDevice<Rusanov>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3,
                  Frame::Inertial>& /*spatial_velocity_one_form*/,

    const Scalar<DataVector>& /*rest_mass_density*/,
    const Scalar<DataVector>& /*electron_fraction*/,
    const Scalar<DataVector>& /*temperature*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*spatial_velocity*/,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>&
    /*equation_of_state*/) const {
  const auto results = std::tuple{packaged_tilde_d,
                                  packaged_tilde_ye,
                                  packaged_tilde_tau,
                                  packaged_tilde_s,
                                  packaged_tilde_b,
                                  packaged_tilde_phi,
                                  packaged_normal_dot_flux_tilde_d,
                                  packaged_normal_dot_flux_tilde_ye,
                                  packaged_normal_dot_flux_tilde_tau,
                                  packaged_normal_dot_flux_tilde_s,
                                  packaged_normal_dot_flux_tilde_b,
                                  packaged_normal_dot_flux_tilde_phi,
                                  packaged_abs_char_speed};
  if (normal_dot_mesh_velocity.has_value()) {
    apply_on_device<RusanovPackageData<true>>(
        exec_, results, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
        tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau, flux_tilde_s,
        flux_tilde_b, flux_tilde_phi, lapse, shift, normal_covector,
        normal_dot_mesh_velocity.value());
  } else {
    apply_on_device<RusanovPackageData<false>>(
        exec_, results, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
        tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau, flux_tilde_s,
        flux_tilde_b, flux_tilde_phi, lapse, shift, normal_covector);
  }
  return max(get(*packaged_abs_char_speed));
}

void OnDevice<Rusanov>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& abs_char_speed_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_ye_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    const dg::Formulation dg_formulation) const {
  apply_on_device<BoundaryTerms<Rusanov>>(
      exec_,
      std::tuple{boundary_correction_tilde_d, boundary_correction_tilde_ye,
                 boundary_correction_tilde_tau, boundary_correction_tilde_s,
                 boundary_correction_tilde_b, boundary_correction_tilde_phi},
      tilde_d_int, tilde_ye_int, tilde_tau_int, tilde_s_int, tilde_b_int,
      tilde_phi_int, normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_ye_int,
      normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_s_int,
      normal_dot_flux_tilde_b_int, normal_dot_flux_tilde_phi_int,
      abs_char_speed_int, tilde_d_ext, tilde_ye_ext, tilde_tau_ext, tilde_s_ext,
      tilde_b_ext, tilde_phi_ext, normal_dot_flux_tilde_d_ext,
      normal_dot_flux_tilde_ye_ext, normal_dot_flux_tilde_tau_ext,
      normal_dot_flux_tilde_s_ext, normal_dot_flux_tilde_b_ext,
      normal_dot_flux_tilde_phi_ext, abs_char_speed_ext, dg_formulation);
}

OnDevice<Hll>::OnDevice(Kokkos::DefaultExecutionSpace exec, Hll hll)
    : hll_(std::move(hll)), exec_(std::move(exec)) {}

OnDevice<Hll>::OnDevice(Hll hll)
    : OnDevice(Kokkos::DefaultExecutionSpace{}, std::move(hll)) {}

double OnDevice<Hll>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_ingoing_char_speed,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  const auto results = std::tuple{packaged_tilde_d,
                                  packaged_tilde_ye,
                                  packaged_tilde_tau,
                                  packaged_tilde_s,
                                  packaged_tilde_b,
                                  packaged_tilde_phi,
                                  packaged_normal_dot_flux_tilde_d,
                                  packaged_normal_dot_flux_tilde_ye,
                                  packaged_normal_dot_flux_tilde_tau,
                                  packaged_normal_dot_flux_tilde_s,
                                  packaged_normal_dot_flux_tilde_b,
                                  packaged_normal_dot_flux_tilde_phi,
                                  packaged_largest_outgoing_char_speed,
                                  packaged_largest_ingoing_char_speed};
  // This needs the data on the whole face, so it is computed on the host
  const bool use_hydro_speeds =
      hll_.use_hydro_speeds(rest_mass_density, tilde_b);
  EquationsOfState::visit_device_view(
      equation_of_state, [&](const auto& device_view) {
        using DeviceView = std::decay_t<decltype(device_view)>;
        if (normal_dot_mesh_velocity.has_value()) {
          apply_on_device<HllPackageData<true, DeviceView>>(
              exec_, results, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
              tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau,
              flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
              spatial_velocity_one_form, rest_mass_density, electron_fraction,
              temperature, spatial_velocity, normal_covector, device_view,
              use_hydro_speeds, hll_.light_speed_density_cutoff(),
              normal_dot_mesh_velocity.value());
        } else {
          apply_on_device<HllPackageData<false, DeviceView>>(
              exec_, results, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
              tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau,
              flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
              spatial_velocity_one_form, rest_mass_density, electron_fraction,
              temperature, spatial_velocity, normal_covector, device_view,
              use_hydro_speeds, hll_.light_speed_density_cutoff());
        }
      });

  return std::max(max(abs(get(*packaged_largest_outgoing_char_speed))),
                  max(abs(get(*packaged_largest_ingoing_char_speed))));
}

void OnDevice<Hll>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& largest_outgoing_char_speed_int,
    const Scalar<DataVector>& largest_ingoing_char_speed_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_ye_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& largest_outgoing_char_speed_ext,
    const Scalar<DataVector>& largest_ingoing_char_speed_ext,
    const dg::Formulation dg_formulation) const {
  apply_on_device<BoundaryTerms<Hll>>(
      exec_,
      std::tuple{boundary_correction_tilde_d, boundary_correction_tilde_ye,
                 boundary_correction_tilde_tau, boundary_correction_tilde_s,
                 boundary_correction_tilde_b, boundary_correction_tilde_phi},
      tilde_d_int, tilde_ye_int, tilde_tau_int, tilde_s_int, tilde_b_int,
      tilde_phi_int, normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_ye_int,
      normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_s_int,
      normal_dot_flux_tilde_b_int, normal_dot_flux_tilde_phi_int,
      largest_outgoing_char_speed_int, largest_ingoing_char_speed_int,
      tilde_d_ext, tilde_ye_ext, tilde_tau_ext, tilde_s_ext, tilde_b_ext,
      tilde_phi_ext, normal_dot_flux_tilde_d_ext, normal_dot_flux_tilde_ye_ext,
      normal_dot_flux_tilde_tau_ext, normal_dot_flux_tilde_s_ext,
      normal_dot_flux_tilde_b_ext, normal_dot_flux_tilde_phi_ext,
      largest_outgoing_char_speed_ext, largest_ingoing_char_speed_ext,
      dg_formulation);
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
#endif  // SPECTRE_KOKKOS
