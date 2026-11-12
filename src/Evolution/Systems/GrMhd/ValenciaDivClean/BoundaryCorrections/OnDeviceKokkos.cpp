// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDeviceKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <boost/preprocessor/tuple/elem.hpp>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceVariablesAtGridPointKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.tpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
namespace {
// Tags for the arguments of the pointwise functions that aren't in a
// `Variables`
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
  using argument_tags = tmpl::append<
      evolved_tags, flux_tags,
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 evolution::dg::Tags::NormalCovector<3>>,
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
          hydro::Tags::SpatialVelocity<DataVector, 3>,
          evolution::dg::Tags::NormalCovector<3>,
          EquationsOfState::Tags::DeviceView<DeviceView>,
          ::Tags::PerElement<UseHydroSpeeds>, LightSpeedDensityCutoff>,
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

// Arguments of the fused pointwise functions that aren't face variables
struct Conormal : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};
struct MeshVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};
struct Sign : db::SimpleTag {
  using type = double;
};
template <bool HasMeshVelocity>
using fused_mesh_velocity_tags =
    tmpl::conditional_t<HasMeshVelocity, tmpl::list<MeshVelocity>,
                        tmpl::list<>>;

// The fluxes (with the mesh velocity term), the normal covector and the
// normal dot the mesh velocity at a grid point of a face
struct FaceFluxesAndNormal {
  tnsr::I<double, 3, Frame::Inertial> tilde_d_flux{};
  tnsr::I<double, 3, Frame::Inertial> tilde_ye_flux{};
  tnsr::I<double, 3, Frame::Inertial> tilde_tau_flux{};
  tnsr::Ij<double, 3, Frame::Inertial> tilde_s_flux{};
  tnsr::IJ<double, 3, Frame::Inertial> tilde_b_flux{};
  tnsr::I<double, 3, Frame::Inertial> tilde_phi_flux{};
  tnsr::i<double, 3, Frame::Inertial> normal_covector{};
  Scalar<double> normal_dot_mesh_velocity{};
};

// Computes the `FaceFluxesAndNormal` with the same operations as the separate
// kernels of `subcell::TimeDerivativeOnDevice` did before they were fused
template <typename... MeshVelocityType>
KOKKOS_FUNCTION FaceFluxesAndNormal face_fluxes_and_normal(
    const Scalar<double>& tilde_d, const Scalar<double>& tilde_ye,
    const Scalar<double>& tilde_tau,
    const tnsr::i<double, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<double, 3, Frame::Inertial>& tilde_b,
    const Scalar<double>& tilde_phi, const Scalar<double>& lapse,
    const tnsr::I<double, 3, Frame::Inertial>& shift,
    const Scalar<double>& sqrt_det_spatial_metric,
    const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<double, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<double>& pressure,
    const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<double>& lorentz_factor,
    const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
    const tnsr::i<double, 3, Frame::Inertial>& conormal, const double sign,
    const MeshVelocityType&... mesh_velocity) {
  static_assert(sizeof...(MeshVelocityType) <= 1);
  FaceFluxesAndNormal result{};
  ComputeFluxes::apply(
      make_not_null(&result.tilde_d_flux), make_not_null(&result.tilde_ye_flux),
      make_not_null(&result.tilde_tau_flux),
      make_not_null(&result.tilde_s_flux), make_not_null(&result.tilde_b_flux),
      make_not_null(&result.tilde_phi_flux), tilde_d, tilde_ye, tilde_tau,
      tilde_s, tilde_b, tilde_phi, lapse, shift, sqrt_det_spatial_metric,
      spatial_metric, inv_spatial_metric, pressure, spatial_velocity,
      lorentz_factor, magnetic_field);
  if constexpr (sizeof...(MeshVelocityType) == 1) {
    const tnsr::I<double, 3, Frame::Inertial>& grid_velocity =
        (mesh_velocity, ...);
    for (size_t j = 0; j < 3; ++j) {
      const double v = grid_velocity.get(j);
      result.tilde_d_flux.get(j) =
          result.tilde_d_flux.get(j) - v * get(tilde_d);
      result.tilde_ye_flux.get(j) =
          result.tilde_ye_flux.get(j) - v * get(tilde_ye);
      result.tilde_tau_flux.get(j) =
          result.tilde_tau_flux.get(j) - v * get(tilde_tau);
      result.tilde_phi_flux.get(j) =
          result.tilde_phi_flux.get(j) - v * get(tilde_phi);
      for (size_t i = 0; i < 3; ++i) {
        result.tilde_s_flux.get(j, i) =
            result.tilde_s_flux.get(j, i) - v * tilde_s.get(i);
        result.tilde_b_flux.get(j, i) =
            result.tilde_b_flux.get(j, i) - v * tilde_b.get(i);
      }
    }
  }
  double magnitude_squared = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      magnitude_squared +=
          conormal.get(i) * conormal.get(j) * inv_spatial_metric.get(i, j);
    }
  }
  const double magnitude = sqrt(magnitude_squared);
  for (size_t i = 0; i < 3; ++i) {
    result.normal_covector.get(i) = sign * conormal.get(i) / magnitude;
  }
  if constexpr (sizeof...(MeshVelocityType) == 1) {
    const tnsr::I<double, 3, Frame::Inertial>& grid_velocity =
        (mesh_velocity, ...);
    get(result.normal_dot_mesh_velocity) = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      get(result.normal_dot_mesh_velocity) +=
          result.normal_covector.get(i) * grid_velocity.get(i);
    }
  }
  return result;
}

// `face_fluxes_and_normal` and `Rusanov::dg_package_data_impl` at a grid point
template <bool HasMeshVelocity>
struct FusedRusanovPackageData {
  using return_tags = Rusanov::dg_package_field_tags;
  using argument_tags =
      tmpl::append<ComputeFluxes::argument_tags, tmpl::list<Conormal, Sign>,
                   fused_mesh_velocity_tags<HasMeshVelocity>>;

  template <typename... MeshVelocityType>
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
      const Scalar<double>& tilde_phi, const Scalar<double>& lapse,
      const tnsr::I<double, 3, Frame::Inertial>& shift,
      const Scalar<double>& sqrt_det_spatial_metric,
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<double, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<double>& pressure,
      const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<double>& lorentz_factor,
      const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
      const tnsr::i<double, 3, Frame::Inertial>& conormal, const double sign,
      const MeshVelocityType&... mesh_velocity) {
    const FaceFluxesAndNormal face = face_fluxes_and_normal(
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
        sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric, pressure,
        spatial_velocity, lorentz_factor, magnetic_field, conormal, sign,
        mesh_velocity...);
    Rusanov::dg_package_data_impl(
        packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
        packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
        packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
        packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
        packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
        packaged_abs_char_speed, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
        tilde_phi, face.tilde_d_flux, face.tilde_ye_flux, face.tilde_tau_flux,
        face.tilde_s_flux, face.tilde_b_flux, face.tilde_phi_flux, lapse, shift,
        face.normal_covector,
        HasMeshVelocity ? &face.normal_dot_mesh_velocity : nullptr);
  }
};

// `face_fluxes_and_normal` and `Hll::dg_package_data_impl` at a grid point
template <bool HasMeshVelocity, typename DeviceView>
struct FusedHllPackageData {
  using return_tags = Hll::dg_package_field_tags;
  using argument_tags = tmpl::append<
      ComputeFluxes::argument_tags,
      tmpl::list<
          hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
          hydro::Tags::RestMassDensity<DataVector>,
          hydro::Tags::ElectronFraction<DataVector>,
          hydro::Tags::Temperature<DataVector>, Conormal, Sign,
          EquationsOfState::Tags::DeviceView<DeviceView>,
          ::Tags::PerElement<UseHydroSpeeds>, LightSpeedDensityCutoff>,
      fused_mesh_velocity_tags<HasMeshVelocity>>;

  template <typename... MeshVelocityType>
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
      const Scalar<double>& tilde_phi, const Scalar<double>& lapse,
      const tnsr::I<double, 3, Frame::Inertial>& shift,
      const Scalar<double>& sqrt_det_spatial_metric,
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<double, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<double>& pressure,
      const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<double>& lorentz_factor,
      const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
      const tnsr::i<double, 3, Frame::Inertial>& spatial_velocity_one_form,
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& electron_fraction,
      const Scalar<double>& temperature,
      const tnsr::i<double, 3, Frame::Inertial>& conormal, const double sign,
      const DeviceView& equation_of_state, const bool use_hydro_speeds,
      const double light_speed_density_cutoff,
      const MeshVelocityType&... mesh_velocity) {
    const FaceFluxesAndNormal face = face_fluxes_and_normal(
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
        sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric, pressure,
        spatial_velocity, lorentz_factor, magnetic_field, conormal, sign,
        mesh_velocity...);
    Hll::dg_package_data_impl(
        packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
        packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
        packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
        packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
        packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
        packaged_largest_outgoing_char_speed,
        packaged_largest_ingoing_char_speed, tilde_d, tilde_ye, tilde_tau,
        tilde_s, tilde_b, tilde_phi, face.tilde_d_flux, face.tilde_ye_flux,
        face.tilde_tau_flux, face.tilde_s_flux, face.tilde_b_flux,
        face.tilde_phi_flux, lapse, shift, spatial_velocity_one_form,
        rest_mass_density, electron_fraction, temperature, spatial_velocity,
        face.normal_covector,
        HasMeshVelocity ? &face.normal_dot_mesh_velocity : nullptr,
        equation_of_state, use_hydro_speeds, light_speed_density_cutoff);
  }
};

// The variables on the face (`fd::FaceVariablesAtGridPoint`),
// `face_fluxes_and_normal` and `Rusanov::dg_package_data_impl` at a grid point.
// The arguments after the equation of state are the fix to the atmosphere (if
// `FixAtmosphere`) and the mesh velocity (if `HasMeshVelocity`).
template <typename DeviceView, bool FixAtmosphere, bool HasMeshVelocity>
struct RusanovPackageDataFromReconstruction {
  using return_tags = Rusanov::dg_package_field_tags;
  using argument_tags = tmpl::append<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>, Conormal, Sign,
                 EquationsOfState::Tags::DeviceView<DeviceView>>,
      tmpl::conditional_t<FixAtmosphere, tmpl::list<fd::FixToAtmosphereView>,
                          tmpl::list<>>,
      fused_mesh_velocity_tags<HasMeshVelocity>>;

  template <typename... Extra>
  KOKKOS_FUNCTION static PointwiseStatus apply(
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
      const Scalar<double>& reconstructed_rest_mass_density,
      const Scalar<double>& reconstructed_temperature,
      const Scalar<double>& electron_fraction,
      const tnsr::I<double, 3, Frame::Inertial>&
          lorentz_factor_times_spatial_velocity,
      const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
      const Scalar<double>& divergence_cleaning_field,
      const Scalar<double>& lapse,
      const tnsr::I<double, 3, Frame::Inertial>& shift,
      const tnsr::ii<double, 3, Frame::Inertial>& spatial_metric,
      const Scalar<double>& sqrt_det_spatial_metric,
      const tnsr::II<double, 3, Frame::Inertial>& inv_spatial_metric,
      const tnsr::i<double, 3, Frame::Inertial>& conormal, const double sign,
      const DeviceView& equation_of_state, const Extra&... extra) {
    static_assert(sizeof...(Extra) == static_cast<size_t>(FixAtmosphere) +
                                          static_cast<size_t>(HasMeshVelocity));
    const auto extras = std::forward_as_tuple(extra...);
    // 1. The variables on the face
    Scalar<double> rest_mass_density{};
    Scalar<double> specific_internal_energy{};
    tnsr::I<double, 3, Frame::Inertial> spatial_velocity{};
    Scalar<double> lorentz_factor{};
    Scalar<double> pressure{};
    Scalar<double> temperature{};
    tnsr::i<double, 3, Frame::Inertial> spatial_velocity_one_form{};
    Scalar<double> tilde_d{};
    Scalar<double> tilde_ye{};
    Scalar<double> tilde_tau{};
    tnsr::i<double, 3, Frame::Inertial> tilde_s{};
    tnsr::I<double, 3, Frame::Inertial> tilde_b{};
    Scalar<double> tilde_phi{};
    // Only kept if the fix to the atmosphere fails, which is an error
    const Scalar<double> no_previous_value{0.0};
    const auto face_variables = [&](const auto&... fix_to_atmosphere) {
      return fd::FaceVariablesAtGridPoint<DeviceView, FixAtmosphere>::apply(
          make_not_null(&rest_mass_density),
          make_not_null(&specific_internal_energy),
          make_not_null(&spatial_velocity), make_not_null(&lorentz_factor),
          make_not_null(&pressure), make_not_null(&temperature),
          make_not_null(&spatial_velocity_one_form), make_not_null(&tilde_d),
          make_not_null(&tilde_ye), make_not_null(&tilde_tau),
          make_not_null(&tilde_s), make_not_null(&tilde_b),
          make_not_null(&tilde_phi), reconstructed_rest_mass_density,
          no_previous_value, no_previous_value, reconstructed_temperature,
          electron_fraction, lorentz_factor_times_spatial_velocity,
          magnetic_field, divergence_cleaning_field, spatial_metric,
          sqrt_det_spatial_metric, equation_of_state, fix_to_atmosphere...);
    };
    PointwiseStatus status{};
    if constexpr (FixAtmosphere) {
      status = face_variables(std::get<0>(extras));
    } else {
      status = face_variables();
    }
    if (status.failed) {
      return status;
    }

    // 2. The fluxes and the normal
    const auto fluxes_and_normal = [&](const auto&... mesh_velocity) {
      return face_fluxes_and_normal(
          tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
          shift, sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
          pressure, spatial_velocity, lorentz_factor, magnetic_field, conormal,
          sign, mesh_velocity...);
    };
    FaceFluxesAndNormal face{};
    if constexpr (HasMeshVelocity) {
      face = fluxes_and_normal(
          std::get<static_cast<size_t>(FixAtmosphere)>(extras));
    } else {
      face = fluxes_and_normal();
    }

    // 3. The packaged data
    Rusanov::dg_package_data_impl(
        packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
        packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
        packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
        packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
        packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
        packaged_abs_char_speed, tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
        tilde_phi, face.tilde_d_flux, face.tilde_ye_flux, face.tilde_tau_flux,
        face.tilde_s_flux, face.tilde_b_flux, face.tilde_phi_flux, lapse, shift,
        face.normal_covector,
        HasMeshVelocity ? &face.normal_dot_mesh_velocity : nullptr);
    return status;
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

// Whether `Hll` uses the hydro characteristic speeds on each face set, as
// `Hll::use_hydro_speeds` decides for a face set
template <typename FaceTags>
Kokkos::View<const bool*> use_hydro_speeds_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const BatchedVariables<FaceTags>& faces, const Hll& hll) {
  const auto& rest_mass_density = get(
      get<::Tags::BatchView<hydro::Tags::RestMassDensity<DataVector>>>(faces));
  const auto& tilde_b = get<::Tags::BatchView<Tags::TildeB<>>>(faces);
  const auto tilde_b_x = get<0>(tilde_b);
  const auto tilde_b_y = get<1>(tilde_b);
  const auto tilde_b_z = get<2>(tilde_b);
  const size_t number_of_grid_points = faces.number_of_grid_points();
  const size_t number_of_elements = faces.number_of_elements();
  const double magnetic_field_cutoff = hll.magnetic_field_magnitude_for_hydro();
  const double density_cutoff = hll.light_speed_density_cutoff();
  const Kokkos::View<bool*> result{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "UseHydroSpeeds"),
      number_of_elements};
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  Kokkos::parallel_for(
      "UseHydroSpeeds",
      TeamPolicy(exec, static_cast<int>(number_of_elements), Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const auto e = static_cast<size_t>(team.league_rank());
        double max_magnitude_tilde_b = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, number_of_grid_points),
            [&](const size_t i, double& local_max) {
              local_max = Kokkos::max(local_max, sqrt(square(tilde_b_x(i, e)) +
                                                      square(tilde_b_y(i, e)) +
                                                      square(tilde_b_z(i, e))));
            },
            Kokkos::Max<double>(max_magnitude_tilde_b));
        double max_rest_mass_density = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, number_of_grid_points),
            [&](const size_t i, double& local_max) {
              local_max = Kokkos::max(local_max, rest_mass_density(i, e));
            },
            Kokkos::Max<double>(max_rest_mass_density));
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          result(e) = not(max_magnitude_tilde_b > magnetic_field_cutoff) and
                      max_rest_mass_density > density_cutoff;
        });
      });
  return result;
}

// Applies `Function` to the `faces`, with the arguments that aren't face
// variables given by `special_argument`. If `status` isn't `nullptr`, the
// `PointwiseStatus` of `Function` is recorded in that
// `DeferredPointwiseStatus` instead of being returned.
template <typename Function, typename ResultTags, typename FaceTags,
          typename SpecialArgument, typename StatusPointer = std::nullptr_t>
auto apply_to_faces(const Kokkos::DefaultExecutionSpace& exec,
                    const gsl::not_null<BatchedVariables<ResultTags>*> results,
                    const BatchedVariables<FaceTags>& faces,
                    const SpecialArgument& special_argument,
                    const StatusPointer status = nullptr) {
  // The face variables are returned by reference and the other arguments,
  // which are cheap to copy, by value
  const auto argument =
      [&]<typename Tag>(tmpl::type_<Tag> tag) -> decltype(auto) {
    if constexpr (tmpl::list_contains_v<FaceTags, Tag>) {
      (void)tag;
      return get<::Tags::BatchView<Tag>>(faces);
    } else {
      return special_argument(tag);
    }
  };
  const auto args =
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, argument(tmpl::type_<ArgumentTags>{})...);
      }(typename Function::argument_tags{});
  if constexpr (std::is_null_pointer_v<StatusPointer>) {
    (void)status;
    return apply_pointwise_on_device<Function>(
        exec, results, typename Function::return_tags{},
        typename Function::argument_tags{}, args);
  } else {
    apply_pointwise_on_device<Function>(
        exec, results, make_not_null(status), typename Function::return_tags{},
        typename Function::argument_tags{}, args);
  }
}
}  // namespace

template <typename Correction>
void package_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<
        BatchedVariables<typename Correction::dg_package_field_tags>*>
        packaged_data,
    const BatchedVariables<fd::face_tags>& faces, const Correction& correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const NormalDotMeshVelocityBatch* const normal_dot_mesh_velocity) {
  if constexpr (std::is_same_v<Correction, Rusanov>) {
    (void)correction;
    (void)equation_of_state;
    const auto special_argument =
        [&](tmpl::type_<NormalDotMeshVelocity> /*meta*/) {
          return *normal_dot_mesh_velocity;
        };
    if (normal_dot_mesh_velocity != nullptr) {
      apply_to_faces<RusanovPackageData<true>>(exec, packaged_data, faces,
                                               special_argument);
    } else {
      apply_to_faces<RusanovPackageData<false>>(exec, packaged_data, faces,
                                                special_argument);
    }
  } else {
    static_assert(std::is_same_v<Correction, Hll>);
    const Kokkos::View<const bool*> use_hydro_speeds =
        use_hydro_speeds_on_device(exec, faces, correction);
    const double light_speed_density_cutoff =
        correction.light_speed_density_cutoff();
    EquationsOfState::visit_device_view(
        equation_of_state, [&](const auto& device_view) {
          using DeviceView = std::decay_t<decltype(device_view)>;
          const auto special_argument = [&]<typename Tag>(
                                            tmpl::type_<Tag> /*meta*/) {
            if constexpr (std::is_same_v<
                              Tag,
                              EquationsOfState::Tags::DeviceView<DeviceView>>) {
              return device_view;
            } else if constexpr (std::is_same_v<
                                     Tag, ::Tags::PerElement<UseHydroSpeeds>>) {
              return use_hydro_speeds;
            } else if constexpr (std::is_same_v<Tag, LightSpeedDensityCutoff>) {
              return light_speed_density_cutoff;
            } else {
              static_assert(std::is_same_v<Tag, NormalDotMeshVelocity>);
              return *normal_dot_mesh_velocity;
            }
          };
          if (normal_dot_mesh_velocity != nullptr) {
            apply_to_faces<HllPackageData<true, DeviceView>>(
                exec, packaged_data, faces, special_argument);
          } else {
            apply_to_faces<HllPackageData<false, DeviceView>>(
                exec, packaged_data, faces, special_argument);
          }
        });
  }
}

template <typename Correction, typename FaceTags>
void fluxes_and_package_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<
        BatchedVariables<typename Correction::dg_package_field_tags>*>
        packaged_data,
    const BatchedVariables<FaceTags>& faces, const FaceCovectorBatch& conormal,
    const double sign, const FaceVectorBatch* const mesh_velocity,
    const Correction& correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  // The arguments that are the same for both corrections
  const auto common_argument = [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    if constexpr (std::is_same_v<Tag, Conormal>) {
      return conormal;
    } else if constexpr (std::is_same_v<Tag, Sign>) {
      return sign;
    } else {
      static_assert(std::is_same_v<Tag, MeshVelocity>);
      return *mesh_velocity;
    }
  };
  if constexpr (std::is_same_v<Correction, Rusanov>) {
    (void)correction;
    (void)equation_of_state;
    if (mesh_velocity != nullptr) {
      apply_to_faces<FusedRusanovPackageData<true>>(exec, packaged_data, faces,
                                                    common_argument);
    } else {
      apply_to_faces<FusedRusanovPackageData<false>>(exec, packaged_data, faces,
                                                     common_argument);
    }
  } else {
    static_assert(std::is_same_v<Correction, Hll>);
    const Kokkos::View<const bool*> use_hydro_speeds =
        use_hydro_speeds_on_device(exec, faces, correction);
    const double light_speed_density_cutoff =
        correction.light_speed_density_cutoff();
    EquationsOfState::visit_device_view(
        equation_of_state, [&](const auto& device_view) {
          using DeviceView = std::decay_t<decltype(device_view)>;
          const auto special_argument = [&]<typename Tag>(
                                            tmpl::type_<Tag> tag) {
            if constexpr (std::is_same_v<
                              Tag,
                              EquationsOfState::Tags::DeviceView<DeviceView>>) {
              return device_view;
            } else if constexpr (std::is_same_v<
                                     Tag, ::Tags::PerElement<UseHydroSpeeds>>) {
              return use_hydro_speeds;
            } else if constexpr (std::is_same_v<Tag, LightSpeedDensityCutoff>) {
              return light_speed_density_cutoff;
            } else {
              return common_argument(tag);
            }
          };
          if (mesh_velocity != nullptr) {
            apply_to_faces<FusedHllPackageData<true, DeviceView>>(
                exec, packaged_data, faces, special_argument);
          } else {
            apply_to_faces<FusedHllPackageData<false, DeviceView>>(
                exec, packaged_data, faces, special_argument);
          }
        });
  }
}

namespace {
// With a `PointwiseStatusSummary` result if `status` is `nullptr`, and else
// recording the status in `*status`
template <typename StatusPointer>
PointwiseStatusSummary package_data_from_reconstruction_impl(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<Rusanov::dg_package_field_tags>*>
        packaged_data,
    const StatusPointer deferred_status,
    const BatchedVariables<fd::reconstruction_face_tags>& faces,
    const FaceCovectorBatch& conormal, const double sign,
    const FaceVectorBatch* const mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* const
        fix_to_atmosphere) {
  PointwiseStatusSummary status{};
  EquationsOfState::visit_device_view(
      equation_of_state, [&](const auto& device_view) {
        using DeviceView = std::decay_t<decltype(device_view)>;
        const auto apply = [&]<bool FixAtmosphere, bool HasMeshVelocity>(
                               std::bool_constant<FixAtmosphere> /*meta*/,
                               std::bool_constant<HasMeshVelocity> /*meta*/) {
          const auto special_argument = [&]<typename Tag>(
                                            tmpl::type_<Tag> /*meta*/) {
            if constexpr (std::is_same_v<Tag, Conormal>) {
              return conormal;
            } else if constexpr (std::is_same_v<Tag, Sign>) {
              return sign;
            } else if constexpr (std::is_same_v<Tag, MeshVelocity>) {
              return *mesh_velocity;
            } else if constexpr (std::is_same_v<Tag, fd::FixToAtmosphereView>) {
              return *fix_to_atmosphere;
            } else {
              static_assert(
                  std::is_same_v<
                      Tag, EquationsOfState::Tags::DeviceView<DeviceView>>);
              return device_view;
            }
          };
          if constexpr (std::is_null_pointer_v<StatusPointer>) {
            status = apply_to_faces<RusanovPackageDataFromReconstruction<
                DeviceView, FixAtmosphere, HasMeshVelocity>>(
                exec, packaged_data, faces, special_argument);
          } else {
            apply_to_faces<RusanovPackageDataFromReconstruction<
                DeviceView, FixAtmosphere, HasMeshVelocity>>(
                exec, packaged_data, faces, special_argument, deferred_status);
          }
        };
        const bool fix_atmosphere = fix_to_atmosphere != nullptr;
        const bool has_mesh_velocity = mesh_velocity != nullptr;
        if (fix_atmosphere and has_mesh_velocity) {
          apply(std::true_type{}, std::true_type{});
        } else if (fix_atmosphere) {
          apply(std::true_type{}, std::false_type{});
        } else if (has_mesh_velocity) {
          apply(std::false_type{}, std::true_type{});
        } else {
          apply(std::false_type{}, std::false_type{});
        }
      });
  return status;
}

}  // namespace

PointwiseStatusSummary package_data_from_reconstruction_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<Rusanov::dg_package_field_tags>*>
        packaged_data,
    const BatchedVariables<fd::reconstruction_face_tags>& faces,
    const FaceCovectorBatch& conormal, const double sign,
    const FaceVectorBatch* const mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* const
        fix_to_atmosphere) {
  return package_data_from_reconstruction_impl(
      exec, packaged_data, nullptr, faces, conormal, sign, mesh_velocity,
      equation_of_state, fix_to_atmosphere);
}

void package_data_from_reconstruction_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<Rusanov::dg_package_field_tags>*>
        packaged_data,
    const gsl::not_null<
        DeferredPointwiseStatus<fd::face_atmosphere_snapshot_tags>*>
        status,
    const BatchedVariables<fd::reconstruction_face_tags>& faces,
    const FaceCovectorBatch& conormal, const double sign,
    const FaceVectorBatch* const mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* const
        fix_to_atmosphere) {
  package_data_from_reconstruction_impl(exec, packaged_data, status.get(),
                                        faces, conormal, sign, mesh_velocity,
                                        equation_of_state, fix_to_atmosphere);
}

template <typename Correction>
void boundary_terms_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<BoundaryCorrectionTags>*>
        boundary_corrections,
    const BatchedVariables<typename Correction::dg_package_field_tags>&
        interior_packaged_data,
    const BatchedVariables<typename Correction::dg_package_field_tags>&
        exterior_packaged_data,
    const dg::Formulation dg_formulation) {
  using Function = BoundaryTerms<Correction>;
  apply_pointwise_on_device<Function>(
      exec, boundary_corrections, typename Function::return_tags{},
      typename Function::argument_tags{},
      [&]<typename... PackagedTags>(tmpl::list<PackagedTags...> /*meta*/) {
        return make_device_arguments<PackagedTags..., Exterior<PackagedTags>...,
                                     DgFormulation>(
            exec,
            get<::Tags::BatchView<PackagedTags>>(interior_packaged_data)...,
            get<::Tags::BatchView<PackagedTags>>(exterior_packaged_data)...,
            dg_formulation);
      }(typename Correction::dg_package_field_tags{}));
}

#define CORRECTION(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template void package_data_on_device<CORRECTION(data)>(                  \
      const Kokkos::DefaultExecutionSpace& exec,                           \
      gsl::not_null<                                                       \
          BatchedVariables<CORRECTION(data)::dg_package_field_tags>*>      \
          packaged_data,                                                   \
      const BatchedVariables<fd::face_tags>& faces,                        \
      const CORRECTION(data) & correction,                                 \
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state, \
      const NormalDotMeshVelocityBatch* normal_dot_mesh_velocity);         \
  template void boundary_terms_on_device<CORRECTION(data)>(                \
      const Kokkos::DefaultExecutionSpace& exec,                           \
      gsl::not_null<BatchedVariables<BoundaryCorrectionTags>*>             \
          boundary_corrections,                                            \
      const BatchedVariables<CORRECTION(data)::dg_package_field_tags>&     \
          interior_packaged_data,                                          \
      const BatchedVariables<CORRECTION(data)::dg_package_field_tags>&     \
          exterior_packaged_data,                                          \
      dg::Formulation dg_formulation);

GENERATE_INSTANTIATIONS(INSTANTIATION, (Rusanov, Hll))

#undef INSTANTIATION

#define FACE_TAGS(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                          \
  template void                                                         \
  fluxes_and_package_data_on_device<CORRECTION(data), FACE_TAGS(data)>( \
      const Kokkos::DefaultExecutionSpace& exec,                        \
      gsl::not_null<                                                    \
          BatchedVariables<CORRECTION(data)::dg_package_field_tags>*>   \
          packaged_data,                                                \
      const BatchedVariables<FACE_TAGS(data)>& faces,                   \
      const FaceCovectorBatch& conormal, double sign,                   \
      const FaceVectorBatch* mesh_velocity,                             \
      const CORRECTION(data) & correction,                              \
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (Rusanov, Hll),
                        (fd::face_tags, fd::device_face_tags))

#undef INSTANTIATION
#undef FACE_TAGS
#undef CORRECTION
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
#endif  // SPECTRE_KOKKOS
