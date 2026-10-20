// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDeviceKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <boost/preprocessor/tuple/elem.hpp>
#include <cstddef>
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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
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
Kokkos::View<const bool*> use_hydro_speeds_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const BatchedVariables<fd::face_tags>& faces, const Hll& hll) {
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
// variables given by `special_argument`
template <typename Function, typename ResultTags, typename SpecialArgument>
void apply_to_faces(const Kokkos::DefaultExecutionSpace& exec,
                    const gsl::not_null<BatchedVariables<ResultTags>*> results,
                    const BatchedVariables<fd::face_tags>& faces,
                    const SpecialArgument& special_argument) {
  // The face variables are returned by reference and the other arguments,
  // which are cheap to copy, by value
  const auto argument =
      [&]<typename Tag>(tmpl::type_<Tag> tag) -> decltype(auto) {
    if constexpr (tmpl::list_contains_v<fd::face_tags, Tag>) {
      (void)tag;
      return get<::Tags::BatchView<Tag>>(faces);
    } else {
      return special_argument(tag);
    }
  };
  apply_pointwise_on_device<Function>(
      exec, results, typename Function::return_tags{},
      typename Function::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, argument(tmpl::type_<ArgumentTags>{})...);
      }(typename Function::argument_tags{}));
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
#undef CORRECTION
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
#endif  // SPECTRE_KOKKOS
