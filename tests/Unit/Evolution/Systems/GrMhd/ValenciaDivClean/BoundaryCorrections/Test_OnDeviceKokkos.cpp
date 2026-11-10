// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace BoundaryCorrections = grmhd::ValenciaDivClean::BoundaryCorrections;
namespace EoS = EquationsOfState;
namespace VdcTags = grmhd::ValenciaDivClean::Tags;
using BoundaryCorrections::BoundaryCorrectionTags;
using grmhd::ValenciaDivClean::fd::face_tags;

struct NormalDotMeshVelocity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Conormal : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};
struct MeshVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

constexpr size_t num_points = 50;
constexpr size_t num_face_sets = 3;

// Random face variables that are physically valid where it matters (e.g.
// positive lapse and density). The magnetic field of the middle face set is
// large and that of the others is small.
Variables<face_tags> random_face(const gsl::not_null<std::mt19937*> generator,
                                 const size_t face_set) {
  Variables<face_tags> face{num_points};
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  fill_with_random_values(make_not_null(&face), generator,
                          make_not_null(&dist));
  std::uniform_real_distribution<> dist_lapse(0.8, 1.2);
  std::uniform_real_distribution<> dist_density(0.1, 2.0);
  std::uniform_real_distribution<> dist_electron_fraction(0.05, 0.5);
  std::uniform_real_distribution<> dist_velocity(-0.3, 0.3);
  fill_with_random_values(
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(face)), generator,
      make_not_null(&dist_lapse));
  fill_with_random_values(
      make_not_null(&get<hydro::Tags::RestMassDensity<DataVector>>(face)),
      generator, make_not_null(&dist_density));
  fill_with_random_values(
      make_not_null(&get<hydro::Tags::Temperature<DataVector>>(face)),
      generator, make_not_null(&dist_density));
  fill_with_random_values(
      make_not_null(&get<hydro::Tags::ElectronFraction<DataVector>>(face)),
      generator, make_not_null(&dist_electron_fraction));
  auto& spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(face);
  fill_with_random_values(make_not_null(&spatial_velocity), generator,
                          make_not_null(&dist_velocity));
  auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(face);
  for (size_t i = 0; i < 3; ++i) {
    spatial_velocity_one_form.get(i) = spatial_velocity.get(i);
    get<VdcTags::TildeB<>>(face).get(i) *= face_set == 1 ? 10.0 : 0.01;
  }
  return face;
}

template <typename TagsList>
BatchedVariables<TagsList> to_device(
    const std::vector<Variables<TagsList>>& elements) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{num_points,
                                                           elements.size()};
  for (size_t e = 0; e < elements.size(); ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    element = elements[e];
  }
  BatchedVariables<TagsList> result{num_points, elements.size()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

template <typename TagsList>
std::vector<Variables<TagsList>> to_host(
    const BatchedVariables<TagsList>& batch) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{
      num_points, batch.number_of_elements()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();
  std::vector<Variables<TagsList>> result{};
  for (size_t e = 0; e < batch.number_of_elements(); ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    result.push_back(element);
  }
  return result;
}

// The packaged data of `Correction::dg_package_data` on the host
template <typename Correction>
Variables<typename Correction::dg_package_field_tags> package_data_on_host(
    const Correction& correction, const Variables<face_tags>& face,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EoS::EquationOfState<true, 3>& eos) {
  using argument_tags =
      tmpl::append<BoundaryCorrectionTags,
                   db::wrap_tags_in<::Tags::Flux, BoundaryCorrectionTags,
                                    tmpl::size_t<3>, Frame::Inertial>,
                   typename Correction::dg_package_data_temporary_tags,
                   typename Correction::dg_package_data_primitive_tags>;
  Variables<typename Correction::dg_package_field_tags> packaged_data{
      num_points};
  tmpl::as_pack<typename Correction::dg_package_field_tags>(
      [&](auto... packaged_tags) {
        tmpl::as_pack<argument_tags>([&](auto... arg_tags) {
          correction.dg_package_data(
              make_not_null(&get<tmpl::type_from<decltype(packaged_tags)>>(
                  packaged_data))...,
              get<tmpl::type_from<decltype(arg_tags)>>(face)...,
              get<evolution::dg::Tags::NormalCovector<3>>(face),
              get<evolution::dg::Actions::detail::NormalVector<3>>(face),
              mesh_velocity, normal_dot_mesh_velocity, eos);
        });
      });
  return packaged_data;
}

// The boundary terms of `Correction::dg_boundary_terms` on the host
template <typename Correction, typename PackagedTags>
Variables<BoundaryCorrectionTags> boundary_terms_on_host(
    const Correction& correction, const Variables<PackagedTags>& interior,
    const Variables<PackagedTags>& exterior,
    const dg::Formulation dg_formulation) {
  Variables<BoundaryCorrectionTags> result{num_points};
  tmpl::as_pack<PackagedTags>([&](auto... packaged_tags) {
    tmpl::as_pack<BoundaryCorrectionTags>([&](auto... result_tags) {
      correction.dg_boundary_terms(
          make_not_null(
              &get<tmpl::type_from<decltype(result_tags)>>(result))...,
          get<tmpl::type_from<decltype(packaged_tags)>>(interior)...,
          get<tmpl::type_from<decltype(packaged_tags)>>(exterior)...,
          dg_formulation);
    });
  });
  return result;
}

// The packaged data and boundary terms of a batch of face sets computed on the
// device must agree with the ones computed on the host for each face set
template <typename Correction>
void check_against_host(const gsl::not_null<std::mt19937*> generator,
                        const Correction& correction,
                        const EoS::EquationOfState<true, 3>& eos) {
  // The device code may be compiled differently (e.g. with fused
  // multiply-adds)
  const Approx custom_approx = Approx::custom().epsilon(1.e-14).scale(1.0);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const Kokkos::DefaultExecutionSpace exec{};
  using packaged_tags = typename Correction::dg_package_field_tags;

  for (const bool moving_mesh : {false, true}) {
    CAPTURE(moving_mesh);
    std::vector<Variables<face_tags>> interior_faces{};
    std::vector<Variables<face_tags>> exterior_faces{};
    std::vector<std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>>
        mesh_velocities(num_face_sets);
    std::vector<Variables<tmpl::list<NormalDotMeshVelocity>>>
        normal_dot_mesh_velocities{};
    for (size_t e = 0; e < num_face_sets; ++e) {
      interior_faces.push_back(random_face(generator, e));
      exterior_faces.push_back(random_face(generator, e));
      // The same normal on both sides, as on the host
      get<evolution::dg::Tags::NormalCovector<3>>(exterior_faces.back()) =
          get<evolution::dg::Tags::NormalCovector<3>>(interior_faces.back());
      normal_dot_mesh_velocities.emplace_back(num_points, 0.0);
      if (moving_mesh) {
        mesh_velocities[e] =
            make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
                generator, make_not_null(&dist), DataVector{num_points});
        get<NormalDotMeshVelocity>(normal_dot_mesh_velocities.back()) =
            dot_product(*mesh_velocities[e],
                        get<evolution::dg::Tags::NormalCovector<3>>(
                            interior_faces.back()));
      }
    }
    const auto device_normal_dot_mesh_velocity =
        to_device(normal_dot_mesh_velocities);
    const auto& normal_dot_mesh_velocity_batch =
        get<::Tags::BatchView<NormalDotMeshVelocity>>(
            device_normal_dot_mesh_velocity);
    const auto package_on_device =
        [&](const std::vector<Variables<face_tags>>& faces) {
          BatchedVariables<packaged_tags> packaged{num_points, num_face_sets};
          BoundaryCorrections::package_data_on_device(
              exec, make_not_null(&packaged), to_device(faces), correction, eos,
              moving_mesh ? &normal_dot_mesh_velocity_batch : nullptr);
          return packaged;
        };
    const auto interior = package_on_device(interior_faces);
    const auto exterior = package_on_device(exterior_faces);
    const auto host_interior = to_host(interior);
    const auto host_exterior = to_host(exterior);

    for (const auto dg_formulation :
         {dg::Formulation::StrongInertial, dg::Formulation::WeakInertial}) {
      CAPTURE(dg_formulation);
      BatchedVariables<BoundaryCorrectionTags> corrections{num_points,
                                                           num_face_sets};
      BoundaryCorrections::boundary_terms_on_device<Correction>(
          exec, make_not_null(&corrections), interior, exterior,
          dg_formulation);
      const auto host_corrections = to_host(corrections);
      for (size_t e = 0; e < num_face_sets; ++e) {
        CAPTURE(e);
        const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
            moving_mesh ? std::optional{get<NormalDotMeshVelocity>(
                              normal_dot_mesh_velocities[e])}
                        : std::nullopt;
        const auto expected_interior = package_data_on_host(
            correction, interior_faces[e], mesh_velocities[e],
            normal_dot_mesh_velocity, eos);
        const auto expected_exterior = package_data_on_host(
            correction, exterior_faces[e], mesh_velocities[e],
            normal_dot_mesh_velocity, eos);
        CHECK_VARIABLES_CUSTOM_APPROX(host_interior[e], expected_interior,
                                      custom_approx);
        CHECK_VARIABLES_CUSTOM_APPROX(host_exterior[e], expected_exterior,
                                      custom_approx);
        CHECK_VARIABLES_CUSTOM_APPROX(
            host_corrections[e],
            boundary_terms_on_host(correction, expected_interior,
                                   expected_exterior, dg_formulation),
            custom_approx);
      }
    }
  }
}

// `fluxes_and_package_data_on_device` must agree with computing the fluxes,
// the mesh velocity terms and the normal covector first and then the packaged
// data, as the FD time derivative does on the host
template <typename Correction>
void check_fused(const gsl::not_null<std::mt19937*> generator,
                 const Correction& correction,
                 const EoS::EquationOfState<true, 3>& eos) {
  const Approx custom_approx = Approx::custom().epsilon(1.e-13).scale(1.0);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const Kokkos::DefaultExecutionSpace exec{};
  using packaged_tags = typename Correction::dg_package_field_tags;
  using geometry_tags = tmpl::list<Conormal, MeshVelocity>;

  for (const bool moving_mesh : {false, true}) {
    for (const double sign : {1.0, -1.0}) {
      CAPTURE(moving_mesh);
      CAPTURE(sign);
      std::vector<Variables<face_tags>> faces{};
      std::vector<Variables<geometry_tags>> geometry{};
      std::vector<Variables<packaged_tags>> expected{};
      for (size_t e = 0; e < num_face_sets; ++e) {
        // A spatial metric close to the flat one, with its inverse, so that
        // the conormal can be normalized
        auto& face = faces.emplace_back(random_face(generator, e));
        auto& spatial_metric =
            get<gr::Tags::SpatialMetric<DataVector, 3>>(face);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = i; j < 3; ++j) {
            spatial_metric.get(i, j) =
                (i == j ? 1.0 : 0.0) + 0.1 * spatial_metric.get(i, j);
          }
        }
        const auto [det_spatial_metric, inv_spatial_metric] =
            determinant_and_inverse(spatial_metric);
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(face) =
            inv_spatial_metric;
        get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(face)) =
            sqrt(get(det_spatial_metric));

        auto& face_geometry = geometry.emplace_back(num_points);
        fill_with_random_values(make_not_null(&face_geometry), generator,
                                make_not_null(&dist));
        auto& conormal = get<Conormal>(face_geometry);
        // Away from zero
        get<0>(conormal) += 2.0;

        // The fluxes, the mesh velocity terms and the normals on the host
        Variables<face_tags> host_face = face;
        tmpl::as_pack<grmhd::ValenciaDivClean::ComputeFluxes::return_tags>(
            [&](auto... flux_tags) {
              tmpl::as_pack<
                  grmhd::ValenciaDivClean::ComputeFluxes::argument_tags>(
                  [&](auto... argument_tags) {
                    grmhd::ValenciaDivClean::ComputeFluxes::apply(
                        make_not_null(
                            &get<tmpl::type_from<decltype(flux_tags)>>(
                                host_face))...,
                        get<tmpl::type_from<decltype(argument_tags)>>(
                            host_face)...);
                  });
            });
        std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity{};
        std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
        if (moving_mesh) {
          mesh_velocity = get<MeshVelocity>(face_geometry);
          tmpl::for_each<BoundaryCorrectionTags>([&](auto tag_v) {
            using Tag = tmpl::type_from<decltype(tag_v)>;
            const auto& variable = get<Tag>(host_face);
            auto& flux =
                get<::Tags::Flux<Tag, tmpl::size_t<3>, Frame::Inertial>>(
                    host_face);
            for (size_t j = 0; j < 3; ++j) {
              if constexpr (std::decay_t<decltype(variable)>::rank() == 0) {
                flux.get(j) -= mesh_velocity->get(j) * get(variable);
              } else {
                for (size_t i = 0; i < 3; ++i) {
                  flux.get(j, i) -= mesh_velocity->get(j) * variable.get(i);
                }
              }
            }
          });
        }
        Scalar<DataVector> magnitude_squared{num_points, 0.0};
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            get(magnitude_squared) += conormal.get(i) * conormal.get(j) *
                                      inv_spatial_metric.get(i, j);
          }
        }
        auto& normal_covector =
            get<evolution::dg::Tags::NormalCovector<3>>(host_face);
        for (size_t i = 0; i < 3; ++i) {
          normal_covector.get(i) =
              sign * conormal.get(i) / sqrt(get(magnitude_squared));
        }
        raise_or_lower_index(
            make_not_null(&get<evolution::dg::Actions::detail::NormalVector<3>>(
                host_face)),
            normal_covector, inv_spatial_metric);
        if (moving_mesh) {
          normal_dot_mesh_velocity =
              dot_product(*mesh_velocity, normal_covector);
        }
        expected.push_back(package_data_on_host(correction, host_face,
                                                mesh_velocity,
                                                normal_dot_mesh_velocity, eos));
      }

      // The fluxes and normals in the face variables on the device are
      // random, so this also checks that they aren't used
      const auto device_geometry = to_device(geometry);
      BatchedVariables<packaged_tags> packaged{num_points, num_face_sets};
      BoundaryCorrections::fluxes_and_package_data_on_device(
          exec, make_not_null(&packaged), to_device(faces),
          get<::Tags::BatchView<Conormal>>(device_geometry), sign,
          moving_mesh ? &get<::Tags::BatchView<MeshVelocity>>(device_geometry)
                      : nullptr,
          correction, eos);
      const auto host_packaged = to_host(packaged);
      for (size_t e = 0; e < num_face_sets; ++e) {
        CAPTURE(e);
        CHECK_VARIABLES_CUSTOM_APPROX(host_packaged[e], expected[e],
                                      custom_approx);
      }
    }
  }
}

// EOSs without a `DeviceView` are an error rather than evaluated on the host
void check_error_without_device_view(
    const gsl::not_null<std::mt19937*> generator) {
  const EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>> eos{
      EoS::HybridEos<EoS::PolytropicFluid<true>>{
          EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5}};
  BatchedVariables<BoundaryCorrections::Hll::dg_package_field_tags> packaged{
      num_points, 1};
  CHECK_THROWS_WITH(
      BoundaryCorrections::package_data_on_device(
          Kokkos::DefaultExecutionSpace{}, make_not_null(&packaged),
          to_device(std::vector{random_face(generator, 0)}),
          BoundaryCorrections::Hll{1.0e100, 1.0e-100}, eos),
      Catch::Matchers::ContainsSubstring(
          "has no DeviceView, so it can't be evaluated on the device"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.OnDevice",
                  "[Unit][GrMhd]") {
  MAKE_GENERATOR(generator);
  const EoS::Equilibrium3D<EoS::IdealFluid<true>> ideal_fluid{
      EoS::IdealFluid<true>{5.0 / 3.0}};
  const EoS::Barotropic3D<EoS::PolytropicFluid<true>> polytrope{
      EoS::PolytropicFluid<true>{100.0, 2.0}};

  check_against_host(make_not_null(&generator), BoundaryCorrections::Rusanov{},
                     ideal_fluid);
  check_fused(make_not_null(&generator), BoundaryCorrections::Rusanov{},
              ideal_fluid);
  for (const auto* eos :
       {static_cast<const EoS::EquationOfState<true, 3>*>(&ideal_fluid),
        static_cast<const EoS::EquationOfState<true, 3>*>(&polytrope)}) {
    // The densities are between 0.1 and 2, so the light speed is used at some
    // points and the hydro speeds at others
    check_against_host(make_not_null(&generator),
                       BoundaryCorrections::Hll{1.0e100, 1.0}, *eos);
    // The magnetic field is large, so the light speed is used everywhere
    check_against_host(make_not_null(&generator),
                       BoundaryCorrections::Hll{1.0e-100, 1.0}, *eos);
    // Only the middle face set has a magnetic field above the cutoff, so it
    // uses the light speed and the others the hydro speeds where the density
    // is large
    check_against_host(make_not_null(&generator),
                       BoundaryCorrections::Hll{1.0, 1.0}, *eos);
    check_fused(make_not_null(&generator), BoundaryCorrections::Hll{1.0, 1.0},
                *eos);
  }
  check_error_without_device_view(make_not_null(&generator));
}
