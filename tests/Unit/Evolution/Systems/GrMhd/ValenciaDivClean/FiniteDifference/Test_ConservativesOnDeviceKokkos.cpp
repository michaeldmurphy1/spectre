// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PiecewisePolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace EoS = EquationsOfState;
using grmhd::ValenciaDivClean::fd::face_tags;
using Fixer = VariableFixing::FixToAtmosphere<3>;
constexpr size_t num_points = 50;
constexpr size_t num_face_sets = 3;

// Random reconstructed primitive variables and metric on a face. The rest mass
// density spans the atmosphere and the limiting densities of the fixer below.
Variables<face_tags> random_face(const gsl::not_null<std::mt19937*> generator) {
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_log_density(-13.0, 0.0);
  std::uniform_real_distribution<> dist_unit(0.0, 1.0);
  std::uniform_real_distribution<> dist_velocity(-0.3, 0.3);
  std::uniform_real_distribution<> dist_field(-1.0, 1.0);
  std::uniform_real_distribution<> dist_metric(0.8, 1.2);
  std::uniform_real_distribution<> dist_off_diagonal(-0.05, 0.05);
  Variables<face_tags> face{num_points,
                            std::numeric_limits<double>::signaling_NaN()};
  auto& rest_mass_density =
      get(get<hydro::Tags::RestMassDensity<DataVector>>(face));
  rest_mass_density = make_with_random_values<DataVector>(
      generator, make_not_null(&dist_log_density), used_for_size);
  for (double& density : rest_mass_density) {
    density = std::pow(10.0, density);
  }
  get(get<hydro::Tags::ElectronFraction<DataVector>>(face)) =
      0.05 + 0.4 * make_with_random_values<DataVector>(
                       generator, make_not_null(&dist_unit), used_for_size);
  get(get<hydro::Tags::Temperature<DataVector>>(face)) =
      1.e-3 * make_with_random_values<DataVector>(
                  generator, make_not_null(&dist_unit), used_for_size);
  get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(face) =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_velocity), used_for_size);
  get<hydro::Tags::MagneticField<DataVector, 3>>(face) =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_field), used_for_size);
  get<hydro::Tags::DivergenceCleaningField<DataVector>>(face) =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_field), used_for_size);
  auto& spatial_metric = get<gr::Tags::SpatialMetric<DataVector, 3>>(face);
  spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_off_diagonal), used_for_size);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist_metric), used_for_size);
  }
  get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(face)) =
      sqrt(get(determinant(spatial_metric)));
  // Values before the computation, which the fix to the atmosphere can read
  get(get<hydro::Tags::SpecificInternalEnergy<DataVector>>(face)) = 1.0;
  get(get<hydro::Tags::Pressure<DataVector>>(face)) = 1.0;
  return face;
}

BatchedVariables<face_tags> to_device(
    const std::vector<Variables<face_tags>>& faces) {
  BatchedVariables<face_tags, Kokkos::HostSpace> host_batch{num_points,
                                                            faces.size()};
  for (size_t e = 0; e < faces.size(); ++e) {
    Variables<face_tags> face{};
    set_data_ref_to_element(make_not_null(&face), host_batch, e);
    face = faces[e];
  }
  BatchedVariables<face_tags> result{num_points, faces.size()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

std::vector<Variables<face_tags>> to_host(
    const BatchedVariables<face_tags>& batch) {
  BatchedVariables<face_tags, Kokkos::HostSpace> host_batch{
      num_points, batch.number_of_elements()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();
  std::vector<Variables<face_tags>> result{};
  for (size_t e = 0; e < batch.number_of_elements(); ++e) {
    Variables<face_tags> face{};
    set_data_ref_to_element(make_not_null(&face), host_batch, e);
    result.push_back(face);
  }
  return result;
}

template <typename TagsList, typename SourceTagsList>
BatchedVariables<TagsList> subset_to_device(
    const std::vector<Variables<SourceTagsList>>& elements) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{num_points,
                                                           elements.size()};
  for (size_t e = 0; e < elements.size(); ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    element.assign_subset(elements[e]);
  }
  BatchedVariables<TagsList> result{num_points, elements.size()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&result), host_batch);
  exec.fence();
  return result;
}

struct Conormal : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};
struct MeshVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

// `package_data_from_reconstruction_on_device` must agree with computing the
// variables on the faces first and then the packaged data, as the FD time
// derivative does with `Hll`
void check_package_data_from_reconstruction(
    const gsl::not_null<std::mt19937*> generator,
    const EoS::EquationOfState<true, 3>& eos,
    const std::optional<Fixer>& fixer) {
  namespace BoundaryCorrections = grmhd::ValenciaDivClean::BoundaryCorrections;
  using packaged_tags = BoundaryCorrections::Rusanov::dg_package_field_tags;
  using geometry_tags = tmpl::list<Conormal, MeshVelocity>;
  const Approx custom_approx = Approx::custom().epsilon(1.e-14).scale(1.0);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  std::uniform_real_distribution<> dist_lapse(0.8, 1.2);
  std::uniform_real_distribution<> dist_shift(-0.1, 0.1);
  const Kokkos::DefaultExecutionSpace exec{};
  const std::optional<Fixer::DeviceView> fixer_view =
      fixer.has_value() ? std::optional{fixer->device_view()} : std::nullopt;
  const Fixer::DeviceView* const fixer_pointer =
      fixer_view.has_value() ? &*fixer_view : nullptr;

  std::vector<Variables<face_tags>> faces{};
  std::vector<Variables<geometry_tags>> geometry{};
  for (size_t e = 0; e < num_face_sets; ++e) {
    auto& face = faces.emplace_back(random_face(generator));
    get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(face) =
        determinant_and_inverse(
            get<gr::Tags::SpatialMetric<DataVector, 3>>(face))
            .second;
    get<gr::Tags::Lapse<DataVector>>(face) =
        make_with_random_values<Scalar<DataVector>>(
            generator, make_not_null(&dist_lapse), DataVector{num_points});
    get<gr::Tags::Shift<DataVector, 3>>(face) =
        make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
            generator, make_not_null(&dist_shift), DataVector{num_points});
    auto& face_geometry = geometry.emplace_back(num_points);
    fill_with_random_values(make_not_null(&face_geometry), generator,
                            make_not_null(&dist));
    // Away from zero
    get<0>(get<Conormal>(face_geometry)) += 2.0;
  }
  const auto device_geometry = subset_to_device<geometry_tags>(geometry);
  const auto conormal = get<::Tags::BatchView<Conormal>>(device_geometry);
  const auto mesh_velocity =
      get<::Tags::BatchView<MeshVelocity>>(device_geometry);
  const auto reconstruction =
      subset_to_device<grmhd::ValenciaDivClean::fd::reconstruction_face_tags>(
          faces);
  for (const bool moving_mesh : {false, true}) {
    for (const double sign : {1.0, -1.0}) {
      CAPTURE(moving_mesh);
      CAPTURE(sign);
      // In two steps
      auto all_variables = to_device(faces);
      const PointwiseStatusSummary expected_status =
          grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
              exec, make_not_null(&all_variables), eos, fixer_pointer);
      BatchedVariables<packaged_tags> expected{num_points, num_face_sets};
      BoundaryCorrections::fluxes_and_package_data_on_device(
          exec, make_not_null(&expected), all_variables, conormal, sign,
          moving_mesh ? &mesh_velocity : nullptr,
          BoundaryCorrections::Rusanov{}, eos);
      // In one
      BatchedVariables<packaged_tags> packaged{num_points, num_face_sets};
      const PointwiseStatusSummary status =
          BoundaryCorrections::package_data_from_reconstruction_on_device(
              exec, make_not_null(&packaged), reconstruction, conormal, sign,
              moving_mesh ? &mesh_velocity : nullptr, eos, fixer_pointer);
      CHECK(status.number_of_failures == expected_status.number_of_failures);
      CHECK(status.number_of_failures == 0);
      const auto to_host_vars =
          [](const BatchedVariables<packaged_tags>& batch) {
            BatchedVariables<packaged_tags, Kokkos::HostSpace> host_batch{
                num_points, num_face_sets};
            const Kokkos::DefaultExecutionSpace host_exec{};
            deep_copy(host_exec, make_not_null(&host_batch), batch);
            host_exec.fence();
            std::vector<Variables<packaged_tags>> result{};
            for (size_t e = 0; e < num_face_sets; ++e) {
              Variables<packaged_tags> element{};
              set_data_ref_to_element(make_not_null(&element), host_batch, e);
              result.emplace_back(element);
            }
            return result;
          };
      const auto host_expected = to_host_vars(expected);
      const auto host_packaged = to_host_vars(packaged);
      for (size_t e = 0; e < num_face_sets; ++e) {
        CAPTURE(e);
        CHECK_VARIABLES_CUSTOM_APPROX(host_packaged[e], host_expected[e],
                                      custom_approx);
      }
    }
  }
}

// The batched version computes the variables on each face set like
// `compute_conservatives_for_reconstruction` on the host
void check_against_host(const gsl::not_null<std::mt19937*> generator,
                        const EoS::EquationOfState<true, 3>& eos,
                        const std::optional<Fixer>& fixer) {
  CAPTURE(fixer.has_value());
  std::vector<Variables<face_tags>> faces{};
  for (size_t e = 0; e < num_face_sets; ++e) {
    faces.push_back(random_face(generator));
  }
  auto batch = to_device(faces);
  const std::optional<Fixer::DeviceView> fixer_view =
      fixer.has_value() ? std::optional{fixer->device_view()} : std::nullopt;
  const PointwiseStatusSummary status =
      grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
          Kokkos::DefaultExecutionSpace{}, make_not_null(&batch), eos,
          fixer_view.has_value() ? &*fixer_view : nullptr);
  CHECK(status.number_of_failures == 0);
  const auto computed = to_host(batch);

  for (size_t e = 0; e < num_face_sets; ++e) {
    CAPTURE(e);
    auto expected = faces[e];
    grmhd::ValenciaDivClean::fd::compute_conservatives_for_reconstruction(
        make_not_null(&expected), eos, fixer.has_value() ? &*fixer : nullptr);
    // The device code may be compiled differently (e.g. with fused
    // multiply-adds), and the kappa limiting of the fix to the atmosphere
    // finds the temperature with a root finder to a relative tolerance of
    // 1e-13
    const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
    tmpl::for_each<face_tags>([&](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      CAPTURE(pretty_type::get_name<Tag>());
      const auto& computed_tensor = get<Tag>(computed[e]);
      const auto& expected_tensor = get<Tag>(expected);
      for (size_t component = 0; component < computed_tensor.size();
           ++component) {
        for (size_t i = 0; i < num_points; ++i) {
          const double value = computed_tensor[component][i];
          const double expected_value = expected_tensor[component][i];
          // Variables that neither version computes stay NaN
          if (std::isnan(expected_value)) {
            CHECK(std::isnan(value));
          } else {
            CHECK(value == custom_approx(expected_value));
          }
        }
      }
    });
  }
}

// The options don't allow a negative minimum temperature, but with one the
// kappa limiting fails (see the test of `fix_to_atmosphere_on_device`). The
// failure is reported, the variables at the failed grid point are left as they
// were before the fix, and the error shows them.
void check_atmosphere_failure(const gsl::not_null<std::mt19937*> generator) {
  auto fixer_view = Fixer{
      1.e-12, 3.e-12, std::nullopt,
      Fixer::KappaLimitingOptions{
          3.e-12, 1.e-3, 3.e-11, 0.01, std::nullopt,
          false}}.device_view();
  fixer_view.kappa_limiting.min_temperature_from_eos = false;
  fixer_view.kappa_limiting.min_temperature = -1.0;
  const EoS::Equilibrium3D<EoS::IdealFluid<true>> eos{
      EoS::IdealFluid<true>{5.0 / 3.0}};
  auto face = random_face(generator);
  // Far above the limiting densities except at one grid point
  get(get<hydro::Tags::RestMassDensity<DataVector>>(face)) = 0.5;
  get(get<hydro::Tags::RestMassDensity<DataVector>>(face))[7] = 1.e-11;
  auto batch = to_device({face, face});
  const PointwiseStatusSummary status =
      grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
          Kokkos::DefaultExecutionSpace{}, make_not_null(&batch), eos,
          &fixer_view);
  CHECK(status.number_of_failures == 2);
  CHECK(status.first_failed_element == 0);
  CHECK(status.first_failed_grid_point == 7);
  const auto computed = to_host(batch);
  CHECK(get(get<hydro::Tags::RestMassDensity<DataVector>>(computed[1]))[7] ==
        1.e-11);
  CHECK(get(get<hydro::Tags::Temperature<DataVector>>(computed[1]))[7] ==
        get(get<hydro::Tags::Temperature<DataVector>>(face))[7]);
  CHECK(get(get<hydro::Tags::Pressure<DataVector>>(computed[1]))[7] == 1.0);
  CHECK_THROWS_WITH(VariableFixing::error_fixing_to_atmosphere_on_device(
                        Kokkos::DefaultExecutionSpace{}, status, batch,
                        "the faces in direction 0 of [B0,(L0I0)]"),
                    Catch::Matchers::ContainsSubstring(
                        "failed on the device at 2 of 100 grid points. At the "
                        "first one, grid point 7 of the faces in direction 0 "
                        "of [B0,(L0I0)], before fixing:"));

  // With a deferred status, the failures are recorded on the device, with the
  // reconstructed variables at one of the failed grid points
  const Kokkos::DefaultExecutionSpace exec{};
  auto deferred_batch = to_device({face, face});
  DeferredPointwiseStatus<
      grmhd::ValenciaDivClean::fd::face_atmosphere_snapshot_tags>
      deferred_status{exec};
  deferred_status.reset(exec);
  grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
      exec, make_not_null(&deferred_batch), make_not_null(&deferred_status),
      eos, &fixer_view);
  deferred_status.copy_to_host(exec);
  exec.fence();
  REQUIRE(deferred_status.failed());
  CHECK(deferred_status.summary().number_of_failures == 2);
  CHECK(deferred_status.summary().first_failed_grid_point == 7);
  // The faces may hold NaNs, so compare what the failure leaves unfixed
  const auto deferred_computed = to_host(deferred_batch);
  CHECK(get(get<hydro::Tags::RestMassDensity<DataVector>>(
            deferred_computed[1]))[7] == 1.e-11);
  CHECK(get(get<hydro::Tags::Pressure<DataVector>>(deferred_computed[1]))[7] ==
        1.0);
  const auto snapshot = deferred_status.snapshot();
  CHECK(get(get<::Tags::AtIndex<hydro::Tags::RestMassDensity<DataVector>>>(
            snapshot)) == 1.e-11);
  CHECK(get(get<::Tags::AtIndex<hydro::Tags::Temperature<DataVector>>>(
            snapshot)) ==
        get(get<hydro::Tags::Temperature<DataVector>>(face))[7]);
  CHECK_THROWS_WITH(
      VariableFixing::error_fixing_to_atmosphere_on_device(
          deferred_status, 100, "the faces in direction 0 of [B0,(L0I0)]"),
      Catch::Matchers::ContainsSubstring(
          "failed on the device at 2 of 100 grid points. At one "
          "of them, grid point 7 of the faces in direction 0 "
          "of [B0,(L0I") and
          Catch::Matchers::ContainsSubstring(
              "rest_mass_density = T()=9.999999999999999395e-12"));
}

void check_error_without_device_view(
    const gsl::not_null<std::mt19937*> generator) {
  const auto eos_ptr =
      EoS::PiecewisePolytropicFluid<true>{1.0e-4, 100.0, 1.5, 2.0}
          .promote_to_3d_eos();
  const auto& eos = *eos_ptr;
  auto batch = to_device({random_face(generator)});
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
          Kokkos::DefaultExecutionSpace{}, make_not_null(&batch), eos, nullptr),
      Catch::Matchers::ContainsSubstring(
          "has no DeviceView, so it can't be evaluated on the device"));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Fd.ConservativesOnDevice",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  const EoS::Equilibrium3D<EoS::IdealFluid<true>> ideal_fluid{
      EoS::IdealFluid<true>{5.0 / 3.0}};
  const EoS::Barotropic3D<EoS::PolytropicFluid<true>> polytrope{
      EoS::PolytropicFluid<true>{100.0, 2.0}};
  const Fixer fixer{1.e-12, 3.e-12,
                    Fixer::VelocityLimitingOptions{0.0, 1.e-4, 3.e-12, 1.e-11},
                    Fixer::KappaLimitingOptions{3.e-12, 1.e-3, 3.e-11, 0.01,
                                                std::nullopt, false}};
  for (const auto* const eos :
       {static_cast<const EoS::EquationOfState<true, 3>*>(&ideal_fluid),
        static_cast<const EoS::EquationOfState<true, 3>*>(&polytrope)}) {
    check_against_host(make_not_null(&generator), *eos, std::nullopt);
    check_against_host(make_not_null(&generator), *eos, fixer);
    check_package_data_from_reconstruction(make_not_null(&generator), *eos,
                                           std::nullopt);
    check_package_data_from_reconstruction(make_not_null(&generator), *eos,
                                           fixer);
  }
  check_atmosphere_failure(make_not_null(&generator));
  check_error_without_device_view(make_not_null(&generator));
}
