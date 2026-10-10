// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
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
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace EoS = EquationsOfState;
using grmhd::ValenciaDivClean::fd::face_tags;
constexpr size_t num_points = 50;
constexpr size_t num_face_sets = 3;

// Random reconstructed primitive variables and metric on a face
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

// The batched version computes the variables on each face set like
// `compute_conservatives_for_reconstruction` on the host
void check_against_host(const gsl::not_null<std::mt19937*> generator,
                        const EoS::EquationOfState<true, 3>& eos) {
  std::vector<Variables<face_tags>> faces{};
  for (size_t e = 0; e < num_face_sets; ++e) {
    faces.push_back(random_face(generator));
  }
  auto batch = to_device(faces);
  grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
      Kokkos::DefaultExecutionSpace{}, make_not_null(&batch), eos);
  const auto computed = to_host(batch);

  for (size_t e = 0; e < num_face_sets; ++e) {
    CAPTURE(e);
    auto expected = faces[e];
    grmhd::ValenciaDivClean::fd::compute_conservatives_for_reconstruction(
        make_not_null(&expected), eos, nullptr);
    // The device code may be compiled differently (e.g. with fused
    // multiply-adds)
    const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
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

void check_error_without_device_view(
    const gsl::not_null<std::mt19937*> generator) {
  const EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>> eos{
      EoS::HybridEos<EoS::PolytropicFluid<true>>{
          EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5}};
  auto batch = to_device({random_face(generator)});
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::fd::compute_conservatives_on_device(
          Kokkos::DefaultExecutionSpace{}, make_not_null(&batch), eos),
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
  for (const auto* const eos :
       {static_cast<const EoS::EquationOfState<true, 3>*>(&ideal_fluid),
        static_cast<const EoS::EquationOfState<true, 3>*>(&polytrope)}) {
    check_against_host(make_not_null(&generator), *eos);
  }
  check_error_without_device_view(make_not_null(&generator));
}
