// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDevice.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
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

using evolved_tags =
    tmpl::list<VdcTags::TildeD, VdcTags::TildeYe, VdcTags::TildeTau,
               VdcTags::TildeS<Frame::Inertial>,
               VdcTags::TildeB<Frame::Inertial>, VdcTags::TildePhi>;
using flux_tags = db::wrap_tags_in<::Tags::Flux, evolved_tags, tmpl::size_t<3>,
                                   Frame::Inertial>;
template <typename Correction>
using argument_tags =
    tmpl::append<evolved_tags, flux_tags,
                 typename Correction::dg_package_data_temporary_tags,
                 typename Correction::dg_package_data_primitive_tags>;

constexpr size_t num_points = 50;

// Random arguments of `dg_package_data` that are physically valid where it
// matters (e.g. positive lapse and density)
template <typename Correction>
Variables<argument_tags<Correction>> make_arguments(
    const gsl::not_null<std::mt19937*> generator) {
  Variables<argument_tags<Correction>> args{num_points};
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  fill_with_random_values(make_not_null(&args), generator,
                          make_not_null(&dist));
  std::uniform_real_distribution<> dist_lapse(0.8, 1.2);
  std::uniform_real_distribution<> dist_density(0.1, 2.0);
  std::uniform_real_distribution<> dist_electron_fraction(0.05, 0.5);
  std::uniform_real_distribution<> dist_velocity(-0.3, 0.3);
  fill_with_random_values(
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(args)), generator,
      make_not_null(&dist_lapse));
  if constexpr (tmpl::list_contains_v<
                    argument_tags<Correction>,
                    hydro::Tags::RestMassDensity<DataVector>>) {
    fill_with_random_values(
        make_not_null(&get<hydro::Tags::RestMassDensity<DataVector>>(args)),
        generator, make_not_null(&dist_density));
    fill_with_random_values(
        make_not_null(&get<hydro::Tags::Temperature<DataVector>>(args)),
        generator, make_not_null(&dist_density));
    fill_with_random_values(
        make_not_null(&get<hydro::Tags::ElectronFraction<DataVector>>(args)),
        generator, make_not_null(&dist_electron_fraction));
    auto& spatial_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(args);
    fill_with_random_values(make_not_null(&spatial_velocity), generator,
                            make_not_null(&dist_velocity));
    auto& spatial_velocity_one_form = get<
        hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>(
        args);
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity_one_form.get(i) = spatial_velocity.get(i);
    }
  }
  return args;
}

// `correction` is a `Correction` or an `OnDevice<Correction>`
template <typename Correction, typename CorrectionOrOnDevice>
Variables<typename Correction::dg_package_field_tags> package_data(
    const CorrectionOrOnDevice& correction,
    const Variables<argument_tags<Correction>>& args,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const EoS::EquationOfState<true, 3>& eos) {
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
  if (mesh_velocity.has_value()) {
    normal_dot_mesh_velocity = dot_product(*mesh_velocity, normal_covector);
  }
  tnsr::I<DataVector, 3, Frame::Inertial> normal_vector{num_points};
  for (size_t i = 0; i < 3; ++i) {
    normal_vector.get(i) = normal_covector.get(i);
  }
  Variables<typename Correction::dg_package_field_tags> packaged_data{
      num_points};
  tmpl::as_pack<typename Correction::dg_package_field_tags>(
      [&](auto... packaged_tags) {
        tmpl::as_pack<argument_tags<Correction>>([&](auto... arg_tags) {
          correction.dg_package_data(
              make_not_null(&get<tmpl::type_from<decltype(packaged_tags)>>(
                  packaged_data))...,
              get<tmpl::type_from<decltype(arg_tags)>>(args)...,
              normal_covector, normal_vector, mesh_velocity,
              normal_dot_mesh_velocity, eos);
        });
      });
  return packaged_data;
}

template <typename Correction, typename PackagedTags>
Variables<evolved_tags> boundary_terms(const Correction& correction,
                                       const Variables<PackagedTags>& interior,
                                       const Variables<PackagedTags>& exterior,
                                       const dg::Formulation dg_formulation) {
  Variables<evolved_tags> result{num_points};
  tmpl::as_pack<PackagedTags>([&](auto... packaged_tags) {
    tmpl::as_pack<evolved_tags>([&](auto... result_tags) {
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

// The packaged data and boundary terms computed on the device must agree with
// the ones computed on the host
template <typename Correction>
void check_against_host(const gsl::not_null<std::mt19937*> generator,
                        const Correction& correction,
                        const EoS::EquationOfState<true, 3>& eos) {
  const BoundaryCorrections::OnDevice<Correction> correction_on_device{
      correction};
  // The device code may be compiled differently (e.g. with fused
  // multiply-adds)
  const Approx custom_approx = Approx::custom().epsilon(1.e-14).scale(1.0);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto normal_covector =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist), DataVector{num_points});

  for (const bool moving_mesh : {false, true}) {
    CAPTURE(moving_mesh);
    std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity{};
    if (moving_mesh) {
      mesh_velocity =
          make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
              generator, make_not_null(&dist), DataVector{num_points});
    }
    const auto interior_args = make_arguments<Correction>(generator);
    const auto exterior_args = make_arguments<Correction>(generator);
    const auto interior = package_data<Correction>(
        correction, interior_args, normal_covector, mesh_velocity, eos);
    const auto exterior = package_data<Correction>(
        correction, exterior_args, normal_covector, mesh_velocity, eos);
    CHECK_VARIABLES_CUSTOM_APPROX(
        package_data<Correction>(correction_on_device, interior_args,
                                 normal_covector, mesh_velocity, eos),
        interior, custom_approx);

    for (const auto dg_formulation :
         {dg::Formulation::StrongInertial, dg::Formulation::WeakInertial}) {
      CAPTURE(dg_formulation);
      CHECK_VARIABLES_CUSTOM_APPROX(
          boundary_terms(correction_on_device, interior, exterior,
                         dg_formulation),
          boundary_terms(correction, interior, exterior, dg_formulation),
          custom_approx);
    }
  }
}

// EOSs without a `DeviceView` are an error rather than evaluated on the host
void check_error_without_device_view(
    const gsl::not_null<std::mt19937*> generator) {
  const EoS::Equilibrium3D<EoS::HybridEos<EoS::PolytropicFluid<true>>> eos{
      EoS::HybridEos<EoS::PolytropicFluid<true>>{
          EoS::PolytropicFluid<true>{100.0, 2.0}, 1.5}};
  const BoundaryCorrections::Hll hll{1.0e100, 1.0e-100};
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto normal_covector =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist), DataVector{num_points});
  CHECK_THROWS_WITH(
      package_data<BoundaryCorrections::Hll>(
          BoundaryCorrections::OnDevice<BoundaryCorrections::Hll>{hll},
          make_arguments<BoundaryCorrections::Hll>(generator), normal_covector,
          std::nullopt, eos),
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
  }
  check_error_without_device_view(make_not_null(&generator));
}
