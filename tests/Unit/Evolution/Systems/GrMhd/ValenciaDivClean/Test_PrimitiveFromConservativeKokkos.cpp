// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Schemes = grmhd::ValenciaDivClean::PrimitiveRecoverySchemes;
using KastaunThenNewmanThenPalenzuela =
    tmpl::list<Schemes::KastaunEtAl, Schemes::NewmanHamlin,
               Schemes::PalenzuelaEtAl>;
using NewmanHamlinThenPalenzuelaEtAl =
    tmpl::list<Schemes::NewmanHamlin, Schemes::PalenzuelaEtAl>;

struct Conservatives {
  Scalar<DataVector> tilde_d;
  Scalar<DataVector> tilde_ye;
  Scalar<DataVector> tilde_tau;
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s;
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b;
  Scalar<DataVector> tilde_phi;
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric;
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric;
  Scalar<DataVector> sqrt_det_spatial_metric;
};

// Random physical states. The magnetic field is zero at every other point, so
// both `KastaunEtAl` and `KastaunEtAlHydro` are used. The rest mass density is
// tiny at every fifth point, so these points are atmosphere, and is scaled by
// `density_scale` elsewhere.
template <typename Eos>
Conservatives random_conservatives(const gsl::not_null<std::mt19937*> generator,
                                   const Eos& eos, const size_t num_points,
                                   const double density_scale = 1.0) {
  const DataVector used_for_size(num_points);
  std::uniform_real_distribution<> dist_density(0.1, 1.0);
  std::uniform_real_distribution<> dist_unit(0.0, 1.0);
  std::uniform_real_distribution<> dist_velocity(-0.4, 0.4);
  std::uniform_real_distribution<> dist_field(-1.0, 1.0);
  std::uniform_real_distribution<> dist_metric(0.8, 1.2);

  auto rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_density), used_for_size);
  get(rest_mass_density) *= density_scale;
  for (size_t i = 0; i < num_points; i += 5) {
    get(rest_mass_density)[i] = 1.e-14;
  }
  auto electron_fraction = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), used_for_size);
  get(electron_fraction) = 0.05 + 0.4 * get(electron_fraction);
  auto temperature = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist_unit), used_for_size);
  get(temperature) = 0.2 + get(temperature);
  const auto specific_internal_energy =
      eos.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  const auto pressure = eos.pressure_from_density_and_temperature(
      rest_mass_density, temperature, electron_fraction);
  auto magnetic_field =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_field), used_for_size);
  for (size_t i = 0; i < num_points; i += 2) {
    for (size_t d = 0; d < 3; ++d) {
      magnetic_field.get(d)[i] = 0.0;
    }
  }
  const auto divergence_cleaning_field =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist_field), used_for_size);

  Conservatives result{};
  result.spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  result.inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  result.sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  for (size_t d = 0; d < 3; ++d) {
    result.spatial_metric.get(d, d) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist_metric), used_for_size);
    result.inv_spatial_metric.get(d, d) = 1.0 / result.spatial_metric.get(d, d);
    get(result.sqrt_det_spatial_metric) *=
        sqrt(result.spatial_metric.get(d, d));
  }
  // The velocity is small enough that v^i v_i < 1 for these metrics
  const auto spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist_velocity), used_for_size);
  Scalar<DataVector> lorentz_factor{num_points, 0.0};
  for (size_t d = 0; d < 3; ++d) {
    get(lorentz_factor) +=
        result.spatial_metric.get(d, d) * square(spatial_velocity.get(d));
  }
  get(lorentz_factor) = 1.0 / sqrt(1.0 - get(lorentz_factor));

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&result.tilde_d), make_not_null(&result.tilde_ye),
      make_not_null(&result.tilde_tau), make_not_null(&result.tilde_s),
      make_not_null(&result.tilde_b), make_not_null(&result.tilde_phi),
      rest_mass_density, electron_fraction, specific_internal_energy, pressure,
      spatial_velocity, lorentz_factor, magnetic_field,
      result.sqrt_det_spatial_metric, result.spatial_metric,
      divergence_cleaning_field);
  return result;
}

using primitive_tags = hydro::grmhd_tags<DataVector>;
using conservative_tags =
    tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
               grmhd::ValenciaDivClean::Tags::TildeYe,
               grmhd::ValenciaDivClean::Tags::TildeTau,
               grmhd::ValenciaDivClean::Tags::TildeS<>,
               grmhd::ValenciaDivClean::Tags::TildeB<>,
               grmhd::ValenciaDivClean::Tags::TildePhi,
               gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>>;

// The initial values of the primitive variables. The recovery uses the pressure
// as the initial guess, and the failed grid points keep these values.
Variables<primitive_tags> initial_primitives(const size_t num_points) {
  Variables<primitive_tags> result{num_points, 0.0};
  get(get<hydro::Tags::RestMassDensity<DataVector>>(result)) = 0.5;
  get(get<hydro::Tags::ElectronFraction<DataVector>>(result)) = 0.1;
  get(get<hydro::Tags::SpecificInternalEnergy<DataVector>>(result)) = 1.0;
  get(get<hydro::Tags::LorentzFactor<DataVector>>(result)) = 1.0;
  get(get<hydro::Tags::Pressure<DataVector>>(result)) = 0.3;
  get(get<hydro::Tags::Temperature<DataVector>>(result)) = 1.0;
  return result;
}

// The conservative variables and metric of a batch of elements, and their
// primitive variables, in device memory
struct DeviceBatch {
  BatchedVariables<conservative_tags> conservatives;
  BatchedVariables<primitive_tags> primitives;

  template <typename Tag>
  const auto& view() const {
    return get<Tags::BatchView<Tag>>(conservatives);
  }
};

DeviceBatch to_device(const std::vector<Conservatives>& cons,
                      const Variables<primitive_tags>& primitives) {
  const size_t num_points = get(cons.front().tilde_d).size();
  const size_t num_elements = cons.size();
  BatchedVariables<conservative_tags, Kokkos::HostSpace> host_cons{
      num_points, num_elements};
  BatchedVariables<primitive_tags, Kokkos::HostSpace> host_prims{num_points,
                                                                 num_elements};
  for (size_t e = 0; e < num_elements; ++e) {
    Variables<conservative_tags> element_cons{};
    set_data_ref_to_element(make_not_null(&element_cons), host_cons, e);
    get<grmhd::ValenciaDivClean::Tags::TildeD>(element_cons) = cons[e].tilde_d;
    get<grmhd::ValenciaDivClean::Tags::TildeYe>(element_cons) =
        cons[e].tilde_ye;
    get<grmhd::ValenciaDivClean::Tags::TildeTau>(element_cons) =
        cons[e].tilde_tau;
    get<grmhd::ValenciaDivClean::Tags::TildeS<>>(element_cons) =
        cons[e].tilde_s;
    get<grmhd::ValenciaDivClean::Tags::TildeB<>>(element_cons) =
        cons[e].tilde_b;
    get<grmhd::ValenciaDivClean::Tags::TildePhi>(element_cons) =
        cons[e].tilde_phi;
    get<gr::Tags::SpatialMetric<DataVector, 3>>(element_cons) =
        cons[e].spatial_metric;
    get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(element_cons) =
        cons[e].inv_spatial_metric;
    get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(element_cons) =
        cons[e].sqrt_det_spatial_metric;
    Variables<primitive_tags> element_prims{};
    set_data_ref_to_element(make_not_null(&element_prims), host_prims, e);
    element_prims = primitives;
  }
  const Kokkos::DefaultExecutionSpace exec{};
  DeviceBatch result{{exec, num_points, num_elements},
                     {exec, num_points, num_elements}};
  deep_copy(exec, make_not_null(&result.conservatives), host_cons);
  deep_copy(exec, make_not_null(&result.primitives), host_prims);
  exec.fence();
  return result;
}

std::vector<Variables<primitive_tags>> primitives_to_host(
    const DeviceBatch& batch) {
  BatchedVariables<primitive_tags, Kokkos::HostSpace> host_prims{
      batch.primitives.number_of_grid_points(),
      batch.primitives.number_of_elements()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&host_prims), batch.primitives);
  exec.fence();
  std::vector<Variables<primitive_tags>> result{};
  for (size_t e = 0; e < host_prims.number_of_elements(); ++e) {
    Variables<primitive_tags> element_prims{};
    set_data_ref_to_element(make_not_null(&element_prims), host_prims, e);
    result.push_back(element_prims);
  }
  return result;
}

// Recovers the primitive variables of a batch of elements on the device
template <typename OrderedListOfSchemes, bool EnforcePhysicality>
PointwiseStatusSummary recover_on_device(
    const gsl::not_null<DeviceBatch*> batch,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions& options) {
  namespace VTags = grmhd::ValenciaDivClean::Tags;
  return grmhd::ValenciaDivClean::primitive_from_conservative_on_device<
      OrderedListOfSchemes, EnforcePhysicality>(
      Kokkos::DefaultExecutionSpace{}, make_not_null(&batch->primitives),
      batch->view<VTags::TildeD>(), batch->view<VTags::TildeYe>(),
      batch->view<VTags::TildeTau>(), batch->view<VTags::TildeS<>>(),
      batch->view<VTags::TildeB<>>(), batch->view<VTags::TildePhi>(),
      batch->view<gr::Tags::SpatialMetric<DataVector, 3>>(),
      batch->view<gr::Tags::InverseSpatialMetric<DataVector, 3>>(),
      batch->view<gr::Tags::SqrtDetSpatialMetric<DataVector>>(), eos, options);
}

// Recovers the primitives of an element on the host, starting from
// `primitives`. Returns whether the recovery succeeded.
template <typename OrderedListOfSchemes, bool EnforcePhysicality>
bool recover_on_host(
    const gsl::not_null<Variables<primitive_tags>*> primitives,
    const Conservatives& cons,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions& options) {
  return grmhd::ValenciaDivClean::
      PrimitiveFromConservative<OrderedListOfSchemes, false>::template apply<
          EnforcePhysicality>(
          make_not_null(
              &get<hydro::Tags::RestMassDensity<DataVector>>(*primitives)),
          make_not_null(
              &get<hydro::Tags::ElectronFraction<DataVector>>(*primitives)),
          make_not_null(&get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
              *primitives)),
          make_not_null(
              &get<hydro::Tags::SpatialVelocity<DataVector, 3>>(*primitives)),
          make_not_null(
              &get<hydro::Tags::MagneticField<DataVector, 3>>(*primitives)),
          make_not_null(&get<hydro::Tags::DivergenceCleaningField<DataVector>>(
              *primitives)),
          make_not_null(
              &get<hydro::Tags::LorentzFactor<DataVector>>(*primitives)),
          make_not_null(&get<hydro::Tags::Pressure<DataVector>>(*primitives)),
          make_not_null(
              &get<hydro::Tags::Temperature<DataVector>>(*primitives)),
          cons.tilde_d, cons.tilde_ye, cons.tilde_tau, cons.tilde_s,
          cons.tilde_b, cons.tilde_phi, cons.spatial_metric,
          cons.inv_spatial_metric, cons.sqrt_det_spatial_metric, eos, options);
}

// Recovers the primitives of each element on the host and of all elements as a
// batch on the device, and checks that they agree. Returns whether the
// recovery succeeded on every element.
template <typename OrderedListOfSchemes, bool EnforcePhysicality>
bool check_on_device(
    const std::vector<Conservatives>& cons,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions& options) {
  CAPTURE(pretty_type::get_name<OrderedListOfSchemes>());
  CAPTURE(EnforcePhysicality);
  const size_t num_points = get(cons.front().tilde_d).size();
  auto batch = to_device(cons, initial_primitives(num_points));
  const PointwiseStatusSummary status =
      recover_on_device<OrderedListOfSchemes, EnforcePhysicality>(
          make_not_null(&batch), eos, options);
  const auto device_prims = primitives_to_host(batch);
  bool all_succeeded = true;
  for (size_t e = 0; e < cons.size(); ++e) {
    CAPTURE(e);
    auto expected = initial_primitives(num_points);
    const bool host_succeeded =
        recover_on_host<OrderedListOfSchemes, EnforcePhysicality>(
            make_not_null(&expected), cons[e], eos, options);
    all_succeeded = all_succeeded and host_succeeded;
    if (not host_succeeded) {
      // The host stops at the first grid point where the recovery fails, so
      // the results differ
      continue;
    }
    // The root finds converge to a tolerance of ~1e-15, and the device code
    // may be compiled differently (e.g. with fused multiply-adds)
    const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
    CHECK_VARIABLES_CUSTOM_APPROX(device_prims[e], expected, custom_approx);
  }
  CHECK((status.number_of_failures == 0) == all_succeeded);
  return all_succeeded;
}

template <typename OrderedListOfSchemes>
void test_recovery(const gsl::not_null<std::mt19937*> generator,
                   const EquationsOfState::EquationOfState<true, 3>& eos) {
  CAPTURE(pretty_type::get_runtime_type_name(eos));
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions options{
      1.e-12, 1.e-12, 1.e4};
  const std::vector cons{random_conservatives(generator, eos, 50)};
  CHECK(check_on_device<OrderedListOfSchemes, true>(cons, eos, options));
  CHECK(check_on_device<OrderedListOfSchemes, false>(cons, eos, options));

  // The whole grid is below the cutoff, so it is all atmosphere
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      large_cutoff_options{10.0, 1.e-10, 1.e4};
  CHECK(check_on_device<OrderedListOfSchemes, true>(cons, eos,
                                                    large_cutoff_options));

  // Unphysical: all schemes fail if physicality isn't enforced
  auto unphysical_cons = cons;
  get(unphysical_cons[0].tilde_tau) = -10.0;
  CHECK_FALSE(check_on_device<OrderedListOfSchemes, false>(unphysical_cons, eos,
                                                           options));
  // Enforcing physicality, only some schemes recover some of these states, but
  // the host and device must agree (which is checked in `check_on_device`)
  check_on_device<OrderedListOfSchemes, true>(unphysical_cons, eos, options);

  // A batch of elements. The rest mass density of the middle element is so
  // small that its whole grid is below the cutoff for the inversion, while
  // that of the others isn't, so the cutoff must be decided for each element.
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      element_cutoff_options{1.e-3, 1.e-12, 1.e4};
  std::vector<Conservatives> batch_cons{};
  for (size_t e = 0; e < 3; ++e) {
    batch_cons.push_back(
        random_conservatives(generator, eos, 20, e == 1 ? 1.e-4 : 1.0));
  }
  REQUIRE(max(get(batch_cons[1].tilde_d)) <
          element_cutoff_options.cutoff_d_for_inversion());
  REQUIRE(max(get(batch_cons[0].tilde_d)) >
          element_cutoff_options.cutoff_d_for_inversion());
  CHECK(check_on_device<OrderedListOfSchemes, true>(batch_cons, eos,
                                                    element_cutoff_options));
}

// The recovery continues past failures, which are reported with their element
// and grid point. The primitive variables at the failed grid points keep their
// values from before the recovery.
void test_failures(const gsl::not_null<std::mt19937*> generator,
                   const EquationsOfState::EquationOfState<true, 3>& eos) {
  using Recovery = tmpl::list<Schemes::NewmanHamlin>;
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions options{
      1.e-12, 1.e-12, 1.e4};
  const size_t num_points = 10;
  std::vector<Conservatives> cons{
      random_conservatives(generator, eos, num_points),
      random_conservatives(generator, eos, num_points)};
  auto batch = to_device(cons, initial_primitives(num_points));
  CHECK(recover_on_device<Recovery, false>(make_not_null(&batch), eos, options)
            .number_of_failures == 0);
  const auto expected = primitives_to_host(batch);

  get(cons[1].tilde_tau)[3] = -10.0;
  get(cons[1].tilde_tau)[7] = -10.0;
  batch = to_device(cons, initial_primitives(num_points));
  const PointwiseStatusSummary status =
      recover_on_device<Recovery, false>(make_not_null(&batch), eos, options);
  CHECK(status.number_of_failures == 2);
  CHECK(status.first_failed_element == 1);
  CHECK(status.first_failed_grid_point == 3);
  const auto prims = primitives_to_host(batch);
  CHECK(prims[0] == expected[0]);
  const auto initial = initial_primitives(num_points);
  tmpl::for_each<primitive_tags>([&](auto tag_v) {
    using Tag = tmpl::type_from<decltype(tag_v)>;
    for (size_t component = 0; component < get<Tag>(prims[1]).size();
         ++component) {
      for (size_t i = 0; i < num_points; ++i) {
        CAPTURE(pretty_type::get_name<Tag>());
        CAPTURE(i);
        CHECK(get<Tag>(prims[1])[component][i] ==
              get<Tag>(i == 3 or i == 7 ? initial : expected[1])[component][i]);
      }
    }
  });
}

// A failure is reported with the variables at the first failed grid point and
// the primitive variables there from before the recovery
void test_error_on_failure(
    const gsl::not_null<std::mt19937*> generator,
    const EquationsOfState::EquationOfState<true, 3>& eos) {
  namespace VTags = grmhd::ValenciaDivClean::Tags;
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions options{
      1.e-12, 1.e-12, 1.e4};
  auto cons = random_conservatives(generator, eos, 10);
  // Every fifth grid point is atmosphere, which doesn't fail
  get(cons.tilde_tau) = -10.0;
  auto primitives = initial_primitives(10);
  get(get<hydro::Tags::LorentzFactor<DataVector>>(primitives)) = 5.0;
  auto batch = to_device({cons}, primitives);
  const PointwiseStatusSummary status =
      recover_on_device<tmpl::list<Schemes::NewmanHamlin>, false>(
          make_not_null(&batch), eos, options);
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::error_recovering_primitives_on_device(
          Kokkos::DefaultExecutionSpace{}, status, batch.primitives,
          batch.view<VTags::TildeD>(), batch.view<VTags::TildeTau>(),
          batch.view<VTags::TildeS<>>(), batch.view<VTags::TildeB<>>(),
          batch.view<gr::Tags::SpatialMetric<DataVector, 3>>(),
          batch.view<gr::Tags::InverseSpatialMetric<DataVector, 3>>(),
          batch.view<gr::Tags::SqrtDetSpatialMetric<DataVector>>(),
          "the element [B0,(L0I0)]"),
      Catch::Matchers::ContainsSubstring(
          "All primitive inversion schemes failed on the device at 8 grid "
          "points. The first is s = 1 of the element [B0,(L0I0)].") and
          Catch::Matchers::ContainsSubstring(
              "previous_lorentz_factor = 5.00000000000000000e+00") and
          Catch::Matchers::ContainsSubstring(
              "previous_rest_mass_density = 5.00000000000000000e-01"));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.GrMhd.ValenciaDivClean.PrimitiveFromConservative.Kokkos",
    "[Unit][GrMhd]") {
  MAKE_GENERATOR(generator);
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
      ideal_fluid{EquationsOfState::IdealFluid<true>{4.0 / 3.0, 0.0}};
  const EquationsOfState::Barotropic3D<EquationsOfState::PolytropicFluid<true>>
      polytrope{EquationsOfState::PolytropicFluid<true>{10.0, 2.0}};
  // A minimum temperature gives a nonzero lower bound of the specific internal
  // energy
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>>
      ideal_fluid_with_minimum_temperature{
          EquationsOfState::IdealFluid<true>{4.0 / 3.0, 1.e-3}};
  for (const auto* eos :
       {static_cast<const EquationsOfState::EquationOfState<true, 3>*>(
            &ideal_fluid),
        static_cast<const EquationsOfState::EquationOfState<true, 3>*>(
            &ideal_fluid_with_minimum_temperature),
        static_cast<const EquationsOfState::EquationOfState<true, 3>*>(
            &polytrope)}) {
    // The schemes and lists of schemes used by the subcell actions
    test_recovery<tmpl::list<Schemes::KastaunEtAl>>(make_not_null(&generator),
                                                    *eos);
    test_recovery<tmpl::list<Schemes::NewmanHamlin>>(make_not_null(&generator),
                                                     *eos);
    test_recovery<tmpl::list<Schemes::PalenzuelaEtAl>>(
        make_not_null(&generator), *eos);
    test_recovery<NewmanHamlinThenPalenzuelaEtAl>(make_not_null(&generator),
                                                  *eos);
    test_recovery<KastaunThenNewmanThenPalenzuela>(make_not_null(&generator),
                                                   *eos);
  }
  test_failures(make_not_null(&generator), ideal_fluid);
  test_error_on_failure(make_not_null(&generator), ideal_fluid);
}
