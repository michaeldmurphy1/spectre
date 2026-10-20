// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariables.hpp"
#endif  // SPECTRE_KOKKOS

namespace grmhd::ValenciaDivClean {
namespace {
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.ComputeFluxes",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const size_t num_pts = 5;

  auto vars = make_with_random_values<Variables<
      tmpl::append<ComputeFluxes::return_tags, ComputeFluxes::argument_tags>>>(
      make_not_null(&gen), make_not_null(&dist), num_pts);

  Variables<ComputeFluxes::return_tags> expected_fluxes{num_pts};
  ComputeFluxes::apply(
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeYe,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<>,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<>,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      get<grmhd::ValenciaDivClean::Tags::TildeD>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildeS<>>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildePhi>(vars),
      get<gr::Tags::Lapse<DataVector>>(vars),
      get<gr::Tags::Shift<DataVector, 3>>(vars),
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars),
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars),
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(vars),
      get<hydro::Tags::Pressure<DataVector>>(vars),
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars),
      get<hydro::Tags::LorentzFactor<DataVector>>(vars),
      get<hydro::Tags::MagneticField<DataVector, 3>>(vars));

  subcell::compute_fluxes(make_not_null(&vars));

  tmpl::for_each<ComputeFluxes::return_tags>(
      [&expected_fluxes, &vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(vars), get<tag>(expected_fluxes));
      });

#ifdef SPECTRE_KOKKOS
  // A batch of elements, with a tag that isn't used by `ComputeFluxes` and
  // must be left unchanged
  using batch_tags =
      tmpl::append<ComputeFluxes::return_tags, ComputeFluxes::argument_tags,
                   tmpl::list<hydro::Tags::Temperature<DataVector>>>;
  const size_t num_elements = 3;
  for (const bool with_exec : {false, true}) {
    CAPTURE(with_exec);
    BatchedVariables<batch_tags, Kokkos::HostSpace> batch{num_pts,
                                                          num_elements};
    std::array<Variables<batch_tags>, num_elements> expected{};
    for (size_t e = 0; e < num_elements; ++e) {
      Variables<batch_tags> element{};
      set_data_ref_to_element(make_not_null(&element), batch, e);
      fill_with_random_values(make_not_null(&element), make_not_null(&gen),
                              make_not_null(&dist));
      gsl::at(expected, e) = element;
      subcell::compute_fluxes(make_not_null(&gsl::at(expected, e)));
    }
    if (with_exec) {
      subcell::compute_fluxes(Kokkos::DefaultExecutionSpace{},
                              make_not_null(&batch));
    } else {
      subcell::compute_fluxes(make_not_null(&batch));
    }
    for (size_t e = 0; e < num_elements; ++e) {
      CAPTURE(e);
      Variables<batch_tags> element{};
      set_data_ref_to_element(make_not_null(&element), batch, e);
      CHECK_VARIABLES_APPROX(element, gsl::at(expected, e));
    }
  }
#endif  // SPECTRE_KOKKOS
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
