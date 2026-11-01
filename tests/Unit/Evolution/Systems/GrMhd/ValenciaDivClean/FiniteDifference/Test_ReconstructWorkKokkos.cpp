// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentralKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace fd = grmhd::ValenciaDivClean::fd;
// The primitives reconstructed by `MonotonisedCentralPrim`
using prims_to_reconstruct_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::Temperature<DataVector>,
               hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
               hydro::Tags::MagneticField<DataVector, 3>,
               hydro::Tags::DivergenceCleaningField<DataVector>>;
using FaceVars = std::array<Variables<fd::tags_list_for_reconstruct>, 3>;
// Monotonised central reconstruction needs 2 ghost cells
constexpr size_t ghost_zone_size = 2;

// `reconstruct_prims_work_on_device` reconstructs the same face values as
// `reconstruct_prims_work` from random data, with and without reconstructing
// rho T instead of T.
void test(const gsl::not_null<std::mt19937*> generator,
          const bool reconstruct_density_times_temperature) {
  CAPTURE(reconstruct_density_times_temperature);
  const Mesh<3> subcell_mesh{5, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const size_t volume_num_pts = subcell_mesh.number_of_grid_points();
  const size_t neighbor_num_pts =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();

  // The values only need to be positive, since no conservatives are computed
  std::uniform_real_distribution<double> dist(0.5, 2.0);
  const auto volume_prims =
      make_with_random_values<Variables<hydro::grmhd_tags<DataVector>>>(
          generator, make_not_null(&dist), DataVector(volume_num_pts));

  DirectionMap<3, Neighbors<3>> neighbors{};
  DirectionalIdMap<3, Variables<prims_to_reconstruct_tags>> neighbor_data{};
  for (size_t i = 0; i < 6; ++i) {
    const auto& direction = gsl::at(Direction<3>::all_directions(), i);
    const ElementId<3> neighbor_id{i + 1, {}};
    neighbors[direction] =
        Neighbors<3>{{neighbor_id}, OrientationMap<3>::create_aligned()};
    neighbor_data[DirectionalId<3>{direction, neighbor_id}] =
        make_with_random_values<Variables<prims_to_reconstruct_tags>>(
            generator, make_not_null(&dist), DataVector(neighbor_num_pts));
  }
  const Element<3> element{ElementId<3>{0, {}}, neighbors,
                           domain::topologies::hypercube<3>};
  const EquationsOfState::IdealFluid<true> eos{1.4};

  const auto make_face_vars = [&reconstructed_num_pts]() {
    FaceVars result{};
    for (auto& vars : result) {
      vars.initialize(reconstructed_num_pts);
    }
    return result;
  };
  FaceVars expected_lower = make_face_vars();
  FaceVars expected_upper = make_face_vars();
  fd::reconstruct_prims_work<prims_to_reconstruct_tags>(
      make_not_null(&expected_lower), make_not_null(&expected_upper),
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotonised_central(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      volume_prims, eos, element, neighbor_data, subcell_mesh, ghost_zone_size,
      false, reconstruct_density_times_temperature, nullptr);

  FaceVars lower = make_face_vars();
  FaceVars upper = make_face_vars();
  fd::reconstruct_prims_work_on_device<prims_to_reconstruct_tags>(
      make_not_null(&lower), make_not_null(&upper),
      [](const Kokkos::DefaultExecutionSpace& exec,
         const auto& upper_side_of_face_vars,
         const auto& lower_side_of_face_vars, const auto& padded_volume_vars,
         const Index<3>& volume_extents) {
        ::fd::device::monotonised_central(exec, upper_side_of_face_vars,
                                          lower_side_of_face_vars,
                                          padded_volume_vars, volume_extents);
      },
      volume_prims, eos, element, neighbor_data, subcell_mesh, ghost_zone_size,
      false, reconstruct_density_times_temperature, nullptr);

  for (size_t d = 0; d < 3; ++d) {
    CAPTURE(d);
    tmpl::for_each<prims_to_reconstruct_tags>([&](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      CAPTURE(db::tag_name<tag>());
      CHECK_ITERABLE_APPROX(get<tag>(gsl::at(lower, d)),
                            get<tag>(gsl::at(expected_lower, d)));
      CHECK_ITERABLE_APPROX(get<tag>(gsl::at(upper, d)),
                            get<tag>(gsl::at(expected_upper, d)));
    });
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Fd.ReconstructWork",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator), false);
  test(make_not_null(&generator), true);
}
