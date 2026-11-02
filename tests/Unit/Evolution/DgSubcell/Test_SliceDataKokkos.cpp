// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SliceDataKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// `slice_data_on_device` gives the same result as `slice_data`, with copies,
// an interpolated direction, a direction left for DG interpolation, and an
// additional buffer
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator,
          const bool use_interpolants) {
  CAPTURE(Dim);
  CAPTURE(use_interpolants);
  constexpr size_t number_of_components = 3;
  constexpr size_t ghost_points = 2;
  constexpr size_t additional_buffer = 4;
  const Mesh<Dim> mesh{6, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto volume = make_with_random_values<DataVector>(
      generator, make_not_null(&dist),
      DataVector(number_of_components * mesh.number_of_grid_points()));

  std::unordered_set<Direction<Dim>> directions{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    directions.insert(direction);
  }
  DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>> interpolants{};
  if (use_interpolants) {
    const size_t slice_points =
        ghost_points * mesh.extents().slice_away(0).product();
    tnsr::I<DataVector, Dim, Frame::ElementLogical> target_points{slice_points};
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t i = 0; i < slice_points; ++i) {
        target_points.get(d)[i] = 0.9 * dist(*generator);
      }
    }
    interpolants[DirectionalId<Dim>{Direction<Dim>::upper_xi(),
                                    ElementId<Dim>{1}}] =
        intrp::Irregular<Dim>{mesh, target_points};
    interpolants[DirectionalId<Dim>{Direction<Dim>::lower_xi(),
                                    ElementId<Dim>{2}}] = std::nullopt;
  }

  const auto expected = evolution::dg::subcell::slice_data(
      volume, mesh.extents(), ghost_points, directions, additional_buffer,
      interpolants);
  const auto result = evolution::dg::subcell::slice_data_on_device(
      Kokkos::DefaultExecutionSpace{}, volume, mesh.extents(), ghost_points,
      directions, additional_buffer, interpolants);
  REQUIRE(result.size() == expected.size());
  for (const auto& [direction, expected_data] : expected) {
    CAPTURE(direction);
    const auto& result_data = result.at(direction);
    REQUIRE(result_data.size() == expected_data.size());
    // The additional buffer, and a direction left for DG interpolation, are
    // not filled
    if (use_interpolants and direction == Direction<Dim>::lower_xi()) {
      continue;
    }
    for (size_t i = 0; i < expected_data.size() - additional_buffer; ++i) {
      CAPTURE(i);
      CHECK(result_data[i] == approx(expected_data[i]));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SliceData.Kokkos",
                  "[Evolution][Unit]") {
  MAKE_GENERATOR(generator);
  for (const bool use_interpolants : {false, true}) {
    test<1>(make_not_null(&generator), use_interpolants);
    test<2>(make_not_null(&generator), use_interpolants);
    test<3>(make_not_null(&generator), use_interpolants);
  }
}
