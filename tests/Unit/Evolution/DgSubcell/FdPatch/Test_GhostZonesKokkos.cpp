// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
using evolution::dg::subcell::fd_patch::FdPatches;
using evolution::dg::subcell::fd_patch::GhostZones;

template <size_t Dim>
std::vector<Element<Dim>> patch_elements(
    const FdPatches<Dim>& patches, const size_t patch,
    const Domain<Dim>& domain,
    const std::vector<std::array<size_t, Dim>>& refinement) {
  std::vector<Element<Dim>> result{};
  for (const auto& id : patches.element_ids(patch)) {
    result.push_back(
        domain::create_initial_element(id, domain.blocks(), refinement));
  }
  return result;
}

// The value of a ghost variable in a cell, identified by its cell index in the
// block, wrapped periodically.
template <size_t Dim>
double value(const size_t variable, const std::array<long, Dim>& block_cell,
             const std::array<long, Dim>& cells_in_block) {
  double result = 1000.0 * static_cast<double>(variable);
  double scale = 1.0;
  for (size_t d = 0; d < Dim; ++d) {
    const long wrapped = (gsl::at(block_cell, d) + gsl::at(cells_in_block, d)) %
                         gsl::at(cells_in_block, d);
    result += scale * static_cast<double>(wrapped);
    scale *= static_cast<double>(gsl::at(cells_in_block, d));
  }
  return result;
}

// Fills the ghost variables of a patch with `value`, applies
// `fill_from_patch`, and checks the interior and all ghost zones whose
// neighbor is in the patch. Returns the faces whose ghost zones were checked.
template <size_t Dim>
size_t check_fill_from_patch(const GhostZones<Dim>& ghost_zones,
                             const FdPatches<Dim>& patches, const size_t patch,
                             const std::array<long, Dim>& cells_in_block) {
  const Kokkos::DefaultExecutionSpace exec{};
  const auto& element_ids = patches.element_ids(patch);
  const size_t number_of_elements = element_ids.size();
  const size_t number_of_variables = ghost_zones.number_of_variables();
  const Index<Dim>& extents = ghost_zones.subcell_extents();
  const auto ghost_width = static_cast<long>(ghost_zones.ghost_width());
  const size_t number_of_cells = extents.product();

  // The cell index in the block of a cell of an element
  const auto block_cell = [&extents](const ElementId<Dim>& id,
                                     const std::array<long, Dim>& cell) {
    std::array<long, Dim> result{};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(result, d) =
          static_cast<long>(id.segment_id(d).index() * extents[d]) +
          gsl::at(cell, d);
    }
    return result;
  };

  const Kokkos::View<double*>::host_mirror_type host_ghost_variables(
      "ghost variables",
      number_of_elements * number_of_variables * number_of_cells);
  for (size_t e = 0; e < number_of_elements; ++e) {
    for (size_t v = 0; v < number_of_variables; ++v) {
      for (size_t i = 0; i < number_of_cells; ++i) {
        std::array<long, Dim> cell{};
        size_t remainder = i;
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(cell, d) = static_cast<long>(remainder % extents[d]);
          remainder /= extents[d];
        }
        host_ghost_variables((e * number_of_variables + v) * number_of_cells +
                             i) =
            value(v, block_cell(element_ids[e], cell), cells_in_block);
      }
    }
  }
  const auto ghost_variables = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space{}, host_ghost_variables);
  const auto padded_vars = fd::device::make_volume_view(
      exec, "padded", number_of_elements, number_of_variables, extents,
      ghost_zones.ghost_width());
  ghost_zones.fill_from_patch(exec, padded_vars, ghost_variables);
  const auto host_padded =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, padded_vars);

  // The value at `cell` of element `e`, where `cell` may be in a ghost zone
  const auto padded_value = [&](const size_t e, const size_t v,
                                const std::array<long, Dim>& cell) {
    std::array<size_t, 3> index{{0, 0, 0}};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(index, d) = static_cast<size_t>(gsl::at(cell, d) + ghost_width);
    }
    return host_padded(e, v, index[2], index[1], index[0]);
  };

  size_t number_of_checked_faces = 0;
  for (size_t e = 0; e < number_of_elements; ++e) {
    for (size_t v = 0; v < number_of_variables; ++v) {
      for (size_t i = 0; i < number_of_cells; ++i) {
        std::array<long, Dim> cell{};
        size_t remainder = i;
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(cell, d) = static_cast<long>(remainder % extents[d]);
          remainder /= extents[d];
        }
        CHECK(padded_value(e, v, cell) ==
              value(v, block_cell(element_ids[e], cell), cells_in_block));
        // The ghost cells next to the face of each boundary cell
        for (const auto& direction : Direction<Dim>::all_directions()) {
          const size_t d = direction.dimension();
          const bool upper = direction.side() == Side::Upper;
          const auto n = static_cast<long>(extents[d]);
          if ((upper and gsl::at(cell, d) < n - ghost_width) or
              (not upper and gsl::at(cell, d) >= ghost_width)) {
            continue;
          }
          bool external = false;
          for (const auto& face : ghost_zones.external_faces()) {
            external =
                external or (face.slot == e and face.direction == direction);
          }
          if (external) {
            continue;
          }
          auto ghost_cell = cell;
          gsl::at(ghost_cell, d) += upper ? ghost_width : -ghost_width;
          CHECK(
              padded_value(e, v, ghost_cell) ==
              value(v, block_cell(element_ids[e], ghost_cell), cells_in_block));
        }
      }
    }
    for (const auto& direction : Direction<Dim>::all_directions()) {
      bool external = false;
      for (const auto& face : ghost_zones.external_faces()) {
        external = external or (face.slot == e and face.direction == direction);
      }
      if (not external) {
        ++number_of_checked_faces;
      }
    }
  }
  return number_of_checked_faces;
}

// A single periodic block, so all faces are inside the patch. With refinement
// level 0 each element is its own neighbor.
template <size_t Dim>
void test_periodic_block() {
  const Kokkos::DefaultExecutionSpace exec{};
  std::array<std::vector<double>, Dim> demarcations{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(demarcations, d) = {0.0, 1.0};
  }
  const auto domain = rectilinear_domain<Dim>(Index<Dim>{1}, demarcations, {},
                                              {}, make_array<Dim>(true));
  for (const size_t refinement_level : {0_st, 1_st}) {
    CAPTURE(refinement_level);
    const std::vector refinement{make_array<Dim>(refinement_level)};
    // 4 DG points, so 7 subcells
    const std::vector extents{make_array<Dim>(4_st)};
    const FdPatches<Dim> patches{domain.blocks(),
                                 refinement,
                                 extents,
                                 Spectral::Quadrature::GaussLobatto,
                                 {},
                                 false};
    REQUIRE(patches.number_of_patches() == 1);
    const auto elements = patch_elements(patches, 0, domain, refinement);
    for (const size_t ghost_width : {1_st, 2_st}) {
      CAPTURE(ghost_width);
      const GhostZones<Dim> ghost_zones{exec, elements,    patches,
                                        0,    ghost_width, 2};
      CHECK(ghost_zones.subcell_extents() == Index<Dim>{7});
      CHECK(ghost_zones.external_faces().empty());
      const auto cells_in_block =
          make_array<Dim>(static_cast<long>(7 * two_to_the(refinement_level)));
      CHECK(check_fill_from_patch(ghost_zones, patches, 0, cells_in_block) ==
            elements.size() * 2 * Dim);
    }
  }
}

// Three periodic blocks in a row, each with two elements, so each element
// has a neighbor in its own patch and one in another patch.
void test_external_faces() {
  const Kokkos::DefaultExecutionSpace exec{};
  const auto domain = rectilinear_domain<1>(
      Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
      {}, {}, std::array{true});
  const std::vector<std::array<size_t, 1>> refinement(3, {{1}});
  const std::vector<std::array<size_t, 1>> extents(3, {{4}});
  const FdPatches<1> patches{domain.blocks(),
                             refinement,
                             extents,
                             Spectral::Quadrature::GaussLobatto,
                             {},
                             false};
  REQUIRE(patches.number_of_patches() == 3);
  const auto elements = patch_elements(patches, 1, domain, refinement);
  const GhostZones<1> ghost_zones{exec, elements, patches, 1, 2, 3};
  const auto& external_faces = ghost_zones.external_faces();
  REQUIRE(external_faces.size() == 2);
  CHECK(external_faces[0].slot == 0);
  CHECK(external_faces[0].direction == Direction<1>::lower_xi());
  CHECK(external_faces[0].neighbor == ElementId<1>{0, {{SegmentId{1, 1}}}});
  CHECK(external_faces[1].slot == 1);
  CHECK(external_faces[1].direction == Direction<1>::upper_xi());
  CHECK(external_faces[1].neighbor == ElementId<1>{2, {{SegmentId{1, 0}}}});
  // Only the face between the two elements is filled
  CHECK(check_fill_from_patch(ghost_zones, patches, 1, std::array{14L}) == 2);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.GhostZonesKokkos",
                  "[Evolution][Unit]") {
  test_periodic_block<1>();
  test_periodic_block<2>();
  test_periodic_block<3>();
  test_external_faces();
}
