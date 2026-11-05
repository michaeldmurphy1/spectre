// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborRdmpAndVolumeData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
using Interpolants =
    DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;

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

// Fills the ghost zones of the external faces from random received data and
// checks them against the ghost data computed on the host.
template <size_t Dim>
void check_fill_from_external(const GhostZones<Dim>& ghost_zones,
                              const std::vector<Element<Dim>>& elements,
                              const Mesh<Dim>& dg_mesh,
                              const gsl::not_null<std::mt19937*> generator) {
  const Kokkos::DefaultExecutionSpace exec{};
  const size_t number_of_variables = ghost_zones.number_of_variables();
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t ghost_width = ghost_zones.ghost_width();
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto host_external_data = make_with_random_values<DataVector>(
      generator, make_not_null(&dist),
      DataVector(ghost_zones.external_data_size()));
  const auto external_data = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space{},
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          host_external_data.data(), host_external_data.size()));
  const auto padded_vars = fd::device::make_volume_view(
      exec, "padded", elements.size(), number_of_variables,
      subcell_mesh.extents(), ghost_width);
  ghost_zones.fill_from_external(exec, padded_vars, external_data);
  const auto host_padded =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, padded_vars);

  size_t expected_offset = 0;
  for (const auto& face : ghost_zones.external_faces()) {
    CAPTURE(face.neighbor);
    CAPTURE(face.direction);
    CHECK(face.data_offset == expected_offset);
    expected_offset += face.data_size;
    // The received data with one RDMP variable appended
    DataVector received(face.data_size + 2);
    std::copy_n(&host_external_data[face.data_offset], face.data_size,
                received.begin());
    received[face.data_size] = 1.0;
    received[face.data_size + 1] = -1.0;
    DirectionalIdMap<Dim, evolution::dg::subcell::GhostData> ghost_data{};
    const DirectionalId<Dim> directional_id{face.direction, face.neighbor};
    evolution::dg::subcell::detail::insert_or_update_neighbor_volume_data_impl<
        true>(make_not_null(&ghost_data), received, 1, directional_id,
              face.neighbor_is_dg ? dg_mesh : subcell_mesh, elements[face.slot],
              subcell_mesh, ghost_width, {}, {}, 0, 0);
    const DataVector& expected =
        ghost_data.at(directional_id).neighbor_ghost_data_for_reconstruction();
    const auto points = fd::device::padded_ghost_zone_points(
        subcell_mesh.extents(), ghost_width, face.direction.dimension(),
        face.direction.side() == Side::Upper);
    REQUIRE(expected.size() == number_of_variables * points.size());
    const size_t nx = host_padded.extent(4);
    const size_t ny = host_padded.extent(3);
    DataVector filled(expected.size());
    for (size_t v = 0; v < number_of_variables; ++v) {
      for (size_t i = 0; i < points.size(); ++i) {
        filled[v * points.size() + i] =
            host_padded(face.slot, v, points[i] / (nx * ny),
                        (points[i] / nx) % ny, points[i] % nx);
      }
    }
    CHECK_ITERABLE_APPROX(filled, expected);
  }
  CHECK(expected_offset == ghost_zones.external_data_size());
}

// Slices random ghost variables of the patch for the external faces and
// checks the result against `slice_data` on the host.
template <size_t Dim>
void check_slice_for_external(
    const GhostZones<Dim>& ghost_zones,
    const std::vector<Interpolants<Dim>>& interpolants,
    const gsl::not_null<std::mt19937*> generator) {
  const Kokkos::DefaultExecutionSpace exec{};
  const size_t number_of_variables = ghost_zones.number_of_variables();
  const Index<Dim>& extents = ghost_zones.subcell_extents();
  const size_t element_size = number_of_variables * extents.product();
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto host_ghost_variables = make_with_random_values<DataVector>(
      generator, make_not_null(&dist),
      DataVector(interpolants.size() * element_size));
  const auto ghost_variables = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space{},
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          host_ghost_variables.data(), host_ghost_variables.size()));
  const Kokkos::View<double*> send_data("send data",
                                        ghost_zones.send_data_size());
  ghost_zones.slice_for_external(exec, send_data, ghost_variables);
  const auto host_send_data =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, send_data);

  size_t expected_offset = 0;
  for (const auto& face : ghost_zones.external_faces()) {
    CAPTURE(face.neighbor);
    CAPTURE(face.direction);
    CHECK(face.send_offset == expected_offset);
    expected_offset += face.send_size;
    const DataVector volume_vars(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(&host_ghost_variables[face.slot * element_size]),
        element_size);
    const auto expected = evolution::dg::subcell::slice_data(
        volume_vars, extents, ghost_zones.ghost_width(), {face.direction}, 0,
        interpolants[face.slot]);
    REQUIRE(expected.at(face.direction).size() == face.send_size);
    const DataVector sent(&host_send_data(face.send_offset), face.send_size);
    CHECK_ITERABLE_APPROX(sent, expected.at(face.direction));
  }
  CHECK(expected_offset == ghost_zones.send_data_size());
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
      const GhostZones<Dim> ghost_zones{
          exec,    elements, std::vector<Interpolants<Dim>>(elements.size()),
          patches, 0,        ghost_width,
          2};
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
void test_external_faces(const gsl::not_null<std::mt19937*> generator) {
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
  // Interpolate the data sent to the lower neighbor to random points, and
  // copy the data sent to the upper neighbor.
  std::vector<Interpolants<1>> interpolants(2);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  interpolants[0][DirectionalId<1>{Direction<1>::lower_xi(),
                                   ElementId<1>{0, {{SegmentId{1, 1}}}}}] =
      intrp::Irregular<1>{
          evolution::dg::subcell::fd::mesh(patches.dg_mesh(1)),
          tnsr::I<DataVector, 1, Frame::ElementLogical>{
              make_with_random_values<DataVector>(
                  generator, make_not_null(&dist), DataVector(2))}};
  const GhostZones<1> ghost_zones{exec, elements, interpolants, patches, 1,
                                  2,    3};
  const auto& external_faces = ghost_zones.external_faces();
  REQUIRE(external_faces.size() == 2);
  CHECK(external_faces[0].slot == 0);
  CHECK(external_faces[0].direction == Direction<1>::lower_xi());
  CHECK(external_faces[0].neighbor == ElementId<1>{0, {{SegmentId{1, 1}}}});
  CHECK(external_faces[1].slot == 1);
  CHECK(external_faces[1].direction == Direction<1>::upper_xi());
  CHECK(external_faces[1].neighbor == ElementId<1>{2, {{SegmentId{1, 0}}}});
  CHECK_FALSE(external_faces[0].neighbor_is_dg);
  CHECK_FALSE(external_faces[1].neighbor_is_dg);
  // The ghost zone of 2 cells, for 3 variables
  CHECK(external_faces[0].data_size == 6);
  CHECK(external_faces[1].data_size == 6);
  // Only the face between the two elements is filled from the patch
  CHECK(check_fill_from_patch(ghost_zones, patches, 1, std::array{14L}) == 2);
  check_fill_from_external(ghost_zones, elements, patches.dg_mesh(1),
                           generator);
  CHECK(external_faces[0].send_size == 6);
  check_slice_for_external(ghost_zones, interpolants, generator);
}

// 3x3 blocks with DG-only outer blocks and 4x4 elements per block, so the
// 2x2 interior elements of the central block are FD and border DG elements of
// the same block.
void test_dg_neighbors(const gsl::not_null<std::mt19937*> generator) {
  const Kokkos::DefaultExecutionSpace exec{};
  const auto domain = rectilinear_domain<2>(
      Index<2>{3, 3}, std::array<std::vector<double>, 2>{
                          {{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 2>> refinement(9, {{2, 2}});
  const std::vector<std::array<size_t, 2>> extents(9, {{5, 5}});
  const FdPatches<2> patches{domain.blocks(),
                             refinement,
                             extents,
                             Spectral::Quadrature::GaussLobatto,
                             {0, 1, 2, 3, 5, 6, 7, 8},
                             true};
  REQUIRE(patches.number_of_patches() == 1);
  const auto elements = patch_elements(patches, 0, domain, refinement);
  for (const size_t ghost_width : {2_st, 3_st}) {
    CAPTURE(ghost_width);
    const std::vector<Interpolants<2>> interpolants(elements.size());
    const GhostZones<2> ghost_zones{
        exec, elements, interpolants, patches, 0, ghost_width, 2};
    // Each element has two DG neighbors
    REQUIRE(ghost_zones.external_faces().size() == 8);
    for (const auto& face : ghost_zones.external_faces()) {
      CHECK(face.neighbor_is_dg);
      CHECK(face.data_size == 2_st * 25_st);
    }
    check_fill_from_external(ghost_zones, elements, patches.dg_mesh(0),
                             generator);
    check_slice_for_external(ghost_zones, interpolants, generator);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.GhostZonesKokkos",
                  "[Evolution][Unit]") {
  test_periodic_block<1>();
  test_periodic_block<2>();
  test_periodic_block<3>();
  MAKE_GENERATOR(generator);
  test_external_faces(make_not_null(&generator));
  test_dg_neighbors(make_not_null(&generator));
}
