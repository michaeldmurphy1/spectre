// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/ZCurve.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Evolution/DgSubcell/SubcellAllowed.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Literals.hpp"

namespace {
using evolution::dg::subcell::fd_patch::FdPatches;

// Every element of the domain is in a patch exactly when it starts on the
// subcell grid, and its location matches its position in the patch.
template <size_t Dim>
void check_consistency(
    const FdPatches<Dim>& patches, const Domain<Dim>& domain,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::vector<size_t>& only_dg_block_ids,
    const bool subcell_enabled_at_external_boundary) {
  size_t number_of_fd_elements = 0;
  for (const auto& element_id :
       initial_element_ids(initial_refinement_levels)) {
    const auto element = domain::create_initial_element(
        element_id, domain.blocks(), initial_refinement_levels);
    const auto dg_mesh = domain::create_initial_mesh(
        initial_extents, element, Spectral::Basis::Legendre,
        Spectral::Quadrature::GaussLobatto);
    const bool is_fd = evolution::dg::subcell::starts_on_subcell_grid(
        element, dg_mesh, only_dg_block_ids,
        subcell_enabled_at_external_boundary);
    const auto location = patches.location(element_id);
    CHECK(location.has_value() == is_fd);
    if (location.has_value()) {
      ++number_of_fd_elements;
      CHECK(patches.element_ids(location->patch).at(location->slot) ==
            element_id);
      CHECK(patches.dg_mesh(location->patch) == dg_mesh);
    }
  }
  CHECK(patches.number_of_elements() == number_of_fd_elements);
  CHECK(patches.number_of_dg_elements() ==
        initial_element_ids(initial_refinement_levels).size() -
            number_of_fd_elements);
  for (size_t patch = 0; patch < patches.number_of_patches(); ++patch) {
    const auto& element_ids = patches.element_ids(patch);
    CHECK_FALSE(element_ids.empty());
    // In the order of the blocks
    CHECK(std::is_sorted(
        element_ids.begin(), element_ids.end(),
        [](const ElementId<Dim>& lhs, const ElementId<Dim>& rhs) {
          return lhs.block_id() < rhs.block_id();
        }));
  }
  test_serialization(patches);
  test_copy_semantics(patches);
}

// 3x3 blocks with DG-only outer blocks, mimicking the BNS cubes surrounded by
// DG-only wedges: only the interior elements of the central block are FD.
void test_dg_only_surrounding_blocks() {
  const auto domain = rectilinear_domain<2>(
      Index<2>{3, 3}, std::array<std::vector<double>, 2>{
                          {{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 2>> refinement(9, {{2, 2}});
  const std::vector<std::array<size_t, 2>> extents(9, {{5, 5}});
  const std::vector<size_t> only_dg_block_ids{0, 1, 2, 3, 5, 6, 7, 8};
  const FdPatches<2> patches{
      domain.blocks(),   refinement,
      extents,           Spectral::Quadrature::GaussLobatto,
      only_dg_block_ids, true};
  REQUIRE(patches.number_of_patches() == 1);
  CHECK(patches.dg_mesh(0) == Mesh<2>{5, Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto});
  const std::vector<ElementId<2>> expected_element_ids{
      ElementId<2>{4, {{SegmentId{2, 1}, SegmentId{2, 1}}}},
      ElementId<2>{4, {{SegmentId{2, 1}, SegmentId{2, 2}}}},
      ElementId<2>{4, {{SegmentId{2, 2}, SegmentId{2, 1}}}},
      ElementId<2>{4, {{SegmentId{2, 2}, SegmentId{2, 2}}}}};
  CHECK(patches.element_ids(0) == expected_element_ids);
  for (size_t slot = 0; slot < expected_element_ids.size(); ++slot) {
    CHECK(patches.location(expected_element_ids[slot]) ==
          FdPatches<2>::Location{0, slot});
  }
  CHECK_FALSE(
      patches.location(ElementId<2>{4, {{SegmentId{2, 0}, SegmentId{2, 1}}}})
          .has_value());
  CHECK_FALSE(
      patches.location(ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{2, 1}}}})
          .has_value());
  check_consistency(patches, domain, refinement, extents, only_dg_block_ids,
                    true);
}

// Without DG-only blocks, elements on the external boundary are FD only if
// subcell is enabled there.
void test_external_boundary() {
  const auto domain = rectilinear_domain<2>(
      Index<2>{3, 3}, std::array<std::vector<double>, 2>{
                          {{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 2>> refinement(9, {{1, 1}});
  const std::vector<std::array<size_t, 2>> extents(9, {{4, 4}});
  for (const bool enabled_at_external_boundary : {true, false}) {
    CAPTURE(enabled_at_external_boundary);
    const FdPatches<2> patches{domain.blocks(),
                               refinement,
                               extents,
                               Spectral::Quadrature::GaussLobatto,
                               {},
                               enabled_at_external_boundary};
    // All FD elements in one patch, across the blocks
    CHECK(patches.number_of_patches() == 1);
    // 4 elements per block. Without subcell at the external boundary, corner
    // blocks keep 1 element, edge blocks 2 and the central block 4.
    CHECK(patches.number_of_elements() ==
          (enabled_at_external_boundary ? 36 : 16));
    check_consistency(patches, domain, refinement, extents, {},
                      enabled_at_external_boundary);
  }
}

// A block that isn't DG-only but whose elements all border DG-only blocks gets
// no patch.
void test_block_without_fd_elements() {
  const auto domain = rectilinear_domain<1>(
      Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 1>> refinement(3, {{0}});
  const std::vector<std::array<size_t, 1>> extents(3, {{5}});
  const std::vector<size_t> only_dg_block_ids{0, 2};
  const FdPatches<1> patches{
      domain.blocks(),   refinement,
      extents,           Spectral::Quadrature::GaussLobatto,
      only_dg_block_ids, true};
  CHECK(patches.number_of_patches() == 0);
  CHECK(patches.number_of_elements() == 0);
  check_consistency(patches, domain, refinement, extents, only_dg_block_ids,
                    true);
}

// The FD elements of several blocks split into several patches: each patch is
// a consecutive piece of the order of the blocks and their Z-curves, the pieces
// differ in size by at most one, and a patch keeps the order of the blocks and
// of `initial_element_ids()`.
void test_split_blocks() {
  const auto domain = rectilinear_domain<2>(
      Index<2>{2, 1},
      std::array<std::vector<double>, 2>{{{0.0, 1.0, 2.0}, {0.0, 1.0}}});
  // 16 elements in block 0, 8 in block 1
  const std::vector<std::array<size_t, 2>> refinement{{{2, 2}}, {{1, 2}}};
  const std::vector<std::array<size_t, 2>> extents(2, {{4, 4}});
  // The order of the FD elements along the Z-curves of the blocks
  std::vector<ElementId<2>> z_curve_order{};
  for (const size_t block : {0_st, 1_st}) {
    auto block_ids = initial_element_ids(block, refinement[block]);
    std::sort(block_ids.begin(), block_ids.end(),
              [](const ElementId<2>& lhs, const ElementId<2>& rhs) {
                return domain::z_curve_index(lhs) < domain::z_curve_index(rhs);
              });
    z_curve_order.insert(z_curve_order.end(), block_ids.begin(),
                         block_ids.end());
  }
  const size_t number_of_elements = z_curve_order.size();
  for (const size_t max_elements :
       {1_st, 3_st, 4_st, 5_st, 7_st, 16_st, 24_st, 100_st}) {
    CAPTURE(max_elements);
    const FdPatches<2> patches{domain.blocks(),
                               refinement,
                               extents,
                               Spectral::Quadrature::GaussLobatto,
                               {},
                               true,
                               max_elements};
    check_consistency(patches, domain, refinement, extents, {}, true);
    const size_t expected_number_of_patches =
        (number_of_elements + max_elements - 1) / max_elements;
    REQUIRE(patches.number_of_patches() == expected_number_of_patches);
    size_t first = 0;
    for (size_t patch = 0; patch < patches.number_of_patches(); ++patch) {
      CAPTURE(patch);
      const auto& element_ids = patches.element_ids(patch);
      CHECK(element_ids.size() <= max_elements);
      CHECK(element_ids.size() >=
            number_of_elements / expected_number_of_patches);
      CHECK(element_ids.size() <=
            (number_of_elements + expected_number_of_patches - 1) /
                expected_number_of_patches);
      CHECK(std::is_sorted(element_ids.begin(), element_ids.end()));
      // The next consecutive piece of the Z-curve order
      std::vector<ElementId<2>> expected_piece(
          std::next(z_curve_order.begin(), static_cast<std::ptrdiff_t>(first)),
          std::next(z_curve_order.begin(),
                    static_cast<std::ptrdiff_t>(first + element_ids.size())));
      std::sort(expected_piece.begin(), expected_piece.end());
      CHECK(element_ids == expected_piece);
      first += element_ids.size();
    }
    CHECK(first == number_of_elements);
  }
  // Block 0 in 2 x 2 quadrants of 4 elements each, and block 1 (a column of 4
  // elements, conforming to block 0) in one patch
  const FdPatches<2> quadrants{domain.blocks(),
                               {{{2, 2}}, {{0, 2}}},
                               extents,
                               Spectral::Quadrature::GaussLobatto,
                               {},
                               true,
                               4};
  REQUIRE(quadrants.number_of_patches() == 5);
  CHECK(quadrants.element_ids(0) ==
        std::vector<ElementId<2>>{
            ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{2, 0}}}},
            ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{2, 1}}}},
            ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{2, 0}}}},
            ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{2, 1}}}}});
  CHECK(quadrants.element_ids(4) ==
        std::vector<ElementId<2>>{
            ElementId<2>{1, {{SegmentId{0, 0}, SegmentId{2, 0}}}},
            ElementId<2>{1, {{SegmentId{0, 0}, SegmentId{2, 1}}}},
            ElementId<2>{1, {{SegmentId{0, 0}, SegmentId{2, 2}}}},
            ElementId<2>{1, {{SegmentId{0, 0}, SegmentId{2, 3}}}}});
  // A patch across the blocks: the last quadrant of block 0 and block 1
  const FdPatches<2> across{domain.blocks(),
                            {{{2, 2}}, {{0, 2}}},
                            extents,
                            Spectral::Quadrature::GaussLobatto,
                            {},
                            true,
                            10};
  REQUIRE(across.number_of_patches() == 2);
  CHECK(across.element_ids(1).size() == 10);
  CHECK(across.element_ids(1).front().block_id() == 0);
  CHECK(across.element_ids(1).back().block_id() == 1);
  CHECK_THROWS_WITH(
      (FdPatches<2>{domain.blocks(),
                    refinement,
                    extents,
                    Spectral::Quadrature::GaussLobatto,
                    {},
                    true,
                    0}),
      Catch::Matchers::ContainsSubstring(
          "The maximum number of elements per FD patch must be positive"));
}

// Elements with different DG meshes are in different patches, numbered by
// the first element with each mesh
void test_different_meshes() {
  const auto domain = rectilinear_domain<1>(
      Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 1>> refinement(3, {{1}});
  const std::vector<std::array<size_t, 1>> extents{{{5}}, {{6}}, {{5}}};
  for (const std::optional<size_t> max_elements :
       {std::optional<size_t>{}, std::optional<size_t>{3}}) {
    const FdPatches<1> patches{domain.blocks(),
                               refinement,
                               extents,
                               Spectral::Quadrature::GaussLobatto,
                               {},
                               true,
                               max_elements};
    check_consistency(patches, domain, refinement, extents, {}, true);
    const Mesh<1> mesh_5{5, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const Mesh<1> mesh_6{6, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    if (max_elements.has_value()) {
      // Blocks 0 and 2 in 2 patches of 2, block 1 in one
      REQUIRE(patches.number_of_patches() == 3);
      CHECK(patches.dg_mesh(0) == mesh_5);
      CHECK(patches.dg_mesh(1) == mesh_5);
      CHECK(patches.dg_mesh(2) == mesh_6);
      CHECK(patches.element_ids(0).front().block_id() == 0);
      CHECK(patches.element_ids(1).front().block_id() == 2);
    } else {
      REQUIRE(patches.number_of_patches() == 2);
      CHECK(patches.dg_mesh(0) == mesh_5);
      CHECK(patches.dg_mesh(1) == mesh_6);
      CHECK(patches.element_ids(0).size() == 4);
      CHECK(patches.element_ids(1).size() == 2);
    }
  }
}

void test_patch_procs() {
  using evolution::dg::subcell::fd_patch::patch_procs;
  CHECK(patch_procs(0, {}).empty());
  // One node: round-robin over its processors
  CHECK(patch_procs(5, {{0, 1, 2}}) == std::vector<size_t>{0, 1, 2, 0, 1});
  // Two nodes (e.g. one per GPU): first over the nodes
  CHECK(patch_procs(4, {{0, 1, 2}, {3, 4, 5}}) ==
        std::vector<size_t>{0, 3, 1, 4});
  // Nodes without usable processors are skipped
  CHECK(patch_procs(4, {{}, {3, 5}, {7}}) == std::vector<size_t>{3, 7, 5, 7});
  CHECK_THROWS_WITH(patch_procs(1, {{}, {}}),
                    Catch::Matchers::ContainsSubstring(
                        "No processor may hold the 1 FD patches"));
}

template <bool EnabledAtExternalBoundary>
struct Metavariables {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary =
        EnabledAtExternalBoundary;
  };
};

void test_tag() {
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::fd_patch::Tags::FdPatches<1>>("FdPatches");
  const auto make_interval = [](const bool periodic) {
    return std::make_unique<domain::creators::Interval>(
        std::array{0.0}, std::array{1.0}, std::array{size_t{2}},
        std::array{size_t{5}}, std::array{periodic});
  };
  using tag = evolution::dg::subcell::fd_patch::Tags::FdPatches<1>;
  // Periodic: no external boundaries, so all 4 elements are FD.
  CHECK(tag::create_from_options<Metavariables<false>>(
            make_interval(true), std::nullopt,
            Spectral::Quadrature::GaussLobatto, std::nullopt, false)
            .number_of_elements() == 4);
  // The 2 elements at the ends are on the external boundary.
  CHECK(tag::create_from_options<Metavariables<false>>(
            make_interval(false), std::nullopt,
            Spectral::Quadrature::GaussLobatto, std::nullopt, false)
            .number_of_elements() == 2);
  CHECK(tag::create_from_options<Metavariables<true>>(
            make_interval(false), std::nullopt,
            Spectral::Quadrature::GaussLobatto, std::nullopt, false)
            .number_of_elements() == 4);
  // Split into patches of at most 3 elements
  CHECK(tag::create_from_options<Metavariables<true>>(
            make_interval(false), std::nullopt,
            Spectral::Quadrature::GaussLobatto, 3, false)
            .number_of_patches() == 2);
  // The DG elements may avoid the processors of the patches
  CHECK_FALSE(tag::create_from_options<Metavariables<true>>(
                  make_interval(false), std::nullopt,
                  Spectral::Quadrature::GaussLobatto, std::nullopt, false)
                  .dedicated_procs());
  const auto dedicated = tag::create_from_options<Metavariables<true>>(
      make_interval(false), std::nullopt, Spectral::Quadrature::GaussLobatto,
      std::nullopt, true);
  CHECK(dedicated.dedicated_procs());
  test_serialization(dedicated);
  CHECK(dedicated != tag::create_from_options<Metavariables<true>>(
                         make_interval(false), std::nullopt,
                         Spectral::Quadrature::GaussLobatto, std::nullopt,
                         false));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.FdPatches",
                  "[Evolution][Unit]") {
  test_dg_only_surrounding_blocks();
  test_external_boundary();
  test_block_without_fd_elements();
  test_split_blocks();
  test_different_meshes();
  test_patch_procs();
  test_tag();
}
