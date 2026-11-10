// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
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
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Evolution/DgSubcell/SubcellAllowed.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

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
      CHECK(patches.block_id(location->patch) == element_id.block_id());
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
    CHECK_FALSE(patches.element_ids(patch).empty());
    if (patch > 0) {
      CHECK(patches.block_id(patch) > patches.block_id(patch - 1));
    }
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
  CHECK(patches.block_id(0) == 4);
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
    CHECK(patches.number_of_patches() == 9);
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
  CHECK(
      tag::create_from_options<Metavariables<false>>(
          make_interval(true), std::nullopt, Spectral::Quadrature::GaussLobatto)
          .number_of_elements() == 4);
  // The 2 elements at the ends are on the external boundary.
  CHECK(tag::create_from_options<Metavariables<false>>(
            make_interval(false), std::nullopt,
            Spectral::Quadrature::GaussLobatto)
            .number_of_elements() == 2);
  CHECK(tag::create_from_options<Metavariables<true>>(
            make_interval(false), std::nullopt,
            Spectral::Quadrature::GaussLobatto)
            .number_of_elements() == 4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.FdPatches",
                  "[Evolution][Unit]") {
  test_dg_only_surrounding_blocks();
  test_external_boundary();
  test_block_without_fd_elements();
  test_tag();
}
