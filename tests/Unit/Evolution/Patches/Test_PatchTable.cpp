// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace {
// The membership rule decides which elements go into patches and with which
// mesh, and the table groups them by mesh
void test_patch_mesh() {
  const auto domain = rectilinear_domain<2>(
      Index<2>{2, 1},
      std::array<std::vector<double>, 2>{{{0.0, 1.0, 2.0}, {0.0, 1.0}}});
  const std::vector<std::array<size_t, 2>> refinement(2, {{1, 1}});
  const Mesh<2> mesh_a{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const Mesh<2> mesh_b{6, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  // Block 0 in patches with one mesh, and the lower half of block 1 with
  // another, the rest outside patches
  const evolution::patches::PatchTable<2> table{
      domain.blocks(), refinement,
      [&](const Element<2>& element) -> std::optional<Mesh<2>> {
        if (element.id().block_id() == 0) {
          return mesh_a;
        }
        if (element.id().segment_id(1).index() == 0) {
          return mesh_b;
        }
        return std::nullopt;
      },
      std::nullopt, true};
  REQUIRE(table.number_of_patches() == 2);
  CHECK(table.dg_mesh(0) == mesh_a);
  CHECK(table.dg_mesh(1) == mesh_b);
  CHECK(table.element_ids(0).size() == 4);
  CHECK(table.element_ids(1).size() == 2);
  CHECK(table.number_of_elements() == 6);
  CHECK(table.number_of_elements_outside_patches() == 2);
  CHECK(table.dedicated_procs());
  const ElementId<2> outside{1, {{SegmentId{1, 0}, SegmentId{1, 1}}}};
  CHECK_FALSE(table.location(outside).has_value());
  for (size_t patch = 0; patch < 2; ++patch) {
    for (size_t slot = 0; slot < table.element_ids(patch).size(); ++slot) {
      CHECK(table.location(table.element_ids(patch)[slot]) ==
            evolution::patches::PatchTable<2>::Location{patch, slot});
    }
  }
  test_serialization(table);
  test_copy_semantics(table);
}

void test_patch_procs() {
  using evolution::patches::patch_procs;
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
                        "No processor may hold the 1 patches"));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.PatchTable", "[Evolution][Unit]") {
  test_patch_mesh();
  test_patch_procs();
}
