// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <pup.h>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Patches/InboxTags.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Evolution/Patches/SendToPatch.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Inbox =
    evolution::patches::Tags::ElementsInbox<1, int, std::vector<double>>;

// The part of the patch table that `send_to_patch` uses
struct MockPatchTable {
  std::optional<evolution::patches::PatchTable<1>::Location> location(
      const ElementId<1>& element_id) const {
    const auto found = patch_of_element.find(element_id);
    if (found == patch_of_element.end()) {
      return std::nullopt;
    }
    return evolution::patches::PatchTable<1>::Location{.patch = found->second,
                                                       .slot = 0};
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | patch_of_element; }
  std::unordered_map<ElementId<1>, size_t> patch_of_element{};
};

struct MockPatchTableTag : db::SimpleTag {
  using type = MockPatchTable;
};

// An action that declares the inbox of the patches
struct ReceiveData {
  using inbox_tags = tmpl::list<Inbox>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename Metavariables>
struct MockPatchComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<MockPatchTableTag>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Testing,
                                        tmpl::list<ReceiveData>>>;
};

struct Metavariables {
  static constexpr bool use_patches = true;
  using patch_table_tag = MockPatchTableTag;
  using patch_component = MockPatchComponent<Metavariables>;
  using component_list = tmpl::list<patch_component>;
};

struct MetavariablesWithoutPatches {
  using component_list = tmpl::list<>;
};

void test_send_to_patch() {
  using component = Metavariables::patch_component;
  const ElementId<1> in_patch_0{0, {{{1, 0}}}};
  const ElementId<1> in_patch_1{0, {{{1, 1}}}};
  const ElementId<1> outside{1};
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {MockPatchTable{{{in_patch_0, 0}, {in_patch_1, 1}}}}};
  for (int patch = 0; patch < 2; ++patch) {
    ActionTesting::emplace_array_component<component>(
        make_not_null(&runner), ActionTesting::NodeId{0},
        ActionTesting::LocalCoreId{0}, patch);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& cache = ActionTesting::cache<component>(runner, 0);
  const DirectionalId<1> sender{Direction<1>::lower_xi(), outside};

  const bool sent = evolution::patches::send_to_patch<Inbox>(
      cache, in_patch_1, 3, sender, std::vector{1.0, 2.0});
  CHECK(sent);
  const auto& inbox_1 =
      ActionTesting::get_inbox_tag<component, Inbox>(runner, 1);
  REQUIRE(inbox_1.size() == 1);
  CHECK(inbox_1.at(3) ==
        Inbox::messages_type{{in_patch_1, sender, std::vector{1.0, 2.0}}});
  CHECK(ActionTesting::get_inbox_tag<component, Inbox>(runner, 0).empty());

  // The receiver isn't in a patch, so the caller sends the data to it
  std::vector<double> kept{3.0};
  const bool sent_outside = evolution::patches::send_to_patch<Inbox>(
      cache, outside, 3, sender, std::move(kept));
  CHECK_FALSE(sent_outside);
  // NOLINTNEXTLINE(bugprone-use-after-move,hicpp-invalid-access-moved)
  CHECK(kept == std::vector{3.0});
  CHECK(ActionTesting::get_inbox_tag<component, Inbox>(runner, 1).size() == 1);

  // Without patches every receiver is an element
  Parallel::GlobalCache<MetavariablesWithoutPatches> empty_cache{};
  static_assert(
      not evolution::patches::uses_patches<MetavariablesWithoutPatches>());
  static_assert(evolution::patches::uses_patches<Metavariables>());
  std::vector<double> not_sent{4.0};
  const bool sent_without_patches = evolution::patches::send_to_patch<Inbox>(
      empty_cache, in_patch_0, 3, sender, std::move(not_sent));
  CHECK_FALSE(sent_without_patches);
  // NOLINTNEXTLINE(bugprone-use-after-move,hicpp-invalid-access-moved)
  CHECK(not_sent == std::vector{4.0});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.SendToPatch", "[Unit][Evolution]") {
  test_send_to_patch();
}
