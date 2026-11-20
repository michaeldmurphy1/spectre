// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/Patches/ActionsKokkos.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// The part of a patch that `RunOnZerothElement` uses
struct MockPatchElements {
  const std::vector<ElementId<3>>& element_ids() const { return element_ids_; }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | element_ids_; }
  std::vector<ElementId<3>> element_ids_{};
};

struct MockPatch {
  const MockPatchElements& patch_elements() const { return elements; }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | elements; }
  MockPatchElements elements{};
};

struct MockPatchTag : db::SimpleTag {
  using type = MockPatch;
};

// An element action that records the element it ran on
struct RecordElement {
  static std::vector<ElementId<3>> recorded;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<3>& element_id, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    recorded.push_back(element_id);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
std::vector<ElementId<3>> RecordElement::recorded{};

template <typename Metavariables>
struct MockPatchArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 tmpl::list<MockPatchTag>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<evolution::patches::Actions::RunOnZerothElement<
              RecordElement, MockPatchTag>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<MockPatchArray<Metavariables>>;
};

// Runs the action on a patch with the `element_ids` and returns the elements
// that `RecordElement` ran on
std::vector<ElementId<3>> run(const std::vector<ElementId<3>>& element_ids) {
  using component = MockPatchArray<Metavariables>;
  RecordElement::recorded.clear();
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {MockPatch{MockPatchElements{element_ids}}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  return RecordElement::recorded;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.RunOnZerothElementKokkos",
                  "[Unit][Evolution]") {
  const ElementId<3> zeroth{0, {{{1, 0}, {1, 0}, {1, 0}}}};
  const ElementId<3> other_in_block_zero{0, {{{1, 1}, {1, 0}, {1, 0}}}};
  const ElementId<3> in_other_block{1, {{{1, 0}, {1, 0}, {1, 0}}}};
  REQUIRE(is_zeroth_element(zeroth));
  REQUIRE(not is_zeroth_element(other_in_block_zero));
  REQUIRE(not is_zeroth_element(in_other_block));
  // The patch holds the zeroth element: the action runs once, on it
  CHECK(run({in_other_block, zeroth, other_in_block_zero}) ==
        std::vector<ElementId<3>>{zeroth});
  // It doesn't: the action doesn't run
  CHECK(run({in_other_block, other_in_block_zero}).empty());
}
