// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Patches/CombinedReduction.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// The part of the patch table that `CombineReductions` uses
struct MockPatchTable {
  size_t number_of_patches() const { return patches; }
  size_t number_of_elements_outside_patches() const { return elements_outside; }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | patches;
    p | elements_outside;
  }
  size_t patches{};
  size_t elements_outside{};
};

struct MockPatchTableTag : db::SimpleTag {
  using type = MockPatchTable;
};

struct AddArrays {
  std::array<double, 2> operator()(const std::array<double, 2>& lhs,
                                   const std::array<double, 2>& rhs) const {
    return {{lhs[0] + rhs[0], lhs[1] + rhs[1]}};
  }
};

// A count, a vector and a sum that the reduction finalizes into the mean
using ReductionData = Parallel::ReductionData<
    Parallel::ReductionDatum<int, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    Parallel::ReductionDatum<std::array<double, 2>, AddArrays>,
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Divides<>,
                             std::index_sequence<1>>>;
using Result = std::tuple<int, double, std::array<double, 2>, double>;

// Records the results each component receives
template <typename Component>
struct Record {
  static std::vector<Result> received;

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const int id,
                    const double count, const std::array<double, 2>& vector,
                    const double mean) {
    received.emplace_back(id, count, vector, mean);
  }
};
template <typename Component>
std::vector<Result> Record<Component>::received{};

struct TargetTag {};
struct DgTag {};
struct PatchTag {};

template <typename Metavariables, typename ChareType, typename RecordTag>
struct MockComponent {
  using metavariables = Metavariables;
  using chare_type = ChareType;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<MockPatchTableTag>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<
          tmpl::list<evolution::patches::Tags::PendingReductions>>>>>;
};

struct Metavariables {
  using patch_table_tag = MockPatchTableTag;
  using target = MockComponent<Metavariables, ActionTesting::MockSingletonChare,
                               TargetTag>;
  using dg_array =
      MockComponent<Metavariables, ActionTesting::MockArrayChare, DgTag>;
  using patch_array =
      MockComponent<Metavariables, ActionTesting::MockArrayChare, PatchTag>;
  using component_list = tmpl::list<target, dg_array, patch_array>;
};

using Target = Metavariables::target;
using DgArray = Metavariables::dg_array;
using PatchArray = Metavariables::patch_array;

using Combine = evolution::patches::Actions::CombineReductions<
    ReductionData, Record<TargetTag>, tmpl::list<Target>, PatchArray, DgArray>;
using CombineAndBroadcast = evolution::patches::Actions::CombineReductions<
    ReductionData, Record<DgTag>, tmpl::list<PatchArray, DgArray>, PatchArray,
    DgArray>;

using Runner = ActionTesting::MockRuntimeSystem<Metavariables>;

void set_up(const gsl::not_null<Runner*> runner) {
  const tuples::TaggedTuple<evolution::patches::Tags::PendingReductions>
      no_pending_reductions{};
  ActionTesting::emplace_singleton_component_and_initialize<Target>(
      runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      no_pending_reductions);
  ActionTesting::emplace_array_component_and_initialize<DgArray>(
      runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      no_pending_reductions);
  ActionTesting::emplace_array_component_and_initialize<PatchArray>(
      runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      no_pending_reductions);
  ActionTesting::set_phase(runner, Parallel::Phase::Testing);
}

size_t number_pending(const Runner& runner) {
  return ActionTesting::get_databox_tag<
             PatchArray, evolution::patches::Tags::PendingReductions>(runner, 0)
      .size();
}

// The DgElementArray and the patches contribute: the first patch combines the
// two partial reductions of the same id, also if another reduction arrives in
// between, and finalizes the result once
void test_two_components() {
  Record<TargetTag>::received.clear();
  Runner runner{{MockPatchTable{.patches = 1, .elements_outside = 4}}};
  set_up(make_not_null(&runner));
  ActionTesting::simple_action<PatchArray, Combine>(
      make_not_null(&runner), 0, 7, 2.0, std::array{1.0, 2.0}, 4.0);
  CHECK(number_pending(runner) == 1);
  ActionTesting::simple_action<PatchArray, Combine>(
      make_not_null(&runner), 0, 8, 1.0, std::array{0.0, 0.0}, 1.0);
  CHECK(number_pending(runner) == 2);
  CHECK(ActionTesting::is_simple_action_queue_empty<Target>(runner, 0));
  ActionTesting::simple_action<PatchArray, Combine>(
      make_not_null(&runner), 0, 7, 3.0, std::array{10.0, 20.0}, 6.0);
  CHECK(number_pending(runner) == 1);
  ActionTesting::invoke_queued_simple_action<Target>(make_not_null(&runner), 0);
  CHECK(ActionTesting::is_simple_action_queue_empty<Target>(runner, 0));
  REQUIRE(Record<TargetTag>::received.size() == 1);
  CHECK(Record<TargetTag>::received[0] ==
        Result{7, 5.0, std::array{11.0, 22.0}, 2.0});
}

// Without elements in the DgElementArray only the patches contribute, and the
// result isn't sent to the DgElementArray
void test_patches_only() {
  Record<DgTag>::received.clear();
  Runner runner{{MockPatchTable{.patches = 1, .elements_outside = 0}}};
  set_up(make_not_null(&runner));
  ActionTesting::simple_action<PatchArray, CombineAndBroadcast>(
      make_not_null(&runner), 0, 3, 2.0, std::array{1.0, 2.0}, 4.0);
  CHECK(number_pending(runner) == 0);
  CHECK(ActionTesting::is_simple_action_queue_empty<DgArray>(runner, 0));
  ActionTesting::invoke_queued_simple_action<PatchArray>(make_not_null(&runner),
                                                         0);
  REQUIRE(Record<DgTag>::received.size() == 1);
  CHECK(Record<DgTag>::received[0] ==
        Result{3, 2.0, std::array{1.0, 2.0}, 2.0});

  // With elements in the DgElementArray both arrays receive the result
  Record<DgTag>::received.clear();
  Runner runner_with_dg{{MockPatchTable{.patches = 1, .elements_outside = 1}}};
  set_up(make_not_null(&runner_with_dg));
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::simple_action<PatchArray, CombineAndBroadcast>(
        make_not_null(&runner_with_dg), 0, 3, 1.0, std::array{1.0, 2.0}, 4.0);
  }
  ActionTesting::invoke_queued_simple_action<PatchArray>(
      make_not_null(&runner_with_dg), 0);
  ActionTesting::invoke_queued_simple_action<DgArray>(
      make_not_null(&runner_with_dg), 0);
  CHECK(Record<DgTag>::received ==
        std::vector<Result>(2, Result{3, 2.0, std::array{2.0, 4.0}, 4.0}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.CombinedReduction",
                  "[Unit][Evolution]") {
  test_two_components();
  test_patches_only();
}
