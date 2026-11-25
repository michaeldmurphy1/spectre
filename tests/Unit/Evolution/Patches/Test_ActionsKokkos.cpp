// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/Patches/ActionsKokkos.hpp"
#include "Evolution/Patches/InboxTags.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Evolution/Patches/SelfStartKokkos.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/DeviceHistoryEvolvedVariablesKokkos.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// The state of the elements of the patch
struct State : db::SimpleTag {
  using type = int;
};
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};
using HistoryTag =
    ::Tags::DeviceHistoryEvolvedVariables<::Tags::Variables<tmpl::list<Var>>>;

const ElementId<3> patch_element{0, {{{1, 0}, {0, 0}, {0, 0}}}};
const ElementId<3> dg_neighbor{1};
const ElementId<3> neighbor_in_patch_1{0, {{{1, 1}, {0, 0}, {0, 0}}}};

// A patch that records what the actions ask of it
struct MockPatch {
  using state_tags = tmpl::list<State>;
  static size_t checks_of_last_step;

  std::vector<evolution::dg::subcell::fd_patch::OutgoingBoundaryData<3>>
  prepare_send(const Kokkos::DefaultExecutionSpace& /*exec*/,
               const int& /*state*/, const TimeStepId& next_time_step_id,
               const size_t integration_order) {
    sent_next_time_step_id = next_time_step_id;
    sent_integration_order = integration_order;
    evolution::dg::BoundaryData<3> data{};
    data.integration_order = integration_order;
    return {{.receiver = dg_neighbor,
             .receiver_is_dg = true,
             .sender = {Direction<3>::lower_xi(), patch_element},
             .data = data},
            {.receiver = neighbor_in_patch_1,
             .receiver_is_dg = false,
             .sender = {Direction<3>::lower_xi(), patch_element},
             .data = data}};
  }

  static bool received_all(
      const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
          messages) {
    return messages.size() == 2;
  }

  void receive(
      const Kokkos::DefaultExecutionSpace& /*exec*/,
      const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
          messages) {
    received += messages.size();
  }

  void compute_time_derivative(
      const Kokkos::DefaultExecutionSpace& /*exec*/,
      const gsl::not_null<int*> state,
      const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
      const TimeStepId& time_step_id,
      const domain::FunctionsOfTimeMap& /*functions_of_time*/) {
    derivative_time_step_ids.push_back(time_step_id);
    derivative_integration_orders.push_back(history->integration_order());
    *state += 10;
  }

  void update_variables(
      const Kokkos::DefaultExecutionSpace& /*exec*/,
      const gsl::not_null<int*> state,
      const gsl::not_null<TimeSteppers::DeviceHistory<>*> /*history*/,
      const TimeStepper& time_stepper, const TimeStepId& time_step_id,
      const TimeStepId& next_time_step_id, const TimeDelta& /*time_step*/) {
    update_orders.push_back(
        get<TimeSteppers::Tags::FixedOrder>(time_stepper.order()));
    update_time_step_ids.emplace_back(time_step_id, next_time_step_id);
    *state += 100;
  }

  static void check_last_step(const Kokkos::DefaultExecutionSpace& /*exec*/) {
    ++checks_of_last_step;
  }

  void save_initial_values(const Kokkos::DefaultExecutionSpace& /*exec*/,
                           const int& state) {
    saved = state;
  }
  void restore_initial_values(const Kokkos::DefaultExecutionSpace& /*exec*/,
                              const gsl::not_null<int*> state) const {
    *state = saved.value();
  }
  void discard_initial_values() { saved.reset(); }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  std::optional<TimeStepId> sent_next_time_step_id{};
  size_t sent_integration_order = 0;
  size_t received = 0;
  std::vector<TimeStepId> derivative_time_step_ids{};
  std::vector<size_t> derivative_integration_orders{};
  std::vector<size_t> update_orders{};
  std::vector<std::pair<TimeStepId, TimeStepId>> update_time_step_ids{};
  std::optional<int> saved{};
};
size_t MockPatch::checks_of_last_step = 0;

struct PatchTag : db::SimpleTag {
  using type = MockPatch;
};

// The part of the patch table that the actions use
struct MockPatchTable {
  static std::optional<evolution::patches::PatchTable<3>::Location> location(
      const ElementId<3>& element_id) {
    if (element_id == neighbor_in_patch_1) {
      return evolution::patches::PatchTable<3>::Location{.patch = 1, .slot = 0};
    }
    return std::nullopt;
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
struct MockPatchTableTag : db::SimpleTag {
  using type = MockPatchTable;
};

using DgInbox =
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3, false>;
using PatchInbox = evolution::patches::Tags::BoundaryDataInbox<3>;

// Declares the inbox of the DG elements
struct ReceiveOnDg {
  using inbox_tags = tmpl::list<DgInbox>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename Metavariables>
struct MockDgElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Testing,
                                        tmpl::list<ReceiveOnDg>>>;
};

struct SelfStartExit;
struct InsideSelfStart;

using patch_simple_tags =
    tmpl::list<PatchTag, State, HistoryTag, ::Tags::TimeStepId,
               ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
               ::Tags::ConcreteTimeStepper<TimeStepper>>;

template <typename Metavariables>
struct MockPatchArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<MockPatchTableTag>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              patch_simple_tags, ::time_stepper_ref_tags<TimeStepper>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<
              evolution::patches::Actions::SendBoundaryData<
                  PatchTag, MockDgElementArray<Metavariables>>,
              evolution::patches::Actions::ReceiveBoundaryData<3, PatchTag>,
              evolution::patches::Actions::ComputeTimeDerivative<PatchTag>,
              evolution::patches::Actions::UpdateVariables<PatchTag>,
              evolution::patches::Actions::CheckLastStepAtSlabBoundaries<
                  PatchTag>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<evolution::patches::Actions::SelfStartInitialize<PatchTag>,
                     evolution::patches::Actions::CheckForSelfStartCompletion<
                         SelfStartExit, PatchTag>,
                     ::Actions::Label<InsideSelfStart>,
                     ::Actions::Label<SelfStartExit>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using patch_table_tag = MockPatchTableTag;
  using patch_component = MockPatchArray<Metavariables>;
  using component_list = tmpl::list<MockPatchArray<Metavariables>,
                                    MockDgElementArray<Metavariables>>;
};

using patch_component = MockPatchArray<Metavariables>;
using dg_component = MockDgElementArray<Metavariables>;
using Runner = ActionTesting::MockRuntimeSystem<Metavariables>;

void set_up(const gsl::not_null<Runner*> runner, const TimeStepId& time_step_id,
            const TimeStepId& next_time_step_id, const TimeDelta& time_step) {
  for (int patch = 0; patch < 2; ++patch) {
    ActionTesting::emplace_array_component_and_initialize<patch_component>(
        runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, patch,
        tuples::tagged_tuple_from_typelist<patch_simple_tags>{
            MockPatch{}, 1, TimeSteppers::DeviceHistory<>{}, time_step_id,
            next_time_step_id, time_step,
            std::make_unique<TimeSteppers::AdamsBashforth>(3)});
  }
  ActionTesting::emplace_array_component<dg_component>(
      runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      dg_neighbor);
  db::mutate<HistoryTag>(
      [](const gsl::not_null<TimeSteppers::DeviceHistory<>*> history) {
        history->integration_order(2);
      },
      make_not_null(&ActionTesting::get_databox<patch_component>(runner, 0)));
}

const MockPatch& patch(const Runner& runner) {
  return ActionTesting::get_databox_tag<patch_component, PatchTag>(runner, 0);
}

// The actions of a substep pass the state, the history and the time step
// ids to the patch, and route the boundary data to the DG elements and the
// other patches
void test_step_actions() {
  const Slab slab{1.0, 2.0};
  const TimeStepId time_step_id{true, 3, slab.start()};
  const TimeStepId next_time_step_id{true, 3,
                                     slab.start() + slab.duration() / 2};
  Runner runner{{MockPatchTable{}}, {domain::FunctionsOfTimeMap{}}};
  set_up(make_not_null(&runner), time_step_id, next_time_step_id,
         slab.duration() / 2);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // SendBoundaryData
  ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
  CHECK(patch(runner).sent_next_time_step_id == next_time_step_id);
  CHECK(patch(runner).sent_integration_order == 2);
  const auto& dg_messages =
      ActionTesting::get_inbox_tag<dg_component, DgInbox>(runner, dg_neighbor)
          .messages;
  REQUIRE(dg_messages.count(time_step_id) == 1);
  REQUIRE(dg_messages.at(time_step_id).size() == 1);
  CHECK(dg_messages.at(time_step_id)[0].first ==
        DirectionalId<3>{Direction<3>::lower_xi(), patch_element});
  CHECK(dg_messages.at(time_step_id)[0].second.integration_order == 2);
  const auto& patch_1_inbox =
      ActionTesting::get_inbox_tag<patch_component, PatchInbox>(runner, 1);
  REQUIRE(patch_1_inbox.count(time_step_id) == 1);
  REQUIRE(patch_1_inbox.at(time_step_id).size() == 1);
  CHECK(std::get<0>(patch_1_inbox.at(time_step_id)[0]) == neighbor_in_patch_1);

  // ReceiveBoundaryData retries until all data of the substep has arrived,
  // and keeps the data of later substeps
  const DirectionalId<3> from_dg{Direction<3>::upper_xi(), dg_neighbor};
  auto& patch_0_inbox =
      ActionTesting::get_inbox_tag<patch_component, PatchInbox>(
          make_not_null(&runner), 0);
  patch_0_inbox[time_step_id].emplace_back(patch_element, from_dg,
                                           evolution::dg::BoundaryData<3>{});
  patch_0_inbox[next_time_step_id].emplace_back(
      patch_element, from_dg, evolution::dg::BoundaryData<3>{});
  CHECK_FALSE(ActionTesting::next_action_if_ready<patch_component>(
      make_not_null(&runner), 0));
  CHECK(patch(runner).received == 0);
  patch_0_inbox[time_step_id].emplace_back(patch_element, from_dg,
                                           evolution::dg::BoundaryData<3>{});
  CHECK(ActionTesting::next_action_if_ready<patch_component>(
      make_not_null(&runner), 0));
  CHECK(patch(runner).received == 2);
  CHECK_FALSE(patch_0_inbox.contains(time_step_id));
  CHECK(patch_0_inbox.contains(next_time_step_id));

  // ComputeTimeDerivative and UpdateVariables
  ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
  CHECK(patch(runner).derivative_time_step_ids ==
        std::vector<TimeStepId>{time_step_id});
  CHECK(patch(runner).derivative_integration_orders == std::vector<size_t>{2});
  ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
  CHECK(patch(runner).update_orders == std::vector<size_t>{3});
  CHECK(patch(runner).update_time_step_ids ==
        std::vector<std::pair<TimeStepId, TimeStepId>>{
            {time_step_id, next_time_step_id}});
  CHECK(ActionTesting::get_databox_tag<patch_component, State>(runner, 0) ==
        111);

  // At a slab boundary the last substep is checked
  MockPatch::checks_of_last_step = 0;
  ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
  CHECK(MockPatch::checks_of_last_step == 1);
}

// The self start saves the variables, takes smaller steps, restores the
// variables at the end of each order and leaves the loop at the end
void test_self_start() {
  const Slab slab{1.0, 2.0};
  const TimeDelta step = slab.duration();
  const TimeStepId self_start_id{true, -2, slab.start()};
  {
    INFO("Start");
    Runner runner{{MockPatchTable{}}, {domain::FunctionsOfTimeMap{}}};
    set_up(make_not_null(&runner), self_start_id,
           TimeStepId{true, -2, slab.start() + step / 3}, step);
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
    ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
    CHECK(patch(runner).saved == std::optional{1});
    // As `SelfStart::Actions::Initialize`: a third of the step for the two
    // values that the third-order stepper needs
    CHECK(ActionTesting::get_databox_tag<patch_component, ::Tags::TimeStep>(
              runner, 0) == step / 3);
    CHECK(ActionTesting::get_databox_tag<
              patch_component, SelfStart::Tags::InitialValue<::Tags::TimeStep>>(
              runner, 0) == std::make_tuple(step));
    // Not at the end of an order: the self start continues
    ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
    CHECK(patch(runner).saved == std::optional{1});
    CHECK(ActionTesting::get_next_action_index<patch_component>(runner, 0) ==
          2);
  }
  {
    INFO("End of an order");
    Runner runner{{MockPatchTable{}}, {domain::FunctionsOfTimeMap{}}};
    set_up(make_not_null(&runner), self_start_id,
           TimeStepId{true, -1, slab.start()}, step);
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
    ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
    db::mutate<State>(
        [](const gsl::not_null<int*> state) { *state = 7; },
        make_not_null(&ActionTesting::get_databox<patch_component>(
            make_not_null(&runner), 0)));
    ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
    CHECK(ActionTesting::get_databox_tag<patch_component, State>(runner, 0) ==
          1);
    CHECK(patch(runner).saved == std::optional{1});
  }
  {
    INFO("End");
    Runner runner{{MockPatchTable{}}, {domain::FunctionsOfTimeMap{}}};
    set_up(make_not_null(&runner), self_start_id,
           TimeStepId{true, 0, slab.start()}, step);
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
    ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
    db::mutate<State>(
        [](const gsl::not_null<int*> state) { *state = 7; },
        make_not_null(&ActionTesting::get_databox<patch_component>(
            make_not_null(&runner), 0)));
    ActionTesting::next_action<patch_component>(make_not_null(&runner), 0);
    CHECK(ActionTesting::get_databox_tag<patch_component, State>(runner, 0) ==
          1);
    CHECK_FALSE(patch(runner).saved.has_value());
    // At the label after the loop
    CHECK(ActionTesting::get_next_action_index<patch_component>(runner, 0) ==
          3);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.ActionsKokkos", "[Unit][Evolution]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  test_step_actions();
  test_self_start();
}
