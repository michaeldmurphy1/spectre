// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/FdPatch/CheckOptions.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using CheckpointEvents =
    Tags::EventsAndTriggers<Triggers::WhenToCheck::AtCheckpoints>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      tmpl::list<CheckpointEvents, Tags::EventsRunAtCleanup>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::Completion>>,
                  tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
};

evolution::dg::subcell::SubcellOptions subcell_options(
    const bool always_use_subcells,
    const ::fd::DerivativeOrder derivative_order) {
  return {4.0,
          1,
          1.0e-7,
          1.0e-3,
          always_use_subcells,
          false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
          false,
          derivative_order,
          1,
          1,
          1};
}

std::vector<std::unique_ptr<Event>> completion() {
  std::vector<std::unique_ptr<Event>> events{};
  events.push_back(std::make_unique<Events::Completion>());
  return events;
}

// Checks the options with the `checkpoint_events` and `cleanup_events` in the
// cache
void check(const evolution::dg::subcell::SubcellOptions& options,
           const LtsMode lts_mode, EventsAndTriggers checkpoint_events,
           std::vector<std::unique_ptr<Event>> cleanup_events) {
  using component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::move(checkpoint_events), std::move(cleanup_events)}};
  ActionTesting::emplace_array_component<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  const auto box = db::create<db::AddSimpleTags<
      evolution::dg::subcell::Tags::SubcellOptions<3>, ::Tags::LtsMode>>(
      options, lts_mode);
  evolution::dg::subcell::fd_patch::check_options(
      box, ActionTesting::cache<component>(runner, 0));
}

void test_check_options() {
  const auto valid = subcell_options(true, ::fd::DerivativeOrder::Two);
  check(valid, LtsMode::Off, EventsAndTriggers{}, {});

  CHECK_THROWS_WITH(
      check(subcell_options(false, ::fd::DerivativeOrder::Two), LtsMode::Off,
            EventsAndTriggers{}, {}),
      Catch::Matchers::ContainsSubstring("FD patches need AlwaysUseSubcells."));
  CHECK_THROWS_WITH(
      check(subcell_options(true, ::fd::DerivativeOrder::Four), LtsMode::Off,
            EventsAndTriggers{}, {}),
      Catch::Matchers::ContainsSubstring(
          "FD patches only support FiniteDifferenceDerivativeOrder: 2, but "
          "got 4"));
  CHECK_THROWS_WITH(
      check(valid, LtsMode::Conservative, EventsAndTriggers{}, {}),
      Catch::Matchers::ContainsSubstring(
          "FD patches don't support local time stepping"));

  EventsAndTriggers::Storage storage{};
  storage.emplace_back(std::make_unique<Triggers::Always>(), completion());
  CHECK_THROWS_WITH(
      check(valid, LtsMode::Off, EventsAndTriggers{std::move(storage)}, {}),
      Catch::Matchers::ContainsSubstring(
          "but there are events in EventsAndTriggersAtCheckpoints or "
          "EventsRunAtCleanup: (") and
          Catch::Matchers::ContainsSubstring("Completion)"));
  CHECK_THROWS_WITH(
      check(valid, LtsMode::Off, EventsAndTriggers{}, completion()),
      Catch::Matchers::ContainsSubstring("EventsRunAtCleanup: (") and
          Catch::Matchers::ContainsSubstring("Completion)"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.CheckOptions",
                  "[Evolution][Unit]") {
  register_factory_classes_with_charm<Metavariables>();
  test_check_options();
}
