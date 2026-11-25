// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <pup.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Context.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// The slab size an element requests, e.g. from its grid spacing
struct LocalStep : db::SimpleTag {
  using type = double;
};

// A step chooser that uses the data of each element, so that the requests
// are reduced over the elements
template <bool MustSetStepSize>
struct LocalChooser : StepChooser<StepChooserUse::Slab> {
  LocalChooser() = default;
  explicit LocalChooser(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(LocalChooser);  // NOLINT
#pragma GCC diagnostic pop
  static constexpr Options::String help = "";
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<LocalStep>;
  TimeStepRequest operator()(const double local_step,
                             const double /*last_step*/) const {
    return {.size_goal = local_step};
  }
  bool uses_local_data() const override { return true; }
  bool can_be_delayed() const override { return true; }
  bool must_set_step_size() const override { return MustSetStepSize; }
};
template <bool MustSetStepSize>
PUP::able::PUP_ID LocalChooser<MustSetStepSize>::my_PUP_ID = 0;  // NOLINT

// Records the reductions instead of reducing over the elements, which
// ActionTesting doesn't support
struct RecordReduction {
  static std::vector<std::pair<int64_t, TimeStepRequestProcessor>> contributed;

  template <typename ParallelComponent, typename Metavariables,
            typename SenderProxy>
  static void contribute(Parallel::GlobalCache<Metavariables>& /*cache*/,
                         const SenderProxy& /*sender_proxy*/,
                         Events::ChangeSlabSize::ReductionData reduction_data) {
    const auto& [slab, requests] = reduction_data.data();
    contributed.emplace_back(slab, requests);
  }
};
std::vector<std::pair<int64_t, TimeStepRequestProcessor>>
    RecordReduction::contributed{};

using element_tags = tmpl::list<Tags::ChangeSlabSize::NumberOfExpectedMessages,
                                Tags::ChangeSlabSize::NewSlabSize, LocalStep>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<element_tags>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using slab_size_reduction = RecordReduction;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::list<StepChoosers::Constant, LocalChooser<false>,
                              LocalChooser<true>>>>;
  };
};

template <typename... Choosers>
Events::ChangeSlabSize make_event(const uint64_t delay_change,
                                  Choosers... choosers) {
  std::vector<std::unique_ptr<StepChooser<StepChooserUse::Slab>>> list{};
  (list.push_back(std::make_unique<Choosers>(std::move(choosers))), ...);
  return {std::move(list), delay_change, Options::Context{}};
}

void test_slab_to_change() {
  const Slab slab(1.0, 2.0);
  const TimeStepId at_boundary(true, 3, slab.start());
  const TimeStepId inside(true, 3, slab.start() + slab.duration() / 2);
  const auto no_delay = make_event(0, StepChoosers::Constant{0.5});
  CHECK(no_delay.slab_to_change(at_boundary) == 3);
  CHECK(no_delay.slab_to_change(inside) == 4);
  const auto with_delay = make_event(2, StepChoosers::Constant{0.5});
  CHECK(with_delay.slab_to_change(at_boundary) == 5);
  CHECK(with_delay.slab_to_change(inside) == 6);

  CHECK_FALSE(no_delay.synchronization_required());
  CHECK(make_event(0, StepChoosers::Constant{0.5}, LocalChooser<false>{})
            .synchronization_required());
}

void test_request_step_sizes() {
  const auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<Metavariables>, LocalStep>>(
      Metavariables{}, 0.25);
  const Slab slab(1.0, 2.0);
  const TimeStepId time_step_id(true, 3, slab.start());
  const auto event =
      make_event(0, StepChoosers::Constant{0.5}, LocalChooser<false>{});
  TimeStepRequestProcessor requests(true);
  event.request_step_sizes(make_not_null(&requests), time_step_id, LtsMode::Off,
                           box);
  TimeStepRequestProcessor expected(true);
  expected.process(TimeStepRequest{.size_goal = 0.5});
  expected.process(TimeStepRequest{.size_goal = 0.25});
  CHECK(requests == expected);
  CHECK(requests.new_step_size_goal() == std::optional{0.25});

  // Choosers that set the step size can't choose the slab size with local
  // time stepping
  const auto sets_step = make_event(0, LocalChooser<true>{});
  TimeStepRequestProcessor lts_requests(true);
  sets_step.request_step_sizes(make_not_null(&lts_requests), time_step_id,
                               LtsMode::Off, box);
  CHECK_THROWS_WITH(
      sets_step.request_step_sizes(make_not_null(&lts_requests), time_step_id,
                                   LtsMode::Conservative, box),
      Catch::Matchers::ContainsSubstring(
          "StepChooser cannot be used for the slab size in an LTS "
          "evolution."));
}

// The event (`operator()`) on an element: the requests are stored for the
// slab to change, or reduced over the elements
void test_send_requests() {
  using component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_array_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0,
      tuples::TaggedTuple<Tags::ChangeSlabSize::NumberOfExpectedMessages,
                          Tags::ChangeSlabSize::NewSlabSize, LocalStep>{
          std::map<int64_t, size_t>{},
          std::map<int64_t, std::vector<TimeStepRequestProcessor>>{}, 0.25});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& box = ActionTesting::get_databox<component>(make_not_null(&runner), 0);
  auto& cache = ActionTesting::cache<component>(runner, 0);
  const Slab slab(1.0, 2.0);
  const TimeStepId time_step_id(true, 3, slab.start());
  const auto run = [&](const Events::ChangeSlabSize& event) {
    event(make_not_null(&box), time_step_id, LtsMode::Off, cache, 0,
          static_cast<const component*>(nullptr),
          Event::ObservationValue{.name = "Time", .value = 1.0});
  };

  TimeStepRequestProcessor constant_requests(true);
  constant_requests.process(TimeStepRequest{.size_goal = 0.5});
  run(make_event(1, StepChoosers::Constant{0.5}));
  CHECK(db::get<Tags::ChangeSlabSize::NumberOfExpectedMessages>(box) ==
        std::map<int64_t, size_t>{{4, 1}});
  CHECK(db::get<Tags::ChangeSlabSize::NewSlabSize>(box) ==
        std::map<int64_t, std::vector<TimeStepRequestProcessor>>{
            {4, {constant_requests}}});
  CHECK(RecordReduction::contributed.empty());

  TimeStepRequestProcessor local_requests(true);
  local_requests.process(TimeStepRequest{.size_goal = 0.25});
  run(make_event(0, LocalChooser<false>{}));
  CHECK(db::get<Tags::ChangeSlabSize::NumberOfExpectedMessages>(box) ==
        std::map<int64_t, size_t>{{3, 1}, {4, 1}});
  CHECK(db::get<Tags::ChangeSlabSize::NewSlabSize>(box).size() == 1);
  CHECK(RecordReduction::contributed ==
        std::vector<std::pair<int64_t, TimeStepRequestProcessor>>{
            {3, local_requests}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.ChangeSlabSize.Event", "[Unit][Time]") {
  test_slab_to_change();
  test_request_step_sizes();
  test_send_requests();
}
