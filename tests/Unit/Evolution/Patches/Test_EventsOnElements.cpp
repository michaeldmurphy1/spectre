// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Patches/EventsOnElements.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/Triggers/OnSubsteps.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarA : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct ScalarB : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <typename Tag>
struct ScalarCompute : Tag, db::ComputeTag {
  using base = Tag;
  using return_type = Scalar<DataVector>;
  using argument_tags = tmpl::list<>;
  static void function(const gsl::not_null<Scalar<DataVector>*> /*result*/) {}
};

// An event that patches don't run
struct UnsupportedEvent : public Event {
  UnsupportedEvent() = default;
  explicit UnsupportedEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(UnsupportedEvent);  // NOLINT
#pragma GCC diagnostic pop
  bool needs_evolved_variables() const override { return false; }
};
PUP::able::PUP_ID UnsupportedEvent::my_PUP_ID = 0;  // NOLINT

// An event that integrates over the volume, as a measurement
struct VolumeIntegralsEvent : public Event {
  using volume_integrand_tags = tmpl::list<ScalarCompute<ScalarA>>;
  VolumeIntegralsEvent() = default;
  explicit VolumeIntegralsEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(VolumeIntegralsEvent);  // NOLINT
#pragma GCC diagnostic pop
  bool needs_evolved_variables() const override { return true; }
};
PUP::able::PUP_ID VolumeIntegralsEvent::my_PUP_ID = 0;  // NOLINT

using NormsOfA = Events::ObserveNorms<tmpl::list<ScalarCompute<ScalarA>>>;
using NormsOfB = Events::ObserveNorms<tmpl::list<ScalarCompute<ScalarB>>>;

struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<Events::Completion, Events::ChangeSlabSize,
                                     NormsOfA, NormsOfB, UnsupportedEvent,
                                     VolumeIntegralsEvent>>,
        tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
};

void test_event_types() {
  static_assert(
      std::is_same_v<evolution::patches::events_on_elements<Metavariables>,
                     tmpl::list<NormsOfA, NormsOfB>>);
  static_assert(
      std::is_same_v<evolution::patches::observe_norms_events<Metavariables>,
                     tmpl::list<NormsOfA, NormsOfB>>);
  static_assert(
      std::is_same_v<evolution::patches::volume_integrals_events<Metavariables>,
                     tmpl::list<VolumeIntegralsEvent>>);
  static_assert(evolution::patches::IntegratesOverVolume<VolumeIntegralsEvent>);
  static_assert(not evolution::patches::IntegratesOverVolume<NormsOfA>);

  // The element boxes have `ScalarA`, so only `ScalarB` is computed for the
  // observations
  static_assert(std::is_same_v<
                evolution::patches::observation_compute_tags<Metavariables>,
                tmpl::list<ScalarCompute<ScalarA>, ScalarCompute<ScalarB>>>);
  static_assert(
      std::is_same_v<evolution::patches::element_observation_compute_tags<
                         Metavariables, db::DataBox<tmpl::list<ScalarA>>>,
                     tmpl::list<ScalarCompute<ScalarB>>>);

  const NormsOfA norms{};
  const VolumeIntegralsEvent integrals{};
  const UnsupportedEvent unsupported{};
  const Events::Completion completion{};
  CHECK(evolution::patches::runs_on_elements<Metavariables>(norms));
  CHECK(evolution::patches::runs_dense_on_elements<Metavariables>(norms));
  CHECK_FALSE(evolution::patches::integrates_over_volume<Metavariables>(norms));
  CHECK(evolution::patches::integrates_over_volume<Metavariables>(integrals));
  CHECK_FALSE(evolution::patches::runs_on_elements<Metavariables>(integrals));
  CHECK_FALSE(evolution::patches::runs_on_elements<Metavariables>(completion));
  CHECK_FALSE(
      evolution::patches::runs_dense_on_elements<Metavariables>(unsupported));

  // Each name once, without template arguments
  CHECK(evolution::patches::event_names<
            evolution::patches::events_on_elements<Metavariables>>() ==
        "ObserveNorms");
  CHECK(
      evolution::patches::event_names<
          tmpl::list<Events::Completion, NormsOfA, Events::ChangeSlabSize>>() ==
      "Completion, ObserveNorms, ChangeSlabSize");
  CHECK(evolution::patches::event_names<tmpl::list<>>().empty());

  CHECK_THROWS_WITH(
      evolution::patches::error_about_unsupported_event<Metavariables>(
          unsupported),
      Catch::Matchers::ContainsSubstring(
          "FD patches only run the events Completion, ChangeSlabSize, "
          "ObserveNorms, but got") and
          Catch::Matchers::ContainsSubstring("UnsupportedEvent"));
  CHECK_THROWS_WITH(
      evolution::patches::error_about_unsupported_dense_event<Metavariables>(
          unsupported),
      Catch::Matchers::ContainsSubstring(
          "FD patches only run the events Completion, ObserveNorms and the "
          "volume integrals of the measurements VolumeIntegralsEvent in "
          "EventsAndDenseTriggers, but got"));
}

template <typename... EventTypes>
EventsAndTriggers::TriggerAndEvents trigger_and_events(
    std::unique_ptr<Trigger> trigger, EventTypes... events) {
  std::vector<std::unique_ptr<Event>> event_list{};
  (event_list.push_back(std::make_unique<EventTypes>(std::move(events))), ...);
  return {std::move(trigger), std::move(event_list)};
}

void test_triggered_events() {
  const auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>>>(
      Metavariables{});
  {
    INFO("Sorted by how patches run them");
    EventsAndTriggers::Storage storage{};
    storage.push_back(trigger_and_events(std::make_unique<Triggers::Always>(),
                                         NormsOfA{}, Events::ChangeSlabSize{},
                                         NormsOfB{}));
    // Doesn't fire
    storage.push_back(trigger_and_events(
        std::make_unique<Triggers::Not>(std::make_unique<Triggers::Always>()),
        Events::Completion{}, NormsOfA{}));
    const EventsAndTriggers events_and_triggers{std::move(storage)};
    const auto events = evolution::patches::triggered_events<Metavariables>(
        events_and_triggers, box);
    CHECK_FALSE(events.completed);
    REQUIRE(events.on_elements.size() == 2);
    CHECK(dynamic_cast<const NormsOfA*>(events.on_elements[0]) != nullptr);
    CHECK(dynamic_cast<const NormsOfB*>(events.on_elements[1]) != nullptr);
    CHECK(events.slab_size_changes.size() == 1);
  }
  {
    INFO("Completion");
    EventsAndTriggers::Storage storage{};
    storage.push_back(trigger_and_events(std::make_unique<Triggers::Always>(),
                                         Events::Completion{}));
    const EventsAndTriggers events_and_triggers{std::move(storage)};
    const auto events = evolution::patches::triggered_events<Metavariables>(
        events_and_triggers, box);
    CHECK(events.completed);
    CHECK(events.on_elements.empty());
    CHECK(events.slab_size_changes.empty());
  }
  {
    INFO("Unsupported events fail even if their trigger doesn't fire");
    EventsAndTriggers::Storage storage{};
    storage.push_back(trigger_and_events(
        std::make_unique<Triggers::Not>(std::make_unique<Triggers::Always>()),
        UnsupportedEvent{}));
    const EventsAndTriggers events_and_triggers{std::move(storage)};
    CHECK_THROWS_WITH(
        evolution::patches::triggered_events<Metavariables>(events_and_triggers,
                                                            box),
        Catch::Matchers::ContainsSubstring("FD patches only run the events"));
  }
  {
    INFO("OnSubsteps");
    EventsAndTriggers::Storage storage{};
    storage.push_back(
        trigger_and_events(std::make_unique<Triggers::OnSubsteps>(
                               std::make_unique<Triggers::Always>()),
                           NormsOfA{}));
    const EventsAndTriggers events_and_triggers{std::move(storage)};
    CHECK_THROWS_WITH(evolution::patches::triggered_events<Metavariables>(
                          events_and_triggers, box),
                      Catch::Matchers::ContainsSubstring(
                          "FD patches don't support OnSubsteps triggers."));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.EventsOnElements",
                  "[Unit][Evolution]") {
  test_event_types();
  test_triggered_events();
}
