// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Patches/CombinedReduction.hpp"
#include "Evolution/Patches/DeviceObservationsKokkos.hpp"
#include "Evolution/Patches/ElementBoxesKokkos.hpp"
#include "Evolution/Patches/EventsOnElements.hpp"
#include "Evolution/Patches/PatchState.hpp"
#include "Evolution/Patches/Tags.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace evolution::patches::Actions {
namespace EventActions_detail {
// Runs the `events` on each element of the patch, as
// `EventsAndTriggers::run_events` does for an element. With `dense_output`,
// the variables of the elements are the last dense output of the patch.
//
// With `Tags::ObserveOnDevice`, the events that the patch can observe on the
// device run there instead (see `observe_norms_on_device`).
template <typename PatchTag, typename ParallelComponent, typename DbTagsList,
          typename Metavariables>
void run_events_on_elements(const std::vector<const Event*>& events,
                            const gsl::not_null<db::DataBox<DbTagsList>*> box,
                            Parallel::GlobalCache<Metavariables>& cache,
                            const Event::ObservationValue& observation_value,
                            const bool dense_output = false) {
  using Patch = typename PatchTag::type;
  std::vector<const Event*> host_events{};
  if constexpr (ComputesDeviceObservables<Patch>) {
    std::vector<const Event*> device_events{};
    for (const Event* const event : events) {
      if (Parallel::get<Tags::ObserveOnDevice>(cache) and
          observes_on_device<Patch, Metavariables>(*event)) {
        device_events.push_back(event);
      } else {
        host_events.push_back(event);
      }
    }
    if (not device_events.empty()) {
      observe_norms_on_device<PatchTag, ParallelComponent>(
          device_events, box, cache, observation_value, dense_output);
    }
  } else {
    host_events = events;
  }
  if (host_events.empty()) {
    return;
  }
  const bool needs_variables =
      alg::any_of(host_events, [](const Event* const event) {
        return event->needs_evolved_variables();
      });
  for_each_element_box<PatchTag>(
      box, cache,
      not needs_variables ? ElementVariables::WithoutValues
      : dense_output      ? ElementVariables::DenseOutput
                          : ElementVariables::State,
      [&](const auto& element_id, auto& element_box) {
        const Kokkos::Profiling::ScopedRegion events_region{
            "patches::element_boxes::run_events"};
        std::optional<Kokkos::Profiling::ScopedRegion> observation_box_region{
            "patches::element_boxes::observation_box"};
        auto observation_box =
            make_observation_box<element_observation_compute_tags<
                Metavariables, std::decay_t<decltype(element_box)>>>(
                make_not_null(&element_box));
        observation_box_region.reset();
        for (const Event* const event : host_events) {
          call_with_dynamic_type<void, events_on_elements<Metavariables>>(
              event, [&]<typename EventType>(EventType* const derived_event) {
                // One profiling region per type of event, named without its
                // template arguments
                static const std::string region_name = []() {
                  const std::string name = pretty_type::name<EventType>();
                  return "patches::element_boxes::event::" +
                         name.substr(0, name.find('<'));
                }();
                const Kokkos::Profiling::ScopedRegion event_region{region_name};
                mutate_apply(*derived_event, make_not_null(&observation_box),
                             cache, element_id,
                             static_cast<const ParallelComponent*>(nullptr),
                             observation_value);
              });
        }
      });
}

// Runs the `events` that integrate tensors over the volume of the elements
// (see `IntegratesOverVolume`), e.g. the measurements of
// control systems, each with the time its trigger fired before: the patch
// integrates over all its elements and contributes the sum once, as the
// event of an element contributes the integrals over the element. The
// integrals are evaluated on the device if `Tags::ObserveOnDevice` and the
// patch can evaluate the integrands there, and else on the DataBox of each
// element. With `dense_output`, from the last dense output of the patch.
template <typename PatchTag, typename ParallelComponent, typename DbTagsList,
          typename Metavariables>
void run_volume_integrals_events(
    const std::vector<std::pair<const Event*, std::optional<double>>>& events,
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    Parallel::GlobalCache<Metavariables>& cache, const int array_index,
    const bool dense_output) {
  using Patch = typename PatchTag::type;
  const Kokkos::Profiling::ScopedRegion region{
      "patches::run_volume_integrals_events"};
  const double time = db::get<::Tags::Time>(*box);
  const auto sender =
      Parallel::get_parallel_component<ParallelComponent>(cache)[array_index];
  for (const auto& [event, previous_trigger_time] : events) {
    call_with_dynamic_type<void, volume_integrals_events<Metavariables>>(
        event, [&]<typename EventType>(const EventType* const /*meta*/) {
          typename EventType::integrals_type integrals{};
          bool on_device = false;
          if constexpr (ComputesDeviceObservables<Patch>) {
            using Sources = typename Patch::device_observable_sources;
            using Universe = device_compute::universe<Metavariables, Sources>;
            constexpr bool evaluable = []<typename... Tags>(
                                           tmpl::list<Tags...> /*meta*/) {
              return (... and
                      device_compute::is_provided<device_compute::base_of<Tags>,
                                                  Sources, Universe>);
            }(typename EventType::volume_integrand_tags{});
            if constexpr (evaluable) {
              if (Parallel::get<Tags::ObserveOnDevice>(cache)) {
                on_device = true;
                integrals = volume_integrals_on_device<PatchTag, EventType>(
                    box, cache, dense_output);
              }
            }
          }
          if (not on_device) {
            for_each_element_box<PatchTag>(
                box, cache,
                dense_output ? ElementVariables::DenseOutput
                             : ElementVariables::State,
                [&](const auto& /*element_id*/, auto& element_box) {
                  auto observation_box =
                      make_observation_box<element_observation_compute_tags<
                          Metavariables, std::decay_t<decltype(element_box)>>>(
                          make_not_null(&element_box));
                  add_integrals(make_not_null(&integrals),
                                EventType::integrals(observation_box));
                });
          }
          EventType::contribute(cache, sender, time, previous_trigger_time,
                                integrals);
        });
  }
}

// Requests a slab size for the elements of the patch as
// `Events::ChangeSlabSize` does for each element, with the step choosers on
// the DataBox of each element, and contributes it to the reduction of the
// `Metavariables::slab_size_reduction`, which combines it with the requests
// of the DG elements
template <typename PatchTag, typename ParallelComponent, typename DbTagsList,
          typename Metavariables>
void change_slab_size(const ::Events::ChangeSlabSize& event,
                      const gsl::not_null<db::DataBox<DbTagsList>*> box,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const int array_index) {
  static_assert(
      ::Events::ChangeSlabSize_detail::has_slab_size_reduction<Metavariables>(),
      "Executables with patches must set Metavariables::slab_size_reduction to "
      "patches::SlabSizeReduction so that the DgElementArray and the patches "
      "agree on the slab size.");
  for (const auto& step_chooser : event.step_choosers()) {
    // These use the history and the error estimates of the elements, or
    // local time stepping, which the DataBoxes of the elements don't hold
    // (see `Patch::make_element_box`)
    if (const std::string name =
            pretty_type::get_runtime_type_name(*step_chooser);
        name.find("PreventRapidIncrease") != std::string::npos or
        name.find("ErrorControl") != std::string::npos or
        name.find("FixedLtsRatio") != std::string::npos) {
      ERROR("Patches don't support the "
            << pretty_type::get_runtime_type_name(*step_chooser)
            << " step chooser in ChangeSlabSize.");
    }
  }
  const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(*box);
  TimeStepRequestProcessor step_requests(time_step_id.time_runs_forward());
  // The step choosers (e.g. `Cfl`) may use the variables. Patches step
  // globally.
  for_each_element_box<PatchTag>(
      box, cache, ElementVariables::State,
      [&](const auto& /*element_id*/, const auto& element_box) {
        event.request_step_sizes(make_not_null(&step_requests), time_step_id,
                                 LtsMode::Off, element_box);
      });
  event.template send_requests<ParallelComponent>(
      box, cache, array_index, event.slab_to_change(time_step_id),
      step_requests);
}
}  // namespace EventActions_detail

/*!
 * \brief Register the elements of the patch with the observers for their
 * events, as `observers::Actions::RegisterEventsWithObservers` does for an
 * element.
 *
 * \details Each element of the patch is a contributor with its `ElementId`,
 * and its registration is that of the events on its DataBox (see
 * `Patch::make_element_box`), so its data is written and reduced like that of
 * an element of the `DgElementArray`. The exception are the events that the
 * patch observes on the device (see `Tags::ObserveOnDevice`): the patch
 * contributes their norms once per subfile, for all its elements that
 * contribute to it, so only the first of these registers.
 */
template <typename PatchTag>
struct RegisterWithObservers {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    constexpr size_t dim = Metavariables::volume_dim;
    if (array_index == 0 and Parallel::get<Tags::ObserveOnDevice>(cache)) {
      report_device_observations<typename PatchTag::type, Metavariables>(box);
    }
    auto& observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    using Patch = typename PatchTag::type;
    const auto observes_on_device = [&cache](const Event& event) {
      if constexpr (ComputesDeviceObservables<Patch>) {
        return Parallel::get<Tags::ObserveOnDevice>(cache) and
               evolution::patches::observes_on_device<Patch, Metavariables>(
                   event);
      } else {
        (void)cache;
        (void)event;
        return false;
      }
    };
    // The events observed on the device and the subfiles that an element has
    // registered for
    std::vector<std::pair<const Event*, observers::ObservationKey>>
        registered_on_device{};
    // Registration doesn't use the variables
    for_each_element_box<PatchTag>(
        make_not_null(&box), cache, ElementVariables::WithoutValues,
        [&](const ElementId<dim>& element_id, const auto& element_box) {
          const auto register_event = [&](const Event& event) {
            if (not runs_on_elements<Metavariables>(event)) {
              return;
            }
            if (const auto type_and_key =
                    observers::get_registration_observation_type_and_key(
                        event, element_box);
                type_and_key.has_value()) {
              if (observes_on_device(event)) {
                std::pair event_and_key{&event, type_and_key->second};
                if (alg::found(registered_on_device, event_and_key)) {
                  return;
                }
                registered_on_device.push_back(std::move(event_and_key));
              }
              Parallel::simple_action<
                  observers::Actions::RegisterContributorWithObserver>(
                  observer, type_and_key->second,
                  Parallel::ArrayComponentId{
                      std::add_pointer_t<ParallelComponent>{nullptr},
                      Parallel::ArrayIndex<ElementId<dim>>{element_id}},
                  type_and_key->first);
            }
          };
          db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>(
              box)
              .for_each_event(register_event);
          db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
              box)
              .for_each_event(register_event);
          db::get<::Tags::EventsAndDenseTriggers>(box).for_each_event(
              register_event);
        });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Run the events of the patch whose trigger fires, at the start of a
 * step.
 *
 * \details
 * - `Events::Completion` stops the evolution of the patch.
 * - Events that each element contributes data to (see
 *   `RunsOnElements`) run on the DataBox of each element of the patch
 *   (see `Patch::make_element_box`), as on the elements of the
 *   `DgElementArray`.
 * - `Events::ChangeSlabSize` evaluates its step choosers on the DataBox of
 *   each element, and the requests are combined with those of the DG elements
 *   (see `SlabSizeReduction`). `::Actions::ChangeSlabSize` changes the slab
 *   size.
 *
 * Other events and `Triggers::OnSubsteps` are an error, since the DG elements
 * would wait for the patches forever (e.g. in a reduction). The observation
 * value is the time.
 */
template <Triggers::WhenToCheck WhenToCheck, typename PatchTag>
struct RunEventsAndTriggers {
  // The first patch combines reductions over all elements, e.g. of the slab
  // size requests (see `CombinedReduction`)
  using simple_tags = tmpl::list<Tags::PendingReductions>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    if (db::get<::Tags::TimeStepId>(box).substep() != 0) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const Kokkos::Profiling::ScopedRegion region{
        "patches::RunEventsAndTriggers"};
    const TriggeredEvents events = triggered_events<Metavariables>(
        db::get<::Tags::EventsAndTriggers<WhenToCheck>>(box), box);
    EventActions_detail::run_events_on_elements<PatchTag, ParallelComponent>(
        events.on_elements, make_not_null(&box), cache,
        Event::ObservationValue{db::tag_name<::Tags::Time>(),
                                db::get<::Tags::Time>(box)});
    for (const auto* const change_slab_size : events.slab_size_changes) {
      EventActions_detail::change_slab_size<PatchTag, ParallelComponent>(
          *change_slab_size, make_not_null(&box), cache, array_index);
    }
    if (events.completed) {
      return {Parallel::AlgorithmExecution::Pause, std::nullopt};
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Run the events of the dense triggers whose times are in the current
 * step, on the elements of the patch, as
 * `evolution::Actions::RunEventsAndDenseTriggers` does for an element.
 *
 * \details Goes between `ComputeTimeDerivative` and `UpdateVariables`, where
 * the host does dense output. Patches use global time stepping, so the
 * triggers are checked once for all elements of the patch, with the DataBox of
 * the patch, whose `::Tags::Time` is the time of the trigger while they are
 * checked and the events run. For each time:
 * - If an event needs the evolved variables, the patch computes them by dense
 *   output (see `Patch::dense_output`), into its scratch data, after checking
 *   that the functions of time are ready at that time. If dense output isn't
 *   possible yet in this substep, e.g. before the corrector of
 *   `TimeSteppers::AdamsMoultonPc`, the events run at a later substep of the
 *   step, as on the host.
 * - `Events::Completion` stops the evolution of the patch.
 * - `ObserveFields` and `ObserveNorms` run on the DataBox of each element of
 *   the patch at the time of the trigger (see `Patch::make_element_box`), with
 *   the dense output. The observation value is the time.
 * - Events that integrate tensors over the volume of the elements, e.g. the
 *   measurements of control systems
 *   (`control_system::measurements::VolumeIntegralsEvent`), are integrated
 *   over all elements of the patch (on the device if `Tags::ObserveOnDevice`
 *   and the patch can evaluate the integrands there), and the patch
 *   contributes the sum once with the time the trigger fired before (see
 *   `EventActions_detail::run_volume_integrals_events`).
 *
 * Other events are an error, since the DG elements would wait for the patches
 * forever (e.g. in a reduction). Dense triggers are skipped during the self
 * start, as on the host.
 */
template <typename PatchTag>
struct RunEventsAndDenseTriggers {
  // The first patch combines reductions over all elements, e.g. of the
  // measurements (see `CombinedReduction`)
  using simple_tags = tmpl::list<Tags::PendingReductions>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      ActionList /*meta*/, const ParallelComponent* const component) {
    constexpr size_t dim = Metavariables::volume_dim;
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    if (time_step_id.slab_number() < 0) {
      // As the elements, skip dense output during the self start
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const Kokkos::Profiling::ScopedRegion region{
        "patches::RunEventsAndDenseTriggers"};
    auto& events_and_dense_triggers =
        db::get_mutable_reference<::Tags::EventsAndDenseTriggers>(
            make_not_null(&box));
    // The events are checked whether or not they trigger, so that a run fails
    // early
    events_and_dense_triggers.for_each_event([](const Event& event) {
      if (dynamic_cast<const Events::Completion*>(&event) == nullptr and
          not runs_dense_on_elements<Metavariables>(event) and
          not integrates_over_volume<Metavariables>(event)) {
        error_about_unsupported_dense_event<Metavariables>(event);
      }
    });

    const double step_end =
        (time_step_id.step_time() + db::get<::Tags::TimeStep>(box)).value();
    const evolution_less_equal<double> before_equal{
        time_step_id.time_runs_forward()};
    // The time of the substep, restored on every return
    const double substep_time = db::get<::Tags::Time>(box);
    const auto set_time = [&box](const double time) {
      if (db::get<::Tags::Time>(box) != time) {
        db::mutate<::Tags::Time>(
            [&time](const gsl::not_null<double*> box_time) {
              *box_time = time;
            },
            make_not_null(&box));
      }
    };
    bool completed = false;
    const auto finish = [&](const Parallel::AlgorithmExecution execution)
        -> Parallel::iterable_action_return_t {
      set_time(substep_time);
      return {completed ? Parallel::AlgorithmExecution::Pause : execution,
              std::nullopt};
    };

    for (;;) {
      const double next_trigger = events_and_dense_triggers.next_trigger(box);
      if (before_equal(step_end, next_trigger)) {
        return finish(Parallel::AlgorithmExecution::Continue);
      }
      set_time(next_trigger);

      const auto triggered = events_and_dense_triggers.is_ready(
          make_not_null(&box), cache, array_index, component);
      using TriggeringState = std::decay_t<decltype(triggered)>;
      if (triggered == TriggeringState::NotReady) {
        return finish(Parallel::AlgorithmExecution::Retry);
      }
      const bool dense_output =
          triggered == TriggeringState::NeedsEvolvedVariables;
      if (dense_output) {
        // As `domain::CheckFunctionsOfTimeAreReadyPostprocessor` on the host
        if (not ::domain::functions_of_time_are_ready_algorithm_callback<
                ::domain::Tags::FunctionsOfTime, dim>(
                cache, array_index, component, next_trigger)) {
          return finish(Parallel::AlgorithmExecution::Retry);
        }
        const auto& time_stepper =
            db::get<::Tags::TimeStepper<TimeStepper>>(box);
        const bool dense_output_succeeded =
            with_state<PatchTag>(box, [&](const auto&... state) {
              return db::mutate<PatchTag, history_tag<DbTagsList>>(
                  [&](const auto patch, const auto history) {
                    return patch->dense_output(
                        Parallel::device_execution_space(), state..., history,
                        time_stepper, next_trigger);
                  },
                  make_not_null(&box));
            });
        if (not dense_output_succeeded) {
          // The events run at a later substep of the step
          return finish(Parallel::AlgorithmExecution::Continue);
        }
      }

      std::vector<const Event*> element_events{};
      std::vector<std::pair<const Event*, std::optional<double>>>
          volume_integrals_events{};
      events_and_dense_triggers.for_each_triggered_event(
          [&](const Event& event,
              const std::optional<double>& previous_trigger_time) {
            if (dynamic_cast<const Events::Completion*>(&event) != nullptr) {
              completed = true;
            } else if (integrates_over_volume<Metavariables>(event)) {
              volume_integrals_events.emplace_back(&event,
                                                   previous_trigger_time);
            } else {
              element_events.push_back(&event);
            }
          });
      EventActions_detail::run_events_on_elements<PatchTag, ParallelComponent>(
          element_events, make_not_null(&box), cache,
          Event::ObservationValue{db::tag_name<::Tags::Time>(), next_trigger},
          dense_output);
      EventActions_detail::run_volume_integrals_events<PatchTag,
                                                       ParallelComponent>(
          volume_integrals_events, make_not_null(&box), cache, array_index,
          dense_output);
      if (not events_and_dense_triggers.reschedule(make_not_null(&box), cache,
                                                   array_index, component)) {
        return finish(Parallel::AlgorithmExecution::Retry);
      }
    }
  }
};
}  // namespace evolution::patches::Actions
#endif  // SPECTRE_KOKKOS
