// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/ElementBox.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/FdPatch/PatchElementsKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/SlabSize.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Evolution/DgSubcell/Tags/InitialTciData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/ErrorIfDataTooBig.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
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
#include "Utilities/TypeTraits/IsA.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>

/// Actions of the FD patches (see `FdPatchArray`)
///
/// The actions are generic: `PatchTag` holds the system-specific patch, which
/// provides `prepare_send` and `take_step` (see, e.g.,
/// `grmhd::ValenciaDivClean::fd_patch::Patch`) and its `PatchElements` as
/// `patch_elements()`. The state of the elements is in the DataBox with the
/// patch's `state_tags`, and the actions pass it to the patch.
namespace evolution::dg::subcell::fd_patch::Actions {
namespace detail {
// The time stepper history of the elements of a patch, the only
// `::Tags::DeviceHistoryEvolvedVariables` in its DataBox
template <typename DbTagsList>
using history_tag = tmpl::front<::Tags::get_all_device_history_tags<
    typename db::DataBox<DbTagsList>::tags_list>>;

// The DataBox tags of the state of the elements of the patch in `PatchTag`
template <typename PatchTag>
using state_tags = typename PatchTag::type::state_tags;

// Returns `function(state...)` with the state of the elements of the patch
template <typename PatchTag, typename DbTagsList, typename Function>
decltype(auto) with_state(const db::DataBox<DbTagsList>& box,
                          const Function& function) {
  return [&box, &function]<typename... StateTags>(
             tmpl::list<StateTags...> /*meta*/) -> decltype(auto) {
    return function(db::get<StateTags>(box)...);
  }(state_tags<PatchTag>{});
}

// Returns `db::mutate<PatchTag, StateTags..., OtherTags...>(function, box)`,
// i.e., `function` gets pointers to the patch, the state of its elements and
// the `OtherTags`
template <typename PatchTag, typename... OtherTags, typename DbTagsList,
          typename Function>
decltype(auto) mutate_with_state(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const Function& function) {
  return [&box, &function]<typename... StateTags>(
             tmpl::list<StateTags...> /*meta*/) -> decltype(auto) {
    return db::mutate<PatchTag, StateTags..., OtherTags...>(function, box);
  }(state_tags<PatchTag>{});
}
}  // namespace detail

/*!
 * \brief Send the initial RDMP data (temporal id 0) and TCI decisions
 * (temporal id 1) of the patch's elements to their DG neighbors, as
 * `SetAndCommunicateInitialRdmpData` and `ComputeAndSendTciOnInitialGrid` do
 * for an element on the host with `AlwaysUseSubcells`.
 *
 * \details The elements of a patch never switch to DG, so they don't run the
 * initial TCI and send it their neighbors only for the DG elements' handshake.
 */
template <typename PatchTag, typename DgElementArray>
struct SendInitialTciData {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    constexpr size_t dim = Metavariables::volume_dim;
    using inbox_tag = evolution::dg::subcell::Tags::InitialTciData<dim>;
    auto& dg_proxy = Parallel::get_parallel_component<DgElementArray>(cache);
    const auto& patch = db::get<PatchTag>(box);
    for (auto& message :
         detail::with_state<PatchTag>(box, [&patch](const auto&... state) {
           return patch.initial_rdmp_data(Parallel::device_execution_space(),
                                          state...);
         })) {
      Parallel::receive_data<inbox_tag>(
          dg_proxy[message.receiver], 0,
          std::pair{message.sender, evolution::dg::subcell::InitialTciData{
                                        {}, std::move(message.rdmp_tci_data)}});
      Parallel::receive_data<inbox_tag>(
          dg_proxy[message.receiver], 1,
          std::pair{message.sender,
                    evolution::dg::subcell::InitialTciData{
                        std::decay_t<decltype(patch)>::tci_status_sent, {}}});
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Wait for the initial RDMP data and TCI decisions that the DG
 * neighbors send to the patch's elements during initialization, and discard
 * them, since the elements of a patch never switch to DG.
 */
template <size_t Dim, typename PatchTag>
struct ReceiveInitialTciData {
  using inbox_tags = tmpl::list<Tags::InitialTciDataInbox<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t number_of_dg_faces =
        db::get<PatchTag>(box).patch_elements().number_of_dg_faces();
    if (number_of_dg_faces == 0) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    auto& inbox = tuples::get<Tags::InitialTciDataInbox<Dim>>(inboxes);
    for (const int temporal_id : {0, 1}) {
      const auto messages = inbox.find(temporal_id);
      if (messages == inbox.end() or
          messages->second.size() < number_of_dg_faces) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
      if (messages->second.size() > number_of_dg_faces) {
        ERROR("The elements of the patch received "
              << messages->second.size() << " initial TCI messages with id "
              << temporal_id << " but have " << number_of_dg_faces
              << " DG neighbors");
      }
    }
    inbox.clear();
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Send the boundary data of the patch's elements to their neighbors
 * outside the patch: DG elements of `DgElementArray`, or elements of other
 * patches.
 *
 * The data is sent with the current `TimeStepId`, as DG elements send it, so
 * DG neighbors receive it as from an FD element.
 */
template <typename PatchTag, typename DgElementArray>
struct SendBoundaryData {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    constexpr size_t dim = Metavariables::volume_dim;
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    const size_t integration_order =
        db::get<detail::history_tag<DbTagsList>>(box).integration_order();
    auto outgoing = detail::with_state<PatchTag>(
        box,
        [&box, &next_time_step_id, &integration_order](const auto&... state) {
          return db::mutate<PatchTag>(
              [&](const auto patch) {
                return patch->prepare_send(Parallel::device_execution_space(),
                                           state..., next_time_step_id,
                                           integration_order);
              },
              make_not_null(&box));
        });
    const auto& fd_patches = db::get<Tags::FdPatches<dim>>(box);
    auto& patch_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    auto& dg_proxy = Parallel::get_parallel_component<DgElementArray>(cache);
    for (auto& message : outgoing) {
      if (message.receiver_is_dg) {
        Parallel::receive_data<
            evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<dim,
                                                                      false>>(
            dg_proxy[message.receiver], time_step_id,
            std::pair{message.sender, std::move(message.data)});
      } else {
        const auto location = fd_patches.location(message.receiver);
        ASSERT(location.has_value(),
               "The FD neighbor " << message.receiver << " isn't in a patch");
        Parallel::receive_data<Tags::BoundaryDataInbox<dim>>(
            patch_proxy[static_cast<int>(location->patch)], time_step_id,
            std::tuple{message.receiver, message.sender,
                       std::move(message.data)});
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Wait for the boundary data of the current substep from the
 * neighbors outside the patch and fill the ghost zones.
 */
template <size_t Dim, typename PatchTag>
struct ReceiveBoundaryData {
  using inbox_tags = tmpl::list<Tags::BoundaryDataInbox<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<Tags::BoundaryDataInbox<Dim>>(inboxes);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const auto messages = inbox.find(time_step_id);
    const typename Tags::BoundaryDataInbox<Dim>::messages_type no_messages{};
    const auto& received =
        messages == inbox.end() ? no_messages : messages->second;
    if (not db::get<PatchTag>(box).patch_elements().received_all(received)) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    db::mutate<PatchTag>(
        [&received](const auto patch) {
          patch->patch_elements().receive(
              Parallel::device_execution_space(), received,
              std::decay_t<decltype(*patch)>::number_of_rdmp_values);
        },
        make_not_null(&box));
    if (messages != inbox.end()) {
      inbox.erase(messages);
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \brief Take the current substep of all elements of the patch.
template <typename PatchTag>
struct TakeStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Kokkos::Profiling::ScopedRegion region{"fd_patch::TakeStep"};
    const auto& time_stepper = db::get<::Tags::TimeStepper<TimeStepper>>(box);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    const TimeDelta& time_step = db::get<::Tags::TimeStep>(box);
    detail::mutate_with_state<PatchTag, detail::history_tag<DbTagsList>>(
        make_not_null(&box),
        [&](const auto patch, const auto... state_and_history) {
          patch->take_step(
              Parallel::device_execution_space(), state_and_history...,
              time_stepper, time_step_id, next_time_step_id, time_step,
              Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
        });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief At slab boundaries, report failures of the last substep of the patch
 * (see `Patch::check_last_step`).
 *
 * \details The patch checks each substep when it sends the boundary data of the
 * next one, so the last substep before a phase change, e.g. a checkpoint or
 * the end of the evolution, would go unchecked. Put this before
 * `PhaseControl::Actions::ExecutePhaseChange`. It waits for the device once
 * per slab.
 */
template <typename PatchTag>
struct CheckLastStepAtSlabBoundaries {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<::Tags::TimeStepId>(box).is_at_slab_boundary()) {
      db::get<PatchTag>(box).check_last_step(
          Parallel::device_execution_space());
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

namespace detail {
// The events that the elements of a patch run as the elements of the
// `DgElementArray` do, on their DataBox (see `make_element_box`): events that
// each element contributes data to
template <typename T>
struct RunsOnElements : std::false_type {};
template <size_t Dim, typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag>
struct RunsOnElements<::dg::Events::ObserveFields<
    Dim, Fields, NonTensorComputeTags, ArraySectionIdTag>> : std::true_type {};
template <typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct RunsOnElements<::Events::ObserveNorms<Fields, NonTensorComputeTags,
                                             ArraySectionIdTag, OptionName>>
    : std::true_type {};
template <typename System, typename VariablesTags>
struct RunsOnElements<::Events::ObserveTimeStep<System, VariablesTags>>
    : std::true_type {};
template <size_t Dim, typename Fields, typename NonTensorComputeTags>
struct RunsOnElements<
    ::Events::ErrorIfDataTooBig<Dim, Fields, NonTensorComputeTags>>
    : std::true_type {};

template <typename Metavariables>
using events_on_elements = tmpl::filter<
    tmpl::at<typename Metavariables::factory_creation::factory_classes, Event>,
    RunsOnElements<tmpl::_1>>;

template <typename EventType>
struct get_observation_compute_tags {
  using type = typename EventType::compute_tags_for_observation_box;
};

// The compute tags of the `ObservationBox` of the events, as
// `EventsAndTriggers::run_events`
template <typename Metavariables>
using observation_compute_tags = tmpl::remove_duplicates<tmpl::filter<
    tmpl::flatten<tmpl::transform<events_on_elements<Metavariables>,
                                  get_observation_compute_tags<tmpl::_1>>>,
    db::is_compute_tag<tmpl::_1>>>;

// Whether the elements of a patch run the `event` (see `RunsOnElements`)
template <typename Metavariables>
bool runs_on_elements(const Event& event) {
  bool result = false;
  tmpl::for_each<events_on_elements<Metavariables>>(
      [&]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
        result = result or dynamic_cast<const EventType*>(&event) != nullptr;
      });
  return result;
}

// Prints a warning that FD patches don't run an event, once on each thread
inline void warn_about_unsupported_event(const Event& /*event*/) {
  thread_local bool warned = false;
  if (not warned) {
    warned = true;
    Parallel::printf(
        "Warning: FD patches only run Completion, ChangeSlabSize, "
        "ObserveFields, ObserveNorms, ObserveTimeStep and ErrorIfDataTooBig "
        "events, so the data of their elements is missing from other "
        "events.\n");
  }
}

// Calls `function(element_id, box)` with the DataBox of each element of the
// patch (see `make_element_box`). If `copy_variables`, the variables of the
// elements are copied from the device, once for all elements. Otherwise they
// have the right size but no meaningful values, which is enough for events
// whose `needs_evolved_variables()` is false (e.g. `ObserveTimeStep` only
// counts the grid points) and saves most of the cost of the DataBoxes.
template <typename PatchTag, typename DbTagsList, typename Metavariables,
          typename Function>
void for_each_element_box(const db::DataBox<DbTagsList>& box,
                          Parallel::GlobalCache<Metavariables>& cache,
                          const bool copy_variables, const Function& function) {
  constexpr size_t dim = Metavariables::volume_dim;
  const Kokkos::Profiling::ScopedRegion region{"fd_patch::element_boxes"};
  const auto& patch = db::get<PatchTag>(box);
  // Report failures of the last substep before anything sees its state
  patch.check_last_step(Parallel::device_execution_space());
  const PatchElements& patch_elements = patch.patch_elements();
  const auto& domain = db::get<::domain::Tags::Domain<dim>>(box);
  using ElementVariables =
      typename std::decay_t<decltype(patch)>::ElementVariables;
  std::vector<ElementVariables> variables{};
  if (copy_variables) {
    const Kokkos::Profiling::ScopedRegion copy_region{
        "fd_patch::element_boxes::copy_variables"};
    variables = with_state<PatchTag>(box, [&patch](const auto&... state) {
      return patch.element_variables(Parallel::device_execution_space(),
                                     state...);
    });
  } else {
    const size_t number_of_cells =
        patch_elements.subcell_mesh().number_of_grid_points();
    variables.resize(patch_elements.number_of_elements());
    for (auto& element_variables : variables) {
      element_variables.evolved.initialize(number_of_cells);
      element_variables.primitives.initialize(number_of_cells);
      element_variables.spacetime.initialize(number_of_cells);
    }
  }
  for (size_t slot = 0; slot < patch_elements.number_of_elements(); ++slot) {
    const Element<dim>& element = patch_elements.elements()[slot];
    auto element_box = make_element_box<Metavariables>(
        cache, element, patch_elements.dg_mesh(),
        ElementMap<dim, Frame::Grid>{element.id(),
                                     domain.blocks()[element.id().block_id()]},
        patch_elements.grid_to_inertial_map(slot).get_clone(),
        db::get<::Tags::Time>(box), db::get<::Tags::TimeStepId>(box),
        db::get<::Tags::TimeStep>(box),
        std::decay_t<decltype(patch)>::tci_status_sent,
        std::move(variables[slot].evolved),
        std::move(variables[slot].primitives),
        std::move(variables[slot].spacetime));
    function(element.id(), element_box);
  }
}

// Runs the `events` on each element of the patch, as
// `EventsAndTriggers::run_events` does for an element
template <typename PatchTag, typename ParallelComponent, typename DbTagsList,
          typename Metavariables>
void run_events_on_elements(const std::vector<const Event*>& events,
                            const db::DataBox<DbTagsList>& box,
                            Parallel::GlobalCache<Metavariables>& cache,
                            const Event::ObservationValue& observation_value) {
  if (events.empty()) {
    return;
  }
  const bool copy_variables = alg::any_of(events, [](const Event* const event) {
    return event->needs_evolved_variables();
  });
  for_each_element_box<PatchTag>(
      box, cache, copy_variables,
      [&](const auto& element_id, auto& element_box) {
        const Kokkos::Profiling::ScopedRegion events_region{
            "fd_patch::element_boxes::run_events"};
        auto observation_box =
            make_observation_box<observation_compute_tags<Metavariables>>(
                make_not_null(&element_box));
        for (const Event* const event : events) {
          call_with_dynamic_type<void, events_on_elements<Metavariables>>(
              event, [&](auto* const derived_event) {
                mutate_apply(*derived_event, make_not_null(&observation_box),
                             cache, element_id,
                             static_cast<const ParallelComponent*>(nullptr),
                             observation_value);
              });
        }
      });
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
      "Executables with FD patches must set Metavariables::slab_size_reduction "
      "to fd_patch::SlabSizeReduction so that the DG elements and the patches "
      "agree on the slab size.");
  const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(*box);
  const auto next_changable_slab = time_step_id.is_at_slab_boundary()
                                       ? time_step_id.slab_number()
                                       : time_step_id.slab_number() + 1;
  const auto slab_to_change =
      next_changable_slab + static_cast<int64_t>(event.delay_change());
  const auto current_slab_size = (time_step_id.time_runs_forward() ? 1 : -1) *
                                 time_step_id.step_time().slab().duration();

  TimeStepRequestProcessor step_requests(time_step_id.time_runs_forward());
  bool synchronization_required = false;
  for (const auto& step_chooser : event.step_choosers()) {
    // These use the history and the error estimates of the elements, or
    // local time stepping, which the DataBoxes of the elements don't hold
    // (see `make_element_box`)
    if (const std::string name =
            pretty_type::get_runtime_type_name(*step_chooser);
        name.find("PreventRapidIncrease") != std::string::npos or
        name.find("ErrorControl") != std::string::npos or
        name.find("FixedLtsRatio") != std::string::npos) {
      ERROR("FD patches don't support the "
            << pretty_type::get_runtime_type_name(*step_chooser)
            << " step chooser in ChangeSlabSize.");
    }
    synchronization_required =
        synchronization_required or step_chooser->uses_local_data();
  }
  // The step choosers (e.g. `Cfl`) may use the variables
  for_each_element_box<PatchTag>(
      *box, cache, true,
      [&](const auto& /*element_id*/, const auto& element_box) {
        for (const auto& step_chooser : event.step_choosers()) {
          step_requests.process(step_chooser->desired_step(
              current_slab_size.value(), element_box));
        }
      });

  db::mutate<::Tags::ChangeSlabSize::NumberOfExpectedMessages>(
      [&](const gsl::not_null<std::map<int64_t, size_t>*> expected) {
        ++(*expected)[slab_to_change];
      },
      box);
  if (synchronization_required) {
    Metavariables::slab_size_reduction::template contribute<ParallelComponent>(
        cache,
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        ::Events::ChangeSlabSize::ReductionData(slab_to_change, step_requests));
  } else {
    db::mutate<::Tags::ChangeSlabSize::NewSlabSize>(
        [&](const gsl::not_null<
            std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                sizes) { (*sizes)[slab_to_change].push_back(step_requests); },
        box);
  }
}
}  // namespace detail

/*!
 * \brief Register the elements of the patch with the observers for their
 * events, as `observers::Actions::RegisterEventsWithObservers` does for an
 * element.
 *
 * \details Each element of the patch is a contributor with its `ElementId`,
 * and its registration is that of the events on its DataBox (see
 * `make_element_box`), so its data is written and reduced like that of an
 * element of the `DgElementArray`.
 */
template <typename PatchTag>
struct RegisterWithObservers {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    constexpr size_t dim = Metavariables::volume_dim;
    auto& observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    // Registration doesn't use the variables
    detail::for_each_element_box<PatchTag>(
        box, cache, false,
        [&](const ElementId<dim>& element_id, const auto& element_box) {
          const auto register_event = [&](const Event& event) {
            if (not detail::runs_on_elements<Metavariables>(event)) {
              return;
            }
            if (const auto type_and_key =
                    observers::get_registration_observation_type_and_key(
                        event, element_box);
                type_and_key.has_value()) {
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
 *   `detail::RunsOnElements`) run on the DataBox of each element of the patch
 *   (see `make_element_box`), as on the elements of the `DgElementArray`.
 * - `Events::ChangeSlabSize` evaluates its step choosers on the DataBox of
 *   each element, and the requests are combined with those of the DG elements
 *   (see `SlabSizeReduction`). `::Actions::ChangeSlabSize` changes the slab
 *   size.
 *
 * Other events are skipped with a warning. The observation value is the time.
 */
template <Triggers::WhenToCheck WhenToCheck, typename PatchTag>
struct RunEventsAndTriggers {
  // The first patch combines the slab size requests (see `SlabSizeReduction`)
  using simple_tags =
      tmpl::conditional_t<WhenToCheck == Triggers::WhenToCheck::AtSlabs,
                          tmpl::list<Tags::PendingSlabSizeRequests>,
                          tmpl::list<>>;

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
        "fd_patch::RunEventsAndTriggers"};
    bool completed = false;
    std::vector<const Event*> events_on_elements{};
    std::vector<const ::Events::ChangeSlabSize*> slab_size_changes{};
    db::get<::Tags::EventsAndTriggers<WhenToCheck>>(box)
        .for_each_trigger_and_events(
            [&](const Trigger& trigger, const auto& events) {
              if (not trigger.is_triggered(box)) {
                return;
              }
              for (const auto& event : events) {
                if (dynamic_cast<const Events::Completion*>(event.get()) !=
                    nullptr) {
                  completed = true;
                } else if (const auto* const change_slab_size =
                               dynamic_cast<const ::Events::ChangeSlabSize*>(
                                   event.get());
                           change_slab_size != nullptr) {
                  slab_size_changes.push_back(change_slab_size);
                } else if (detail::runs_on_elements<Metavariables>(*event)) {
                  events_on_elements.push_back(event.get());
                } else {
                  detail::warn_about_unsupported_event(*event);
                }
              }
            });
    detail::run_events_on_elements<PatchTag, ParallelComponent>(
        events_on_elements, box, cache,
        Event::ObservationValue{db::tag_name<::Tags::Time>(),
                                db::get<::Tags::Time>(box)});
    for (const auto* const change_slab_size : slab_size_changes) {
      detail::change_slab_size<PatchTag, ParallelComponent>(
          *change_slab_size, make_not_null(&box), cache, array_index);
    }
    if (completed) {
      return {Parallel::AlgorithmExecution::Pause, std::nullopt};
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Start the self start of the patch: save the evolved and primitive
 * variables of its elements and set the step size of the self start, as
 * `SelfStart::Actions::Initialize` does for an element.
 *
 * \details The rest of the self start is generic, except for
 * `CheckForSelfStartCompletion` (see `self_start_procedure`).
 */
template <typename PatchTag>
struct SelfStartInitialize {
  using simple_tags =
      tmpl::list<SelfStart::Tags::InitialValue<::Tags::TimeStep>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const TimeDelta initial_step = db::get<::Tags::TimeStep>(box);
    const auto values_needed =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box).slab_number() == 0
            ? 0
            : get<TimeSteppers::Tags::FixedOrder>(
                  db::get<::Tags::TimeStepper<TimeStepper>>(box).order()) -
                  1;
    detail::with_state<PatchTag>(box, [&](const auto&... state) {
      db::mutate<PatchTag, SelfStart::Tags::InitialValue<::Tags::TimeStep>,
                 ::Tags::TimeStep>(
          [&](const auto patch, const auto saved_step, const auto time_step) {
            patch->save_initial_values(Parallel::device_execution_space(),
                                       state...);
            *saved_step = std::make_tuple(initial_step);
            *time_step = initial_step / (values_needed + 1);
          },
          make_not_null(&box));
    });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Restore the evolved and primitive variables of the elements of the
 * patch at the end of each order of the self start, and leave the loop of the
 * self start at the label `ExitTag` at its end, as
 * `SelfStart::Actions::CheckForCompletion` does for an element.
 */
template <typename ExitTag, typename PatchTag>
struct CheckForSelfStartCompletion {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const TimeStepId& next_time_step_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    const bool done_with_order = next_time_step_id.is_at_slab_boundary();
    const bool done = next_time_step_id.slab_number() == 0;
    if (done_with_order or done) {
      detail::mutate_with_state<PatchTag>(
          make_not_null(&box), [&](const auto patch, const auto... state) {
            if (done_with_order) {
              patch->restore_initial_values(Parallel::device_execution_space(),
                                            state...);
            }
            if (done) {
              patch->discard_initial_values();
            }
          });
    }
    if (done) {
      return {Parallel::AlgorithmExecution::Continue,
              tmpl::index_of<ActionList, ::Actions::Label<ExitTag>>::value};
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

namespace detail {
struct SelfStartBegin;
struct SelfStartEnd;
}  // namespace detail

/// The self start of a patch (see `SelfStart::self_start_procedure`), with
/// the `StepActions` of the patch. Only saving and restoring the variables of
/// the elements is specific to patches.
template <typename PatchTag, typename StepActions>
using self_start_procedure = tmpl::flatten<tmpl::list<
    SelfStartInitialize<PatchTag>, ::Actions::Label<detail::SelfStartBegin>,
    CheckForSelfStartCompletion<detail::SelfStartEnd, PatchTag>,
    ::Actions::MutateApply<AdvanceTime<>>,
    SelfStart::Actions::CheckForOrderIncrease, StepActions,
    ::Actions::Goto<detail::SelfStartBegin>,
    ::Actions::Label<detail::SelfStartEnd>, SelfStart::Actions::Cleanup,
    ::Actions::MutateApply<AdvanceTime<>>, Parallel::Actions::TerminatePhase>>;
}  // namespace evolution::dg::subcell::fd_patch::Actions
#endif  // SPECTRE_KOKKOS
