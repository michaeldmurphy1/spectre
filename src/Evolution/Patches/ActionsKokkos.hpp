// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <concepts>
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
#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/Patches/DeviceObservablesKokkos.hpp"
#include "Evolution/Patches/InboxTags.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Evolution/Patches/SlabSize.hpp"
#include "Evolution/Patches/Tags.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
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
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/ErrorIfDataTooBig.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/OnSubsteps.hpp"
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

/// Actions of the patches (see `PatchArray`)
///
/// The actions don't depend on the method or the system: `PatchTag` holds the
/// patch of a system and a method (e.g.
/// `grmhd::ValenciaDivClean::fd_patch::Patch`), which provides the operations
/// that the actions call (see the patch section of the GPU support guide). The
/// state of the elements is in the DataBox with the patch's `state_tags`, and
/// the actions pass it to the patch.
namespace evolution::patches::Actions {
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
 * \brief Send the boundary data of the patch's elements to their neighbors
 * outside the patch: elements of the `DgElementArray`, or elements of other
 * patches.
 *
 * The data is sent with the current `TimeStepId`, as elements send it, so the
 * neighbors in the `DgElementArray` receive it as from an element there.
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
    const auto& patch_table =
        db::get<typename Metavariables::patch_table_tag>(box);
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
        const auto location = patch_table.location(message.receiver);
        ASSERT(location.has_value(),
               "The neighbor " << message.receiver << " isn't in a patch");
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
 * neighbors outside the patch and pass it to the patch (`Patch::receive`).
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
    if (not db::get<PatchTag>(box).received_all(received)) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    db::mutate<PatchTag>(
        [&received](const auto patch) {
          patch->receive(Parallel::device_execution_space(), received);
        },
        make_not_null(&box));
    if (messages != inbox.end()) {
      inbox.erase(messages);
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Compute the time derivative of the current substep of all elements
 * of the patch and record it in the time stepper history (see
 * `Patch::compute_time_derivative`).
 *
 * \details The substep is completed by `UpdateVariables`. Dense output, which
 * needs the recorded history but not the update, goes in between, as on the
 * host.
 */
template <typename PatchTag>
struct ComputeTimeDerivative {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Kokkos::Profiling::ScopedRegion region{
        "patches::ComputeTimeDerivative"};
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    detail::mutate_with_state<PatchTag, detail::history_tag<DbTagsList>>(
        make_not_null(&box),
        [&](const auto patch, const auto... state_and_history) {
          patch->compute_time_derivative(
              Parallel::device_execution_space(), state_and_history...,
              time_step_id,
              Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
        });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \brief Update the variables of all elements of the patch to the end of the
/// current substep (see `Patch::update_variables`), after
/// `ComputeTimeDerivative`.
template <typename PatchTag>
struct UpdateVariables {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Kokkos::Profiling::ScopedRegion region{"patches::UpdateVariables"};
    const auto& time_stepper = db::get<::Tags::TimeStepper<TimeStepper>>(box);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    const TimeDelta& time_step = db::get<::Tags::TimeStep>(box);
    detail::mutate_with_state<PatchTag, detail::history_tag<DbTagsList>>(
        make_not_null(&box),
        [&](const auto patch, const auto... state_and_history) {
          patch->update_variables(Parallel::device_execution_space(),
                                  state_and_history..., time_stepper,
                                  time_step_id, next_time_step_id, time_step);
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
// `DgElementArray` do, on their DataBox (see `Patch::make_element_box`): events
// that
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

// The events that the elements of a patch run at the times of dense triggers,
// on their DataBox with the dense output (a subset of `RunsOnElements`)
template <typename T>
struct RunsDenseOnElements : std::false_type {};
template <size_t Dim, typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag>
struct RunsDenseOnElements<::dg::Events::ObserveFields<
    Dim, Fields, NonTensorComputeTags, ArraySectionIdTag>> : std::true_type {};
template <typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct RunsDenseOnElements<::Events::ObserveNorms<
    Fields, NonTensorComputeTags, ArraySectionIdTag, OptionName>>
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

// Whether the DataBox of the elements (see `Patch::make_element_box`) has the
// item of the `ComputeTag` or one of its subitems, e.g. the geometry for
// observations, which the events then don't compute again
//
// (A partial specialization rather than a lambda in the template argument,
// since nvcc's front end crashes on the lambda.)
template <typename ElementBox, typename Subitems>
struct any_in_element_box;

template <typename ElementBox, typename... Subitems>
struct any_in_element_box<ElementBox, tmpl::list<Subitems...>>
    : std::bool_constant<(false or ... or
                          db::tag_is_retrievable_v<Subitems, ElementBox>)> {};

template <typename ComputeTag, typename ElementBox>
struct in_element_box
    : std::bool_constant<
          db::tag_is_retrievable_v<typename ComputeTag::base, ElementBox> or
          any_in_element_box<ElementBox,
                             typename db::Subitems<ComputeTag>::type>::value> {
};

// The `observation_compute_tags` of the items the DataBox of the elements
// doesn't have
template <typename Metavariables, typename ElementBox>
using element_observation_compute_tags =
    tmpl::remove_if<observation_compute_tags<Metavariables>,
                    in_element_box<tmpl::_1, tmpl::pin<ElementBox>>>;

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

template <typename Metavariables>
bool runs_dense_on_elements(const Event& event) {
  bool result = false;
  tmpl::for_each<tmpl::filter<events_on_elements<Metavariables>,
                              RunsDenseOnElements<tmpl::_1>>>(
      [&]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
        result = result or dynamic_cast<const EventType*>(&event) != nullptr;
      });
  return result;
}

// Errors because patches don't run the `event`, so the DG elements would wait
// for them forever (e.g. in a reduction or an interpolation)
[[noreturn]] inline void error_about_unsupported_event(const Event& event) {
  ERROR(
      "FD patches only run Completion, ChangeSlabSize, ObserveFields, "
      "ObserveNorms, ObserveTimeStep and ErrorIfDataTooBig events, but got "
      << pretty_type::get_runtime_type_name(event));
}

[[noreturn]] inline void error_about_unsupported_dense_event(
    const Event& event) {
  ERROR(
      "FD patches only run Completion, ObserveFields and ObserveNorms events "
      "in EventsAndDenseTriggers, but got "
      << pretty_type::get_runtime_type_name(event));
}

// Where `for_each_element_box` gets the variables of the elements from
enum class ElementVariables {
  // The right size but no meaningful values
  WithoutValues,
  // The state of the elements at the current substep
  State,
  // The last dense output of the patch (see `Patch::dense_output`)
  DenseOutput
};

// Calls `function(element_id, box)` with the DataBox of each element of the
// patch (see `Patch::make_element_box`) at the time in the DataBox of the
// patch. The `variables` of the elements are copied from the device, once for
// all elements. `ElementVariables::WithoutValues` is enough for events whose
// `needs_evolved_variables()` is false (e.g. `ObserveTimeStep` only counts the
// grid points) and saves most of the cost of the DataBoxes.
template <typename PatchTag, typename DbTagsList, typename Metavariables,
          typename Function>
void for_each_element_box(const gsl::not_null<db::DataBox<DbTagsList>*> box_ptr,
                          Parallel::GlobalCache<Metavariables>& cache,
                          const ElementVariables variables_source,
                          const Function& function) {
  constexpr size_t dim = Metavariables::volume_dim;
  const Kokkos::Profiling::ScopedRegion region{"patches::element_boxes"};
  const auto& box = *box_ptr;
  const auto& patch = db::get<PatchTag>(box);
  // Report failures of the last substep before anything sees its state
  patch.check_last_step(Parallel::device_execution_space());
  patch.prepare_element_boxes(
      Parallel::device_execution_space(), db::get<::Tags::Time>(box),
      Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
  const auto& domain = db::get<::domain::Tags::Domain<dim>>(box);
  std::vector<typename std::decay_t<decltype(patch)>::ElementVariables>
      variables{};
  if (variables_source == ElementVariables::WithoutValues) {
    variables = patch.element_variables_without_values();
  } else {
    const Kokkos::Profiling::ScopedRegion copy_region{
        "patches::element_boxes::copy_variables"};
    const double time = db::get<::Tags::Time>(box);
    // The patch copies the variables into its host memory
    variables = with_state<PatchTag>(box, [&](const auto&... state) {
      return db::mutate<PatchTag>(
          [&](const auto mutable_patch) {
            return variables_source == ElementVariables::State
                       ? mutable_patch->element_variables(
                             Parallel::device_execution_space(), state...)
                       : mutable_patch->dense_element_variables(
                             Parallel::device_execution_space(), state...,
                             time);
          },
          box_ptr);
    });
  }
  const auto& element_ids = patch.patch_elements().element_ids();
  for (size_t slot = 0; slot < element_ids.size(); ++slot) {
    std::optional<Kokkos::Profiling::ScopedRegion> make_box_region{
        "patches::element_boxes::make_box"};
    auto element_box = patch.make_element_box(
        cache, domain, slot, db::get<::Tags::Time>(box),
        db::get<::Tags::TimeStepId>(box), db::get<::Tags::TimeStep>(box),
        std::move(variables[slot]));
    make_box_region.reset();
    function(element_ids[slot], element_box);
  }
}

// Whether `T` is an `Events::ObserveNorms`
template <typename T>
struct IsObserveNorms : std::false_type {};
template <typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct IsObserveNorms<::Events::ObserveNorms<Fields, NonTensorComputeTags,
                                             ArraySectionIdTag, OptionName>>
    : std::true_type {};

template <typename Metavariables>
using observe_norms_events =
    tmpl::filter<events_on_elements<Metavariables>, IsObserveNorms<tmpl::_1>>;

// Whether patches of type `Patch` can compute observables on the device (see
// `Patch::device_observables`)
template <typename Patch>
concept ComputesDeviceObservables = requires(const std::string& name) {
  { Patch::is_device_observable(name) } -> std::same_as<bool>;
};

// The tensors of the `ObserveNorms` `event` that patches of type `Patch` can't
// compute on the device, or `std::nullopt` if `event` isn't an
// `ObserveNorms` that patches can observe on the device at all
template <typename Patch, typename Metavariables>
std::optional<std::vector<std::string>> tensors_not_observable_on_device(
    const Event& event) {
  if constexpr (not ComputesDeviceObservables<Patch>) {
    (void)event;
    return std::nullopt;
  } else {
    std::optional<std::vector<std::string>> result{};
    tmpl::for_each<observe_norms_events<Metavariables>>(
        [&]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
          const auto* const derived_event =
              dynamic_cast<const EventType*>(&event);
          // Sections need the DataBox of the elements
          if (derived_event != nullptr and
              std::is_same_v<typename EventType::array_section_id_tag, void>) {
            result = std::vector<std::string>{};
            for (const std::string& name : derived_event->tensor_names()) {
              if (not Patch::is_device_observable(name)) {
                result->push_back(name);
              }
            }
          }
        });
    return result;
  }
}

// Whether the patch computes the observations of `event` on the device:
// `ObserveNorms` of tensors the patch can all compute on the device
template <typename Patch, typename Metavariables>
bool observes_on_device(const Event& event) {
  const auto missing =
      tensors_not_observable_on_device<Patch, Metavariables>(event);
  return missing.has_value() and missing->empty();
}

// Runs the `ObserveNorms` `events` (see `observes_on_device`) on the device:
// the patch computes the observed tensors of all its elements on the device,
// they are reduced there (see `norm_statistics_on_device`) and only the
// results of each element are copied to the host and contributed to the
// observer, as if the event had run on the DataBox of the element. With
// `dense_output`, the tensors are computed from the last dense output of the
// patch.
template <typename PatchTag, typename ParallelComponent, typename DbTagsList,
          typename Metavariables>
void observe_norms_on_device(const std::vector<const Event*>& events,
                             const gsl::not_null<db::DataBox<DbTagsList>*> box,
                             Parallel::GlobalCache<Metavariables>& cache,
                             const Event::ObservationValue& observation_value,
                             const bool dense_output) {
  constexpr size_t dim = Metavariables::volume_dim;
  const Kokkos::Profiling::ScopedRegion region{
      "patches::observe_norms_on_device"};
  const auto& exec = Parallel::device_execution_space();
  // Report failures of the last substep before anything sees its state
  db::get<PatchTag>(*box).check_last_step(exec);

  // The tensors of all events, each computed once
  std::vector<std::string> tensor_names{};
  const auto for_each_event = [&events](const auto& function) {
    for (const Event* const event : events) {
      call_with_dynamic_type<void, observe_norms_events<Metavariables>>(
          event, function);
    }
  };
  for_each_event([&tensor_names](const auto* const derived_event) {
    for (const std::string& name : derived_event->tensor_names()) {
      if (not alg::found(tensor_names, name)) {
        tensor_names.push_back(name);
      }
    }
  });

  const double time = db::get<::Tags::Time>(*box);
  // The observables are memory of the patch, valid until the next call
  const evolution::patches::DeviceObservables& observables =
      *with_state<PatchTag>(*box, [&](const auto&... state) {
        return db::mutate<PatchTag>(
            [&](const auto mutable_patch) {
              return &mutable_patch->device_observables(
                  exec, tensor_names, state..., dense_output, time,
                  Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
            },
            box);
      });
  const auto statistics =
      evolution::patches::norm_statistics_on_device(exec, observables);

  const Kokkos::Profiling::ScopedRegion contribute_region{
      "patches::observe_norms_on_device::contribute"};
  const auto& patch = db::get<PatchTag>(*box);
  const auto& element_ids = patch.patch_elements().element_ids();
  const auto& domain = db::get<::domain::Tags::Domain<dim>>(*box);
  const size_t number_of_points =
      patch.patch_elements().subcell_mesh().number_of_grid_points();
  // The last component is the volume
  const size_t volume_component = statistics.extent(1) - 1;
  // The names and statistics of the tensors, whose statistics are replaced by
  // those of each element in turn
  ::Events::NormStatisticsOfTensors tensors_statistics{};
  for (const auto& tensor : observables.tensors) {
    tensors_statistics.emplace(
        tensor.name,
        std::pair{tensor.component_names, std::vector<::Events::NormStatistics>(
                                              tensor.component_names.size())});
  }
  for (size_t slot = 0; slot < element_ids.size(); ++slot) {
    for (const auto& tensor : observables.tensors) {
      auto& component_statistics = tensors_statistics.at(tensor.name).second;
      for (size_t c = 0; c < component_statistics.size(); ++c) {
        component_statistics[c] = statistics(slot, tensor.first_component + c);
      }
    }
    for_each_event([&](const auto* const derived_event) {
      derived_event->contribute_statistics(
          cache, element_ids[slot],
          static_cast<const ParallelComponent*>(nullptr), observation_value,
          observers::get_section_observation_key<void>(*box), domain,
          number_of_points, statistics(slot, volume_component).integral,
          tensors_statistics);
    });
  }
}

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

// Prints which `ObserveNorms` events patches observe on the device and, for
// the others, which tensors they can't compute there (see
// `observes_on_device`), so that a slow observation isn't a surprise
template <typename Patch, typename Metavariables, typename DbTagsList>
void report_device_observations(const db::DataBox<DbTagsList>& box) {
  const auto report = [](const Event& event) {
    tmpl::for_each<observe_norms_events<Metavariables>>(
        [&event]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
          const auto* const derived_event =
              dynamic_cast<const EventType*>(&event);
          if (derived_event == nullptr) {
            return;
          }
          std::string tensors{};
          for (const std::string& name : derived_event->tensor_names()) {
            tensors += (tensors.empty() ? "" : ", ") + name;
          }
          const auto missing =
              tensors_not_observable_on_device<Patch, Metavariables>(event);
          if (missing.has_value() and missing->empty()) {
            Parallel::printf("Patches observe the norms of %s on the device.\n",
                             tensors);
          } else {
            std::string reason{};
            if (missing.has_value()) {
              for (const std::string& name : *missing) {
                reason += (reason.empty() ? "" : ", ") + name;
              }
              reason = "they can't compute " + reason + " on the device";
            } else {
              reason = ComputesDeviceObservables<Patch>
                           ? "the event uses array sections"
                           : "they can't compute observables on the device";
            }
            Parallel::printf(
                "Patches observe the norms of %s on the host, because %s.\n",
                tensors, reason);
          }
        });
  };
  db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>(box)
      .for_each_event(report);
  db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(box)
      .for_each_event(report);
  db::get<::Tags::EventsAndDenseTriggers>(box).for_each_event(report);
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
    synchronization_required =
        synchronization_required or step_chooser->uses_local_data();
  }
  // The step choosers (e.g. `Cfl`) may use the variables
  for_each_element_box<PatchTag>(
      box, cache, ElementVariables::State,
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
 * `Patch::make_element_box`), so its data is written and reduced like that of
 * an
 * element of the `DgElementArray`.
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
      detail::report_device_observations<typename PatchTag::type,
                                         Metavariables>(box);
    }
    auto& observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    // Registration doesn't use the variables
    detail::for_each_element_box<PatchTag>(
        make_not_null(&box), cache, detail::ElementVariables::WithoutValues,
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
 *   `detail::RunsOnElements`) run on the DataBox of each element of the patch
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
        "patches::RunEventsAndTriggers"};
    bool completed = false;
    std::vector<const Event*> events_on_elements{};
    std::vector<const ::Events::ChangeSlabSize*> slab_size_changes{};
    db::get<::Tags::EventsAndTriggers<WhenToCheck>>(box)
        .for_each_trigger_and_events([&](const Trigger& trigger,
                                         const auto& events) {
          // Patches only run events at the first substep. The events and
          // triggers are checked whether or not they trigger, so that a
          // run fails early.
          if (dynamic_cast<const ::Triggers::OnSubsteps*>(&trigger) !=
              nullptr) {
            ERROR("FD patches don't support OnSubsteps triggers.");
          }
          for (const auto& event : events) {
            if (dynamic_cast<const Events::Completion*>(event.get()) ==
                    nullptr and
                dynamic_cast<const ::Events::ChangeSlabSize*>(event.get()) ==
                    nullptr and
                not detail::runs_on_elements<Metavariables>(*event)) {
              detail::error_about_unsupported_event(*event);
            }
          }
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
            } else {
              events_on_elements.push_back(event.get());
            }
          }
        });
    detail::run_events_on_elements<PatchTag, ParallelComponent>(
        events_on_elements, make_not_null(&box), cache,
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
 *
 * Other events are an error, since the DG elements would wait for the patches
 * forever (e.g. in a reduction). Dense triggers are skipped during the self
 * start, as on the host. `::Tags::PreviousTriggerTime` isn't set.
 */
template <typename PatchTag>
struct RunEventsAndDenseTriggers {
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
          not detail::runs_dense_on_elements<Metavariables>(event)) {
        detail::error_about_unsupported_dense_event(event);
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
            detail::with_state<PatchTag>(box, [&](const auto&... state) {
              return db::mutate<PatchTag, detail::history_tag<DbTagsList>>(
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

      std::vector<const Event*> events_on_elements{};
      events_and_dense_triggers.for_each_triggered_event(
          [&](const Event& event) {
            if (dynamic_cast<const Events::Completion*>(&event) != nullptr) {
              completed = true;
            } else {
              events_on_elements.push_back(&event);
            }
          });
      detail::run_events_on_elements<PatchTag, ParallelComponent>(
          events_on_elements, make_not_null(&box), cache,
          Event::ObservationValue{db::tag_name<::Tags::Time>(), next_trigger},
          dense_output);
      if (not events_and_dense_triggers.reschedule(make_not_null(&box), cache,
                                                   array_index, component)) {
        return finish(Parallel::AlgorithmExecution::Retry);
      }
    }
  }
};

namespace detail {
template <typename Action>
struct const_global_cache_tags_of {
  using type = tmpl::list<>;
};
template <typename Action>
  requires requires { typename Action::const_global_cache_tags; }
struct const_global_cache_tags_of<Action> {
  using type = typename Action::const_global_cache_tags;
};
}  // namespace detail

/*!
 * \brief Run the element action `Action` if the patch holds the zeroth
 * element, as the zeroth element of the `DgElementArray` would.
 *
 * \details Some actions run only on the element for which
 * `is_zeroth_element()` is true, e.g. to change a function of time once
 * (`control_system::Actions::SwitchGridRotationToSettle`). The elements of the
 * patches aren't in the `DgElementArray`, so if the zeroth element is in a
 * patch, no element of the array runs them. This runs `Action` with the
 * DataBox of the patch that holds the zeroth element and its `ElementId` as
 * the array index. `Action` may only read items that the DataBox of the patch
 * has too, e.g. the time, the step and the global cache.
 */
template <typename Action, typename PatchTag>
struct RunOnZerothElement {
  using const_global_cache_tags =
      typename detail::const_global_cache_tags_of<Action>::type;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      const ActionList meta, const ParallelComponent* const component) {
    const auto& element_ids =
        db::get<PatchTag>(box).patch_elements().element_ids();
    for (const auto& element_id : element_ids) {
      if (is_zeroth_element(element_id)) {
        return Action::apply(box, inboxes, cache, element_id, meta, component);
      }
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
}  // namespace evolution::patches::Actions
#endif  // SPECTRE_KOKKOS
