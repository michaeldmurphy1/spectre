// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Evolution/DgSubcell/Tags/InitialTciData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#ifdef SPECTRE_KOKKOS
/// Actions of the FD patches (see `FdPatchArray`)
///
/// The actions are generic: `PatchTag` holds the system-specific patch, which
/// provides `prepare_send`, `received_all`, `receive` and `take_step` (see,
/// e.g., `grmhd::ValenciaDivClean::fd_patch::Patch`).
namespace evolution::dg::subcell::fd_patch::Actions {
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
         patch.initial_rdmp_data(Parallel::device_execution_space())) {
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
        db::get<PatchTag>(box).number_of_dg_faces();
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
    auto outgoing = db::mutate<PatchTag>(
        [&next_time_step_id](const auto patch) {
          return patch->prepare_send(Parallel::device_execution_space(),
                                     next_time_step_id);
        },
        make_not_null(&box));
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
    const auto& time_stepper = db::get<::Tags::TimeStepper<TimeStepper>>(box);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    const TimeDelta& time_step = db::get<::Tags::TimeStep>(box);
    db::mutate<PatchTag>(
        [&](const auto patch) {
          patch->take_step(
              Parallel::device_execution_space(), time_stepper, time_step_id,
              next_time_step_id, time_step,
              Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

namespace detail {
template <typename T>
struct ObserveFieldsTraits : std::false_type {};

template <size_t Dim, typename... Tensors, typename NonTensorComputeTags,
          typename ArraySectionIdTag>
struct ObserveFieldsTraits<::dg::Events::ObserveFields<
    Dim, tmpl::list<Tensors...>, NonTensorComputeTags, ArraySectionIdTag>>
    : std::true_type {
  using tensors = tmpl::list<Tensors...>;
  using array_section_id_tag = ArraySectionIdTag;
};

// The `dg::Events::ObserveFields` events of the executable
template <typename Metavariables>
using observe_fields_events = tmpl::filter<
    tmpl::at<typename Metavariables::factory_creation::factory_classes, Event>,
    ObserveFieldsTraits<tmpl::_1>>;

// Calls `function` with `event` cast to its `ObserveFields` type and returns
// `true` if it is one of the `observe_fields_events`
template <typename Metavariables, typename Function>
bool visit_observe_fields(const Event& event, const Function& function) {
  bool found = false;
  tmpl::for_each<observe_fields_events<Metavariables>>(
      [&event, &function,
       &found]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
        if (const auto* const observe_fields =
                dynamic_cast<const EventType*>(&event);
            not found and observe_fields != nullptr) {
          static_assert(std::is_void_v<typename ObserveFieldsTraits<
                            EventType>::array_section_id_tag>,
                        "FD patches don't support observing array sections.");
          found = true;
          function(*observe_fields);
        }
      });
  return found;
}

// Prints a warning that FD patches don't run an event, once on each thread
inline void warn_about_unsupported_event(const Event& /*event*/) {
  thread_local bool warned = false;
  if (not warned) {
    warned = true;
    Parallel::printf(
        "Warning: FD patches only run Completion and ObserveFields events, so "
        "the data of their elements is missing from other events, e.g. "
        "ObserveNorms.\n");
  }
}

// Observes the fields of the elements of the patch as `ObserveFields` does for
// each element, so the data of patches and DG elements is in the same files
template <typename ParallelComponent, typename EventType, typename PatchType,
          typename Metavariables>
void observe_fields(const EventType& event, const PatchType& patch,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const Domain<Metavariables::volume_dim>& domain,
                    const double time) {
  constexpr size_t dim = Metavariables::volume_dim;
  if (event.interpolation_mesh().has_value() or
      event.projection_mesh().has_value()) {
    ERROR(
        "FD patches don't support InterpolateToMesh or ProjectToMesh in "
        "ObserveFields.");
  }
  const auto& variables_to_observe = event.variables_to_observe();
  const auto data = patch.observation_data(
      Parallel::device_execution_space(), time,
      Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
  for (size_t slot = 0; slot < patch.number_of_elements(); ++slot) {
    const ElementId<dim>& element_id = patch.element_ids()[slot];
    if (not event.observes_element(domain, element_id)) {
      continue;
    }
    std::vector<TensorComponent> components{};
    // The tensors in the order of `ObserveFields`
    tmpl::for_each<typename ObserveFieldsTraits<EventType>::tensors>(
        [&]<typename TensorTag>(tmpl::type_<TensorTag> /*meta*/) {
          const std::string tag_name = db::tag_name<TensorTag>();
          const auto requested = variables_to_observe.find(tag_name);
          if (requested == variables_to_observe.end()) {
            return;
          }
          if constexpr (tt::is_a_v<Tensor, typename TensorTag::type>) {
            const typename TensorTag::type tensor{};
            for (size_t i = 0; i < tensor.size(); ++i) {
              const std::string component_name =
                  tag_name + tensor.component_suffix(i);
              const auto component = data[slot].find(component_name);
              if (component == data[slot].end()) {
                // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
                [[maybe_unused]] static const bool warned = [&tag_name]() {
                  Parallel::printf(
                      "Warning: FD patches can't observe '%s', so it is "
                      "missing for their elements.\n",
                      tag_name);
                  return true;
                }();
                return;
              }
              if (requested->second == FloatingPointType::Float) {
                components.emplace_back(
                    component_name,
                    std::vector<float>{component->second.begin(),
                                       component->second.end()});
              } else {
                components.emplace_back(component_name, component->second);
              }
            }
          }
        });
    observers::contribute_volume_data<true>(
        cache, observers::ObservationId{time, event.subfile_path() + ".vol"},
        event.subfile_path(),
        Parallel::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<dim>>{element_id}},
        ElementVolumeData{element_id, std::move(components),
                          patch.subcell_mesh()},
        event.dependency());
  }
}
}  // namespace detail

/*!
 * \brief Register the elements of the patch with the observers for the
 * `dg::Events::ObserveFields` events, as `RegisterEventsWithObservers` does
 * for an element.
 *
 * \details Patches observe only `ObserveFields` events (and run `Completion`)
 * at steps and slabs, see `RunEventsAndTriggers`. Each element of the patch is
 * a contributor with its `ElementId`, so its data is written like that of an
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
    const auto& patch = db::get<PatchTag>(box);
    const auto& domain = db::get<::domain::Tags::Domain<dim>>(box);
    auto& observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    const auto register_event = [&](const Event& event) {
      detail::visit_observe_fields<Metavariables>(
          event, [&](const auto& observe_fields) {
            for (const auto& element_id : patch.element_ids()) {
              if (observe_fields.observes_element(domain, element_id)) {
                Parallel::simple_action<
                    observers::Actions::RegisterContributorWithObserver>(
                    observer,
                    observers::ObservationKey{observe_fields.subfile_path() +
                                              ".vol"},
                    Parallel::ArrayComponentId{
                        std::add_pointer_t<ParallelComponent>{nullptr},
                        Parallel::ArrayIndex<ElementId<dim>>{element_id}},
                    observers::TypeOfObservation::Volume);
              }
            }
          });
    };
    db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>(box)
        .for_each_event(register_event);
    db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(box)
        .for_each_event(register_event);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Run the events of the patch whose trigger fires, at the start of a
 * step.
 *
 * \details Patches run only the events whose data they have:
 * `Events::Completion`, which stops the evolution of the patch, and
 * `dg::Events::ObserveFields`, for which the elements of the patch contribute
 * their data on the subcell grid like FD elements of the `DgElementArray`
 * (see `RegisterWithObservers`). Other events are skipped with a warning, e.g.
 * reductions like `Events::ObserveNorms` don't include the elements of
 * patches. The observation value is the time.
 */
template <Triggers::WhenToCheck WhenToCheck, typename PatchTag>
struct RunEventsAndTriggers {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    if (db::get<::Tags::TimeStepId>(box).substep() != 0) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    constexpr size_t dim = Metavariables::volume_dim;
    const auto& patch = db::get<PatchTag>(box);
    const auto& domain = db::get<::domain::Tags::Domain<dim>>(box);
    const double time = db::get<::Tags::Time>(box);
    bool completed = false;
    db::get<::Tags::EventsAndTriggers<WhenToCheck>>(box)
        .for_each_trigger_and_events([&](const Trigger& trigger,
                                         const auto& events) {
          if (not trigger.is_triggered(box)) {
            return;
          }
          for (const auto& event : events) {
            if (dynamic_cast<const Events::Completion*>(event.get()) !=
                nullptr) {
              completed = true;
            } else if (not detail::visit_observe_fields<Metavariables>(
                           *event, [&](const auto& observe_fields) {
                             detail::observe_fields<ParallelComponent>(
                                 observe_fields, patch, cache, domain, time);
                           })) {
              detail::warn_about_unsupported_event(*event);
            }
          }
        });
    if (completed) {
      // A diagnostic, e.g. to compare with `Events::ObserveNorms`, which
      // doesn't include the elements of patches
      Parallel::printf("%s", patch.norms(Parallel::device_execution_space()));
      return {Parallel::AlgorithmExecution::Pause, std::nullopt};
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::subcell::fd_patch::Actions
#endif  // SPECTRE_KOKKOS
