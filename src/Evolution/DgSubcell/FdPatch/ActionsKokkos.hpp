// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// Actions of the FD patches (see `FdPatchArray`)
///
/// The actions are generic: `PatchTag` holds the system-specific patch, which
/// provides `prepare_send`, `received_all`, `receive` and `take_step` (see,
/// e.g., `grmhd::ValenciaDivClean::fd_patch::Patch`).
namespace evolution::dg::subcell::fd_patch::Actions {
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

/*!
 * \brief Stop the evolution of the patch when the trigger of a
 * `Events::Completion` fires, at the start of a step.
 *
 * \details Patches don't run the other events, since their DataBox doesn't
 * hold what the events of an executable use. Only the triggers of the events
 * that include `Events::Completion` are checked. If `PatchTag` isn't `void`,
 * the norms of the patch (`norms()`) are printed when the evolution stops, as
 * a diagnostic until patches support observations.
 */
template <Triggers::WhenToCheck WhenToCheck, typename PatchTag = void>
struct CheckForCompletion {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<::Tags::TimeStepId>(box).substep() != 0) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    bool completed = false;
    db::get<::Tags::EventsAndTriggers<WhenToCheck>>(box)
        .for_each_trigger_and_events(
            [&box, &completed](const Trigger& trigger, const auto& events) {
              if (not completed and
                  alg::any_of(events,
                              [](const auto& event) {
                                return dynamic_cast<const Events::Completion*>(
                                           event.get()) != nullptr;
                              }) and
                  trigger.is_triggered(box)) {
                completed = true;
              }
            });
    if (completed) {
      // A diagnostic until patches support observations
      if constexpr (not std::is_void_v<PatchTag>) {
        Parallel::printf("%s", db::get<PatchTag>(box).norms(
                                   Parallel::device_execution_space()));
      }
      return {Parallel::AlgorithmExecution::Pause, std::nullopt};
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::subcell::fd_patch::Actions
#endif  // SPECTRE_KOKKOS
