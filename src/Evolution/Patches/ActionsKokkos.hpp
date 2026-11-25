// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/Patches/InboxTags.hpp"
#include "Evolution/Patches/PatchState.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>

/// Actions of the patches (see `PatchArray`)
///
/// The actions don't depend on the method or the system: `PatchTag` holds the
/// patch of a system and a method (e.g.
/// `grmhd::ValenciaDivClean::fd_patch::Patch`), which provides the operations
/// that the actions call (see the patch section of the GPU support guide). The
/// state of the elements is in the DataBox with the patch's `state_tags` (see
/// `evolution::patches::with_state`), and the actions pass it to the patch.
/// This file has the actions of a step; see also `EventActionsKokkos.hpp`,
/// `SelfStartKokkos.hpp` and `RunOnZerothElement.hpp`.
namespace evolution::patches::Actions {
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
        db::get<history_tag<DbTagsList>>(box).integration_order();
    auto outgoing = with_state<PatchTag>(
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
    mutate_with_state<PatchTag, history_tag<DbTagsList>>(
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
    mutate_with_state<PatchTag, history_tag<DbTagsList>>(
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
}  // namespace evolution::patches::Actions
#endif  // SPECTRE_KOKKOS
