// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Patches/PatchState.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace evolution::patches::Actions {
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
    with_state<PatchTag>(box, [&](const auto&... state) {
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
      mutate_with_state<PatchTag>(
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

namespace SelfStart_detail {
struct SelfStartBegin;
struct SelfStartEnd;
}  // namespace SelfStart_detail

/// The self start of a patch (see `SelfStart::self_start_procedure`), with
/// the `StepActions` of the patch. Only saving and restoring the variables of
/// the elements is specific to patches.
template <typename PatchTag, typename StepActions>
using self_start_procedure = tmpl::flatten<tmpl::list<
    SelfStartInitialize<PatchTag>,
    ::Actions::Label<SelfStart_detail::SelfStartBegin>,
    CheckForSelfStartCompletion<SelfStart_detail::SelfStartEnd, PatchTag>,
    ::Actions::MutateApply<AdvanceTime<>>,
    SelfStart::Actions::CheckForOrderIncrease, StepActions,
    ::Actions::Goto<SelfStart_detail::SelfStartBegin>,
    ::Actions::Label<SelfStart_detail::SelfStartEnd>,
    SelfStart::Actions::Cleanup, ::Actions::MutateApply<AdvanceTime<>>,
    Parallel::Actions::TerminatePhase>>;
}  // namespace evolution::patches::Actions
#endif  // SPECTRE_KOKKOS
