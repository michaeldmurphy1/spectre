// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
/// The time stepper history of the elements of a patch, the only
/// `::Tags::DeviceHistoryEvolvedVariables` in its DataBox
template <typename DbTagsList>
using history_tag = tmpl::front<::Tags::get_all_device_history_tags<
    typename db::DataBox<DbTagsList>::tags_list>>;

/// The DataBox tags of the state of the elements of the patch in `PatchTag`
template <typename PatchTag>
using state_tags = typename PatchTag::type::state_tags;

/// Returns `function(state...)` with the state of the elements of the patch
template <typename PatchTag, typename DbTagsList, typename Function>
decltype(auto) with_state(const db::DataBox<DbTagsList>& box,
                          const Function& function) {
  return [&box, &function]<typename... StateTags>(
             tmpl::list<StateTags...> /*meta*/) -> decltype(auto) {
    return function(db::get<StateTags>(box)...);
  }(state_tags<PatchTag>{});
}

/// Returns `db::mutate<PatchTag, StateTags..., OtherTags...>(function, box)`,
/// i.e., `function` gets pointers to the patch, the state of its elements and
/// the `OtherTags`
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

/// The DataBox tags that `store_initial_patch` sets: the patch in
/// `PatchTag`, the state of its elements, their time stepper history in
/// `HistoryTag` and whether the step size is controlled by error estimates
template <typename PatchTag, typename HistoryTag>
using initial_patch_tags =
    tmpl::append<tmpl::list<PatchTag>, state_tags<PatchTag>,
                 tmpl::list<HistoryTag, ::Tags::StepperErrorEstimatesEnabled>>;

/*!
 * \brief Store a newly built `patch`, the initial `state` of its elements and
 * their time stepper `history` in the DataBox of the patch, as the actions
 * that initialize the patches of the systems do.
 *
 * \details The step size of patches isn't controlled by error estimates, so
 * `::Tags::StepperErrorEstimatesEnabled` is `false`.
 */
template <typename PatchTag, typename HistoryTag, typename DbTagsList>
void store_initial_patch(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    typename PatchTag::type patch,
    tuples::tagged_tuple_from_typelist<state_tags<PatchTag>> state,
    typename HistoryTag::type history) {
  db::mutate<PatchTag, HistoryTag, ::Tags::StepperErrorEstimatesEnabled>(
      [&patch, &history](const auto patch_ptr, const auto history_ptr,
                         const auto error_estimates_enabled) {
        *patch_ptr = std::move(patch);
        *history_ptr = std::move(history);
        *error_estimates_enabled = false;
      },
      box);
  // One tag at a time: nvcc's front end mismatches a parameter pack in the
  // middle of the parameters of a single `db::mutate` lambda
  tmpl::for_each<state_tags<PatchTag>>(
      [&box, &state]<typename StateTag>(tmpl::type_<StateTag> /*meta*/) {
        db::mutate<StateTag>(
            [&state](const auto state_ptr) {
              *state_ptr = std::move(get<StateTag>(state));
            },
            box);
      });
}
}  // namespace evolution::patches
