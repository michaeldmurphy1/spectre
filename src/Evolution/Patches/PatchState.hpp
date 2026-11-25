// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
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
}  // namespace evolution::patches
