// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches::Actions {
namespace RunOnZerothElement_detail {
template <typename Action>
struct const_global_cache_tags_of {
  using type = tmpl::list<>;
};
template <typename Action>
  requires requires { typename Action::const_global_cache_tags; }
struct const_global_cache_tags_of<Action> {
  using type = typename Action::const_global_cache_tags;
};
}  // namespace RunOnZerothElement_detail

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
      typename RunOnZerothElement_detail::const_global_cache_tags_of<
          Action>::type;

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
}  // namespace evolution::patches::Actions
