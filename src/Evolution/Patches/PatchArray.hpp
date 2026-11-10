// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
namespace Actions {
/*!
 * \brief Errors in the `Parallel::Phase::LoadBalancing` phase: patches can't
 * migrate, since their data is on the device of their node and they hold
 * scratch data that isn't serialized (see `PatchArray`).
 */
struct ErrorOnLoadBalancing {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  [[noreturn]] static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ERROR(
        "FD patches can't migrate, so they don't support the LoadBalancing "
        "phase. Remove it from the PhaseChangeAndTriggers.");
  }
};
}  // namespace Actions

/*!
 * \brief The parallel component of the patches: one array member per patch
 * (see `PatchTable`), which steps all the elements of the patch on the device.
 *
 * \details The patches are distributed round-robin over the processors,
 * skipping the `procs_to_ignore`. The `DgElementArray` of the same executable
 * holds only the elements that aren't in patches (see
 * `PatchTable::location`).
 *
 * The `Metavariables::patch_table_tag` holds the `PatchTable`. Patches can't
 * migrate, so the `Parallel::Phase::LoadBalancing` phase is an error (see
 * `Actions::ErrorOnLoadBalancing`).
 */
template <class Metavariables, class PhaseDepActionList>
struct PatchArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::push_back<
      PhaseDepActionList,
      Parallel::PhaseActions<Parallel::Phase::LoadBalancing,
                             tmpl::list<Actions::ErrorOnLoadBalancing>>>;
  using array_index = int;

  using const_global_cache_tags =
      tmpl::list<typename Metavariables::patch_table_tag>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  using array_allocation_tags = tmpl::list<>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_items,
      const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
      /*array_allocation_items*/,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& patch_array =
        Parallel::get_parallel_component<PatchArray>(local_cache);
    const size_t number_of_patches =
        Parallel::get<typename Metavariables::patch_table_tag>(local_cache)
            .number_of_patches();
    const auto number_of_procs = Parallel::number_of_procs<size_t>(local_cache);
    size_t which_proc = 0;
    for (size_t patch = 0; patch < number_of_patches; ++patch) {
      while (procs_to_ignore.contains(which_proc)) {
        which_proc = (which_proc + 1) % number_of_procs;
      }
      patch_array(static_cast<int>(patch))
          .insert(global_cache, initialization_items, which_proc);
      which_proc = (which_proc + 1) % number_of_procs;
    }
    patch_array.doneInserting();
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const bool force = false) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<PatchArray>(local_cache)
        .start_phase(next_phase, force);
  }
};
}  // namespace evolution::patches
