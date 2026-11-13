// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief The parallel component of the FD patches: one array member per patch
 * (see `FdPatches`), which steps all the FD elements of the patch on the
 * device.
 *
 * \details The patches are distributed round-robin over the nodes
 * (processes, i.e. GPUs), and over the processors of each node, skipping the
 * `procs_to_ignore` (see `patch_procs()`). With
 * `FdPatches::dedicated_procs()`, the `DgElementArray` avoids the processors
 * of the patches. The `DgElementArray` of the same
 * executable holds only the DG elements (see `FdPatches::location`).
 */
template <class Metavariables, class PhaseDepActionList>
struct FdPatchArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = int;

  using const_global_cache_tags = tmpl::list<Tags::FdPatches<volume_dim>>;

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
        Parallel::get_parallel_component<FdPatchArray>(local_cache);
    const size_t number_of_patches =
        Parallel::get<Tags::FdPatches<volume_dim>>(local_cache)
            .number_of_patches();
    const auto procs = Parallel::get<Tags::FdPatches<volume_dim>>(local_cache)
                           .procs(local_cache, procs_to_ignore);
    for (size_t patch = 0; patch < number_of_patches; ++patch) {
      patch_array(static_cast<int>(patch))
          .insert(global_cache, initialization_items, procs[patch]);
    }
    patch_array.doneInserting();
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const bool force = false) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<FdPatchArray>(local_cache)
        .start_phase(next_phase, force);
  }
};
}  // namespace evolution::dg::subcell::fd_patch
