// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace evolution::dg::subcell::fd_patch {
namespace Tags {
/// The slab size requests that the `CombineSlabSizes` action of the first
/// patch has received for each slab, and from how many parallel components
struct PendingSlabSizeRequests : db::SimpleTag {
  using type = std::map<int64_t, std::pair<size_t, TimeStepRequestProcessor>>;
};
}  // namespace Tags

namespace Actions {
/*!
 * \brief Combine the slab size requests that the `DgElementArray` and the
 * `FdPatchArray` reduced over their elements for `Events::ChangeSlabSize`,
 * and send the result to both, as the reduction over a single component
 * does.
 *
 * \details Runs on the first patch. The `DgElementArray` contributes only if
 * it has elements, since an empty array can't reduce.
 */
template <typename PatchComponent, typename DgComponent>
struct CombineSlabSizes {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const int64_t slab,
                    const TimeStepRequestProcessor& requests) {
    const bool dg_contributes =
        Parallel::get<Tags::FdPatches<Metavariables::volume_dim>>(cache)
            .number_of_dg_elements() > 0;
    const size_t expected = dg_contributes ? 2 : 1;
    std::optional<TimeStepRequestProcessor> combined{};
    db::mutate<Tags::PendingSlabSizeRequests>(
        [&](const gsl::not_null<
            std::map<int64_t, std::pair<size_t, TimeStepRequestProcessor>>*>
                pending) {
          auto [entry, inserted] =
              pending->try_emplace(slab, std::make_pair(0_st, requests));
          if (not inserted) {
            entry->second.second += requests;
          }
          ++entry->second.first;
          if (entry->second.first == expected) {
            combined = std::move(entry->second.second);
            pending->erase(entry);
          }
        },
        make_not_null(&box));
    if (not combined.has_value()) {
      return;
    }
    Parallel::simple_action<::Events::ChangeSlabSize_detail::StoreNewSlabSize>(
        Parallel::get_parallel_component<PatchComponent>(cache), slab,
        combined.value());
    if (dg_contributes) {
      Parallel::simple_action<
          ::Events::ChangeSlabSize_detail::StoreNewSlabSize>(
          Parallel::get_parallel_component<DgComponent>(cache), slab,
          combined.value());
    }
  }
};
}  // namespace Actions

/*!
 * \brief The `Metavariables::slab_size_reduction` of executables with FD
 * patches (see `Events::ChangeSlabSize`).
 *
 * \details The elements of the `DgElementArray` and the patches of the
 * `FdPatchArray` each reduce their slab size requests to the first patch,
 * which combines them (see `Actions::CombineSlabSizes`). Without patches, the
 * elements reduce as if this weren't set.
 */
template <typename PatchComponent, typename DgComponent>
struct SlabSizeReduction {
  template <typename ParallelComponent, typename Metavariables,
            typename SenderProxy>
  static void contribute(
      Parallel::GlobalCache<Metavariables>& cache,
      const SenderProxy& sender_proxy,
      ::Events::ChangeSlabSize::ReductionData reduction_data) {
    if (Parallel::get<Tags::FdPatches<Metavariables::volume_dim>>(cache)
            .number_of_patches() == 0) {
      Parallel::contribute_to_reduction<
          ::Events::ChangeSlabSize_detail::StoreNewSlabSize>(
          std::move(reduction_data), sender_proxy,
          Parallel::get_parallel_component<ParallelComponent>(cache));
    } else {
      Parallel::contribute_to_reduction<
          Actions::CombineSlabSizes<PatchComponent, DgComponent>>(
          std::move(reduction_data), sender_proxy,
          Parallel::get_parallel_component<PatchComponent>(cache)[0]);
    }
  }
};
}  // namespace evolution::dg::subcell::fd_patch
