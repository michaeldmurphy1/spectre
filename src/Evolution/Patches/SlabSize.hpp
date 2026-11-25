// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "Evolution/Patches/CombinedReduction.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
/*!
 * \brief The `Metavariables::slab_size_reduction` of executables with patches
 * (see `Events::ChangeSlabSize`).
 *
 * \details The elements of the `DgElementArray` and the patches of the
 * `PatchArray` reduce their slab size requests over all elements (see
 * `CombinedReduction`), and both store the new slab size, as the reduction
 * over a single component does. Without patches, the elements reduce as if
 * this weren't set.
 */
template <typename PatchComponent, typename DgComponent>
struct SlabSizeReduction {
  template <typename ParallelComponent, typename Metavariables,
            typename SenderProxy>
  static void contribute(
      Parallel::GlobalCache<Metavariables>& cache,
      const SenderProxy& sender_proxy,
      ::Events::ChangeSlabSize::ReductionData reduction_data) {
    CombinedReduction<PatchComponent, DgComponent>::template contribute<
        ::Events::ChangeSlabSize_detail::StoreNewSlabSize,
        tmpl::list<PatchComponent, DgComponent>>(
        cache, std::move(reduction_data), sender_proxy);
  }
};
}  // namespace evolution::patches
