// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"

namespace evolution::dg::subcell::fd_patch {
/// Whether the executable evolves FD elements in FD patches
/// (`Metavariables::use_fd_patches`)
template <typename Metavariables>
constexpr bool uses_fd_patches() {
  if constexpr (requires { Metavariables::use_fd_patches; }) {
    return Metavariables::use_fd_patches;
  } else {
    return false;
  }
}

/*!
 * \brief If the `receiver` is an element of an FD patch, send the `data` from
 * the `sender` to the patch's `PatchInboxTag` and return `true`. Otherwise
 * return `false` without using the `data`, so the caller sends it to the
 * element.
 *
 * \details This is how DG elements send to FD neighbors in executables with FD
 * patches (`Metavariables::use_fd_patches`, which also provide
 * `fd_patches_tag` and `fd_patch_component`). In other executables it returns
 * `false`.
 */
template <typename PatchInboxTag, typename Metavariables, size_t Dim,
          typename TemporalId, typename Data>
bool send_to_patch(Parallel::GlobalCache<Metavariables>& cache,
                   const ElementId<Dim>& receiver,
                   const TemporalId& temporal_id,
                   const DirectionalId<Dim>& sender, Data&& data) {
  if constexpr (uses_fd_patches<Metavariables>()) {
    const auto location =
        Parallel::get<typename Metavariables::fd_patches_tag>(cache).location(
            receiver);
    if (not location.has_value()) {
      return false;
    }
    Parallel::receive_data<PatchInboxTag>(
        Parallel::get_parallel_component<
            typename Metavariables::fd_patch_component>(
            cache)[static_cast<int>(location->patch)],
        temporal_id,
        typename PatchInboxTag::message_type{receiver, sender,
                                             std::forward<Data>(data)});
    return true;
  } else {
    (void)cache;
    (void)receiver;
    (void)temporal_id;
    (void)sender;
    (void)data;
    return false;
  }
}
}  // namespace evolution::dg::subcell::fd_patch
