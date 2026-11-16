// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Evolution/DgSubcell/Tags/InitialTciData.hpp"
#include "Evolution/Patches/ActionsKokkos.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// Actions of patches of FD elements, in addition to those of
/// `evolution::patches::Actions`
namespace evolution::dg::subcell::fd_patch::Actions {
/*!
 * \brief Send the initial RDMP data (temporal id 0) and TCI decisions
 * (temporal id 1) of the patch's elements to their DG neighbors, as
 * `SetAndCommunicateInitialRdmpData` and `ComputeAndSendTciOnInitialGrid` do
 * for an element on the host with `AlwaysUseSubcells`.
 *
 * \details The elements of a patch never switch to DG, so they don't run the
 * initial TCI and send it their neighbors only for the DG elements' handshake.
 */
template <typename PatchTag, typename DgElementArray>
struct SendInitialTciData {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int /*array_index*/,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    constexpr size_t dim = Metavariables::volume_dim;
    using inbox_tag = evolution::dg::subcell::Tags::InitialTciData<dim>;
    auto& dg_proxy = Parallel::get_parallel_component<DgElementArray>(cache);
    const auto& patch = db::get<PatchTag>(box);
    for (auto& message :
         evolution::patches::Actions::detail::with_state<PatchTag>(
             box, [&patch](const auto&... state) {
               return patch.initial_rdmp_data(
                   Parallel::device_execution_space(), state...);
             })) {
      Parallel::receive_data<inbox_tag>(
          dg_proxy[message.receiver], 0,
          std::pair{message.sender, evolution::dg::subcell::InitialTciData{
                                        {}, std::move(message.rdmp_tci_data)}});
      Parallel::receive_data<inbox_tag>(
          dg_proxy[message.receiver], 1,
          std::pair{message.sender,
                    evolution::dg::subcell::InitialTciData{
                        std::decay_t<decltype(patch)>::tci_status_sent, {}}});
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Wait for the initial RDMP data and TCI decisions that the DG
 * neighbors send to the patch's elements during initialization, and discard
 * them, since the elements of a patch never switch to DG.
 */
template <size_t Dim, typename PatchTag>
struct ReceiveInitialTciData {
  using inbox_tags = tmpl::list<Tags::InitialTciDataInbox<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t number_of_dg_faces =
        db::get<PatchTag>(box).patch_elements().number_of_dg_faces();
    if (number_of_dg_faces == 0) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    auto& inbox = tuples::get<Tags::InitialTciDataInbox<Dim>>(inboxes);
    for (const int temporal_id : {0, 1}) {
      const auto messages = inbox.find(temporal_id);
      if (messages == inbox.end() or
          messages->second.size() < number_of_dg_faces) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
      if (messages->second.size() > number_of_dg_faces) {
        ERROR("The elements of the patch received "
              << messages->second.size() << " initial TCI messages with id "
              << temporal_id << " but have " << number_of_dg_faces
              << " DG neighbors");
      }
    }
    inbox.clear();
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace evolution::dg::subcell::fd_patch::Actions
#endif  // SPECTRE_KOKKOS
