// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::fd_patch::Tags {
/*!
 * \brief An inbox of an FD patch for data sent to its elements, keyed by
 * `TemporalId`.
 *
 * \details The data is the same as an element receives, with the
 * `DirectionalId` of the sender (the direction from the receiver to the sender
 * in the receiver's frame). Since a patch receives data for all its elements,
 * the data also holds the `ElementId` of the receiving element.
 *
 * Every message triggers the patch's algorithm, which checks whether all data
 * has arrived.
 */
template <size_t Dim, typename TemporalId, typename Data>
struct ElementsInbox {
  using temporal_id = TemporalId;
  using stored_type = Data;
  /// The receiving element, the sender and the data
  using message_type =
      std::tuple<ElementId<Dim>, DirectionalId<Dim>, stored_type>;
  /// The messages of a temporal id. A `std::vector`, so that Charm++
  /// registers `message_type` as the type of the data that is sent.
  using messages_type = std::vector<message_type>;
  using type = std::map<temporal_id, messages_type>;

  static bool insert_into_inbox(const gsl::not_null<type*> inbox,
                                const temporal_id& temporal_id_of_data,
                                message_type&& data) {
    (*inbox)[temporal_id_of_data].push_back(std::move(data));
    return true;
  }

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');
    ss << pad << "FdPatchElementsInbox:\n";
    for (const auto& [temporal_id_of_data, messages] : inbox) {
      ss << pad << " Temporal id: " << temporal_id_of_data << "\n";
      for (const auto& [receiver, sender, data] : messages) {
        ss << pad << "  Receiver: " << receiver << ", sender: " << sender
           << "\n";
      }
    }
    return ss.str();
  }
};

/// The inbox of an FD patch for the boundary data that DG elements and other
/// patches send to its elements, as
/// `evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox` for an element.
/// See `received_all_boundary_data`.
template <size_t Dim>
struct BoundaryDataInbox
    : ElementsInbox<Dim, TimeStepId, evolution::dg::BoundaryData<Dim>> {};

/// The inbox of an FD patch for the RDMP data (temporal id 0) and TCI
/// decisions (temporal id 1) that DG neighbors send to its elements during
/// initialization, as `evolution::dg::subcell::Tags::InitialTciData` for an
/// element.
template <size_t Dim>
struct InitialTciDataInbox
    : ElementsInbox<Dim, int, evolution::dg::subcell::InitialTciData> {};
}  // namespace evolution::dg::subcell::fd_patch::Tags
