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
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::fd_patch::Tags {
/*!
 * \brief The inbox of an FD patch for the boundary data sent to its elements
 * by DG elements and other patches.
 *
 * \details The data is the same as for
 * `evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox`: the
 * `evolution::dg::BoundaryData` and the `DirectionalId` of the sender, with
 * the direction from the receiver to the sender in the receiver's frame. Since
 * a patch receives data for all its elements, the data also holds the
 * `ElementId` of the receiving element.
 *
 * Every message triggers the patch's algorithm, which checks whether all data
 * for the current time step has arrived (see `received_all_boundary_data`).
 */
template <size_t Dim>
struct BoundaryDataInbox {
  using temporal_id = TimeStepId;
  using stored_type = evolution::dg::BoundaryData<Dim>;
  /// The receiving element, the sender and the data
  using message_type =
      std::tuple<ElementId<Dim>, DirectionalId<Dim>, stored_type>;
  /// The messages of a time step. A `std::vector`, so that Charm++ registers
  /// `message_type` as the type of the data that is sent.
  using messages_type = std::vector<message_type>;
  using type = std::map<temporal_id, messages_type>;

  static bool insert_into_inbox(const gsl::not_null<type*> inbox,
                                const temporal_id& time_step_id,
                                message_type&& data) {
    (*inbox)[time_step_id].push_back(std::move(data));
    return true;
  }

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');
    ss << pad << "FdPatchBoundaryDataInbox:\n";
    for (const auto& [time_step_id, messages] : inbox) {
      ss << pad << " Time: " << time_step_id << "\n";
      for (const auto& [receiver, sender, boundary_data] : messages) {
        ss << pad << "  Receiver: " << receiver << ", sender: " << sender
           << "\n";
      }
    }
    return ss.str();
  }
};
}  // namespace evolution::dg::subcell::fd_patch::Tags
