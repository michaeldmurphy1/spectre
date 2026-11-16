// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Evolution/Patches/InboxTags.hpp"

namespace evolution::dg::subcell::fd_patch::Tags {
/// The inbox of a patch of FD elements for the RDMP data (temporal id 0) and
/// TCI decisions (temporal id 1) that DG neighbors send to its elements during
/// initialization, as `evolution::dg::subcell::Tags::InitialTciData` for an
/// element.
template <size_t Dim>
struct InitialTciDataInbox
    : evolution::patches::Tags::ElementsInbox<
          Dim, int, evolution::dg::subcell::InitialTciData> {};
}  // namespace evolution::dg::subcell::fd_patch::Tags
