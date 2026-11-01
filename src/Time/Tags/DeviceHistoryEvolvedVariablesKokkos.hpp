// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"

#ifdef SPECTRE_KOKKOS
namespace Tags {
/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup TimeGroup
 * \brief Tag for the TimeStepper history of the variables `Tag` with the
 * data on the device.
 *
 * An element can have both this and `HistoryEvolvedVariables<Tag>`, of
 * which at most one holds data.  Actions that only use the times of the
 * histories (e.g., to decide whether the step size can change) handle
 * both, using `TimeSteppers::DeviceHistory::bookkeeping()`.
 *
 * \tparam Tag tag for the variables
 */
template <typename Tag>
struct DeviceHistoryEvolvedVariables : db::SimpleTag {
  using type = TimeSteppers::DeviceHistory<>;
};
}  // namespace Tags
#endif  // SPECTRE_KOKKOS
