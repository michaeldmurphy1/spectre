// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"

namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief ERRORs if the options of the executable ask for something that FD
 * patches don't support, whatever the evolved system.
 *
 * \details FD patches need:
 * - `AlwaysUseSubcells`, since they assume that the elements that start on
 *   the subcell grid stay there and the others stay on the DG grid (see
 *   `patch_table()`);
 * - FD derivative order 2, since the device time derivatives compute only
 *   second-order FD derivatives of the fluxes (no high-order flux
 *   corrections);
 * - global time stepping, since with local time stepping the DG neighbors
 *   would send data at times the patches never reach;
 * - events only in `EventsAndTriggersAtSteps` and `EventsAndTriggersAtSlabs`
 *   (see `evolution::patches::Actions::RunEventsAndTriggers`), since the DG
 *   elements would wait forever for the patches in the others.
 *
 * The `box` is that of a patch, with the
 * `evolution::dg::subcell::Tags::SubcellOptions<3>`, the `::Tags::LtsMode`
 * and the `::Tags::EventsAndDenseTriggers`.
 */
template <typename DbTagsList, typename Metavariables>
void check_options(const db::DataBox<DbTagsList>& box,
                   const Parallel::GlobalCache<Metavariables>& cache) {
  const auto& subcell_options =
      db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box);
  if (not subcell_options.always_use_subcells()) {
    ERROR("FD patches need AlwaysUseSubcells.");
  }
  if (subcell_options.finite_difference_derivative_order() !=
      ::fd::DerivativeOrder::Two) {
    ERROR("FD patches only support FiniteDifferenceDerivativeOrder: 2, but got "
          << subcell_options.finite_difference_derivative_order());
  }
  if (db::get<::Tags::LtsMode>(box) != LtsMode::Off) {
    ERROR("FD patches don't support local time stepping, but LtsMode is "
          << db::get<::Tags::LtsMode>(box));
  }
  std::vector<std::string> unsupported_events{};
  const auto add_event = [&unsupported_events](const Event& event) {
    unsupported_events.push_back(pretty_type::get_runtime_type_name(event));
  };
  db::get<::Tags::EventsAndDenseTriggers>(box).for_each_event(add_event);
  Parallel::get<
      ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtCheckpoints>>(cache)
      .for_each_event(add_event);
  for (const auto& event : Parallel::get<::Tags::EventsRunAtCleanup>(cache)) {
    add_event(*event);
  }
  if (not unsupported_events.empty()) {
    ERROR(
        "FD patches only run events in EventsAndTriggersAtSteps and "
        "EventsAndTriggersAtSlabs, but there are events in "
        "EventsAndDenseTriggers, EventsAndTriggersAtCheckpoints or "
        "EventsRunAtCleanup: "
        << unsupported_events);
  }
}
}  // namespace evolution::dg::subcell::fd_patch
