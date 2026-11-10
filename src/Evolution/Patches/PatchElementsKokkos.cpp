// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/PatchElementsKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <array>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
PatchElements::PatchElements(
    const Domain<3>& domain, const PatchTable<3>& patch_table,
    const size_t patch,
    const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
    const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map)
    : patch_(patch),
      dg_mesh_(patch_table.dg_mesh(patch)),
      element_ids_(patch_table.element_ids(patch)),
      grid_to_inertial_map_(grid_to_inertial_map.get_clone()) {
  const size_t number_of_elements = element_ids_.size();

  // Elements and their maps
  elements_.reserve(number_of_elements);
  element_maps_.reserve(number_of_elements);
  for (const auto& element_id : element_ids_) {
    elements_.push_back(::domain::create_initial_element(
        element_id, domain.blocks(), initial_refinement_levels));
    element_maps_.emplace_back(element_id,
                               domain.blocks()[element_id.block_id()]);
  }
}

std::string PatchElements::element_description(const size_t slot) const {
  std::ostringstream description{};
  description << "element " << element_ids_.at(slot) << " of patch " << patch_;
  return description.str();
}

TimeSteppers::DeviceHistory<> PatchElements::initial_history(
    const TimeStepper& time_stepper, const size_t number_of_values) const {
  // The order that needs no past steps, as
  // `Initialization::TimeStepperHistory`
  const size_t starting_order =
      visit(
          []<typename Tag>(
              const std::pair<tmpl::type_<Tag>, typename Tag::type&&> order) {
            if constexpr (std::is_same_v<Tag, TimeSteppers::Tags::FixedOrder>) {
              return order.second;
            } else {
              return order.second.minimum;
            }
          },
          time_stepper.order()) -
      time_stepper.number_of_past_steps();
  return TimeSteppers::DeviceHistory<>{starting_order, number_of_values,
                                       number_of_elements()};
}
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
