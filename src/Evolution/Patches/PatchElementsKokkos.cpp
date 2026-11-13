// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/PatchElementsKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
PatchElements::PatchElements(
    const Domain<3>& domain, const PatchTable<3>& patch_table,
    const size_t patch,
    const std::vector<std::array<size_t, 3>>& initial_refinement_levels)
    : patch_(patch),
      dg_mesh_(patch_table.dg_mesh(patch)),
      element_ids_(patch_table.element_ids(patch)) {
  const size_t number_of_elements = element_ids_.size();

  // Elements and their maps, with one grid to inertial map per block
  elements_.reserve(number_of_elements);
  element_maps_.reserve(number_of_elements);
  map_of_slot_.reserve(number_of_elements);
  std::vector<size_t> map_blocks{};
  for (const auto& element_id : element_ids_) {
    elements_.push_back(::domain::create_initial_element(
        element_id, domain.blocks(), initial_refinement_levels));
    const Block<3>& block = domain.blocks()[element_id.block_id()];
    element_maps_.emplace_back(element_id, block);
    const auto map_block = alg::find(map_blocks, block.id());
    map_of_slot_.push_back(
        static_cast<size_t>(std::distance(map_blocks.begin(), map_block)));
    if (map_block == map_blocks.end()) {
      map_blocks.push_back(block.id());
      grid_to_inertial_maps_.push_back(
          block.is_time_dependent()
              ? block.moving_mesh_grid_to_inertial_map().get_clone()
              : ::domain::make_coordinate_map_base<Frame::Grid,
                                                   Frame::Inertial>(
                    ::domain::CoordinateMaps::Identity<3>{}));
      mesh_is_moving_ =
          mesh_is_moving_ or not grid_to_inertial_maps_.back()->is_identity();
    }
  }

  // The corners of the elements of each map
  std::vector<size_t> number_of_map_elements(grid_to_inertial_maps_.size(), 0);
  for (const size_t map : map_of_slot_) {
    ++number_of_map_elements[map];
  }
  for (const size_t number : number_of_map_elements) {
    map_corners_.emplace_back(8 * number);
  }
  std::ranges::fill(number_of_map_elements, 0);
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    const size_t map = map_of_slot_[slot];
    for (size_t corner = 0; corner < 8; ++corner) {
      tnsr::I<double, 3, Frame::ElementLogical> logical_corner{};
      for (size_t d = 0; d < 3; ++d) {
        logical_corner.get(d) = ((corner >> d) & 1) == 0 ? -1.0 : 1.0;
      }
      const auto grid_corner = element_maps_[slot](logical_corner);
      for (size_t d = 0; d < 3; ++d) {
        map_corners_[map].get(d)[8 * number_of_map_elements[map] + corner] =
            grid_corner.get(d);
      }
    }
    ++number_of_map_elements[map];
  }
}

const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
PatchElements::grid_to_inertial_map(const size_t slot) const {
  return *grid_to_inertial_maps_.at(map_of_slot_.at(slot));
}

const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
PatchElements::map(const size_t map) const {
  return *grid_to_inertial_maps_.at(map);
}

size_t PatchElements::map_of_slot(const size_t slot) const {
  return map_of_slot_.at(slot);
}

const tnsr::I<DataVector, 3, Frame::Grid>& PatchElements::map_corners(
    const size_t map) const {
  return map_corners_.at(map);
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
