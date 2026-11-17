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
#include "Evolution/Patches/GridToInertial.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
namespace {
// The map of each element from the maps of the blocks
void gather_maps(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<::domain::CoordinateMaps::DeviceMap<3>*>& of_elements,
    const Kokkos::View<::domain::CoordinateMaps::DeviceMap<3>*>& table,
    const Kokkos::View<size_t*>& map_of_slot) {
  Kokkos::parallel_for(
      "patches::PatchElements gather maps",
      Kokkos::RangePolicy<>(exec, 0, of_elements.extent(0)),
      KOKKOS_LAMBDA(const size_t slot) {
        of_elements(slot) = table(map_of_slot(slot));
      });
}
}  // namespace

PatchElements::PatchElements(
    const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
    const PatchTable<3>& patch_table, const size_t patch,
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

  // The maps for device kernels
  device_maps_ =
      DeviceMaps{exec, grid_to_inertial_maps_.size(), number_of_elements};
  if (mesh_is_moving_) {
    observation_device_maps_ =
        DeviceMaps{exec, grid_to_inertial_maps_.size(), number_of_elements};
  }
  device_map_of_slot_ = Kokkos::View<size_t*>(
      Kokkos::view_alloc(exec, "Patch map of each slot"), number_of_elements);
  const auto host_map_of_slot = Kokkos::create_mirror_view(
      Kokkos::WithoutInitializing, device_map_of_slot_);
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    host_map_of_slot(slot) = map_of_slot_[slot];
  }
  Kokkos::deep_copy(exec, device_map_of_slot_, host_map_of_slot);
  // The host mirror must stay valid until the copy is done
  exec.fence("patches::PatchElements map of each slot");
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

PatchElements::DeviceMaps::DeviceMaps(const Kokkos::DefaultExecutionSpace& exec,
                                      const size_t number_of_maps,
                                      const size_t number_of_elements)
    : table(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                               "Patch device maps"),
            number_of_maps),
      host_table(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                                    "Patch device maps on the host"),
                 number_of_maps),
      of_elements(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                                     "Patch device maps of the elements"),
                  number_of_elements) {}

bool PatchElements::fill_device_maps(
    const Kokkos::DefaultExecutionSpace& exec, const DeviceMaps& device_maps,
    const double time,
    const ::domain::FunctionsOfTimeMap& functions_of_time) const {
  bool all_affine = true;
  for (size_t map = 0; map < grid_to_inertial_maps_.size(); ++map) {
    device_maps.host_table(map) =
        device_grid_to_inertial(*grid_to_inertial_maps_[map], time,
                                functions_of_time, map_corners_[map]);
    all_affine = all_affine and device_maps.host_table(map).is_affine();
  }
  Kokkos::deep_copy(exec, device_maps.table, device_maps.host_table);
  gather_maps(exec, device_maps.of_elements, device_maps.table,
              device_map_of_slot_);
  return all_affine;
}

bool PatchElements::update_device_maps(
    const Kokkos::DefaultExecutionSpace& exec, const double time,
    const ::domain::FunctionsOfTimeMap& functions_of_time) {
  return fill_device_maps(exec, device_maps_, time, functions_of_time);
}

Kokkos::View<const ::domain::CoordinateMaps::DeviceMap<3>*>
PatchElements::device_maps_at(
    const Kokkos::DefaultExecutionSpace& exec, const double time,
    const ::domain::FunctionsOfTimeMap& functions_of_time) const {
  ASSERT(mesh_is_moving_, "The maps of patch "
                              << patch_
                              << " don't depend on time, so use device_maps()");
  fill_device_maps(exec, observation_device_maps_, time, functions_of_time);
  return observation_device_maps_.of_elements;
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
