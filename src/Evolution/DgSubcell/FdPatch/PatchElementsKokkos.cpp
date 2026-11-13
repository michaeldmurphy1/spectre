// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/PatchElementsKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd_patch {
PatchElements::PatchElements(
    const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
    const FdPatches<3>& fd_patches, const size_t patch,
    const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const size_t ghost_width, const size_t number_of_ghost_variables,
    const size_t number_of_packaged_variables)
    : patch_(patch),
      dg_mesh_(fd_patches.dg_mesh(patch)),
      subcell_mesh_(evolution::dg::subcell::fd::mesh(dg_mesh_)),
      element_ids_(fd_patches.element_ids(patch)) {
  const size_t number_of_elements = element_ids_.size();

  // Elements and their maps, with one grid to inertial map per block
  elements_.reserve(number_of_elements);
  element_maps_.reserve(number_of_elements);
  map_of_slot_.reserve(number_of_elements);
  std::vector<size_t> map_blocks{};
  for (const auto& element_id : element_ids_) {
    elements_.push_back(domain::create_initial_element(
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
              : domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                    domain::CoordinateMaps::Identity<3>{}));
      mesh_is_moving_ =
          mesh_is_moving_ or not grid_to_inertial_maps_.back()->is_identity();
    }
  }

  // Ghost zones and data from DG neighbors
  check_cells_coincide_across_blocks(domain, elements_, fd_patches, patch,
                                     ghost_width);
  ghost_zones_ = GhostZones<3>{
      exec,
      elements_,
      std::vector<DirectionalIdMap<3, std::optional<intrp::Irregular<3>>>>(
          number_of_elements),
      fd_patches,
      patch,
      ghost_width,
      number_of_ghost_variables};
  padded_ghost_variables_ = ::fd::device::make_volume_view(
      exec, "FD patch ghost variables", number_of_elements,
      number_of_ghost_variables, subcell_mesh_.extents(), ghost_width);
  received_ghost_data_ = Kokkos::View<double*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch received ghost data"),
      ghost_zones_.external_data_size());
  send_data_ = Kokkos::View<double*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch send data"),
      ghost_zones_.send_data_size());
  packaged_data_from_dg_ = PackagedDataFromDg<3>{exec,
                                                 elements_,
                                                 fd_patches,
                                                 patch,
                                                 number_of_packaged_variables,
                                                 number_of_packaged_variables,
                                                 0};
  received_mortar_data_ = Kokkos::View<double*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch received mortar data"),
      packaged_data_from_dg_.received_data_size());
  // Pinned, so that copies between them and the device don't wait for the
  // device
  const auto pinned_buffer = [&exec](const std::string& label,
                                     const size_t size) {
    return HostBuffer{
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label), size};
  };
  host_received_ghost_data_ = pinned_buffer(
      "FD patch received ghost data on the host", received_ghost_data_.size());
  host_received_mortar_data_ =
      pinned_buffer("FD patch received mortar data on the host",
                    received_mortar_data_.size());
  host_send_data_ =
      pinned_buffer("FD patch send data on the host", send_data_.size());

  // Geometry
  logical_to_grid_ = logical_to_grid(exec, element_maps_, dg_mesh_);
  // The corners of the elements of each map, where it must be affine
  std::vector<size_t> number_of_map_elements(grid_to_inertial_maps_.size(), 0);
  for (const size_t map : map_of_slot_) {
    ++number_of_map_elements[map];
  }
  for (const size_t number : number_of_map_elements) {
    map_corners_.emplace_back(8 * number);
  }
  std::fill(number_of_map_elements.begin(), number_of_map_elements.end(), 0);
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    const size_t map = map_of_slot_[slot];
    for (size_t corner = 0; corner < 8; ++corner) {
      for (size_t d = 0; d < 3; ++d) {
        map_corners_[map].get(d)[8 * number_of_map_elements[map] + corner] =
            logical_to_grid_.corners.get(d)[8 * slot + corner];
      }
    }
    ++number_of_map_elements[map];
  }
  grid_to_inertial_ = Kokkos::View<AffineGridToInertial*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch grid to inertial maps"),
      3 * number_of_elements);
  div_mesh_velocity_ = Kokkos::View<double*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch divergence of the mesh velocity"),
      number_of_elements);
  host_grid_to_inertial_ = decltype(host_grid_to_inertial_)(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch grid to inertial maps on the host"),
      grid_to_inertial_.size());
  host_div_mesh_velocity_ = decltype(host_div_mesh_velocity_)(
      Kokkos::view_alloc(
          exec, Kokkos::WithoutInitializing,
          "FD patch divergence of the mesh velocity on the host"),
      div_mesh_velocity_.size());
  Index<3> face_extents = subcell_mesh_.extents();
  ++face_extents[0];
  face_geometry_ = BatchedVariables<inertial_face_tags>{
      exec, face_extents.product(), 3 * number_of_elements};
  cell_geometry_ = BatchedVariables<inertial_cell_tags>{
      exec, subcell_mesh_.number_of_grid_points(), number_of_elements};
  update_geometry(exec, time, functions_of_time);
}

const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
PatchElements::grid_to_inertial_map(const size_t slot) const {
  return *grid_to_inertial_maps_.at(map_of_slot_.at(slot));
}

std::string PatchElements::element_description(const size_t slot) const {
  std::ostringstream description{};
  description << "element " << element_ids_.at(slot) << " of FD patch "
              << patch_;
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

void PatchElements::update_geometry(
    const Kokkos::DefaultExecutionSpace& exec, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  if (geometry_computed_ and not mesh_is_moving_) {
    return;
  }
  std::vector<AffineGridToInertial> maps(grid_to_inertial_maps_.size());
  for (size_t map = 0; map < maps.size(); ++map) {
    maps[map] = affine_grid_to_inertial(*grid_to_inertial_maps_[map], time,
                                        functions_of_time, map_corners_[map]);
  }
  const size_t number_of_elements = element_ids_.size();
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    const AffineGridToInertial& map = maps[map_of_slot_[slot]];
    for (size_t d = 0; d < 3; ++d) {
      host_grid_to_inertial_(d * number_of_elements + slot) = map;
    }
    host_div_mesh_velocity_(slot) = map.div_mesh_velocity;
  }
  Kokkos::deep_copy(exec, grid_to_inertial_, host_grid_to_inertial_);
  Kokkos::deep_copy(exec, div_mesh_velocity_, host_div_mesh_velocity_);
  inertial_geometry_on_device(exec, make_not_null(&face_geometry_),
                              make_not_null(&cell_geometry_), logical_to_grid_,
                              grid_to_inertial_);
  geometry_computed_ = true;
}

void PatchElements::fill_ghost_zones(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<const double*>& ghost_variables) {
  ghost_zones_.fill_from_patch(exec, padded_ghost_variables_, ghost_variables);
}

void PatchElements::enqueue_boundary_data(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<const double*>& ghost_variables) {
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::PatchElements::enqueue_boundary_data"};
  ghost_zones_.slice_for_external(exec, send_data_, ghost_variables);
  Kokkos::deep_copy(exec, host_send_data_, send_data_);
}

std::vector<OutgoingBoundaryData<3>> PatchElements::boundary_data_to_send(
    const std::vector<RdmpTciData>& rdmp_tci_data,
    const TimeStepId& next_time_step_id, const int tci_status,
    const size_t integration_order) const {
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::PatchElements::boundary_data_to_send"};
  return pack_boundary_data(
      ghost_zones_,
      gsl::span<const double>{host_send_data_.data(), host_send_data_.size()},
      element_ids_, dg_mesh_, rdmp_tci_data, next_time_step_id, tci_status,
      integration_order);
}

std::vector<OutgoingInitialRdmpData<3>> PatchElements::initial_rdmp_data(
    const std::vector<RdmpTciData>& rdmp_tci_data) const {
  return initial_rdmp_data_for_dg_neighbors(ghost_zones_, element_ids_,
                                            rdmp_tci_data);
}

size_t PatchElements::number_of_dg_faces() const {
  return fd_patch::number_of_dg_faces(ghost_zones_);
}

bool PatchElements::received_all(
    const Tags::BoundaryDataInbox<3>::messages_type& messages) const {
  return received_all_boundary_data(ghost_zones_, element_ids_, messages);
}

void PatchElements::receive(
    const Kokkos::DefaultExecutionSpace& exec,
    const Tags::BoundaryDataInbox<3>::messages_type& messages,
    const size_t number_of_rdmp_values) {
  if (not has_external_faces()) {
    return;
  }
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::PatchElements::receive"};
  const auto& host_ghost_data = host_received_ghost_data_;
  const auto& host_mortar_data = host_received_mortar_data_;
  unpack_boundary_data(
      gsl::span<double>{host_ghost_data.data(), host_ghost_data.size()},
      gsl::span<double>{host_mortar_data.data(), host_mortar_data.size()},
      ghost_zones_, packaged_data_from_dg_, element_ids_, messages,
      number_of_rdmp_values);
  Kokkos::deep_copy(exec, received_ghost_data_, host_ghost_data);
  Kokkos::deep_copy(exec, received_mortar_data_, host_mortar_data);
  ghost_zones_.fill_from_external(exec, padded_ghost_variables_,
                                  received_ghost_data_);
  // The host buffers mustn't change until the copies are done. They change
  // only in the next `receive()`, and the boundary data of the next substep
  // is sent before that, which waits for `exec`.
}
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
