// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/PatchElementsKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <array>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd_patch {
PatchElements::PatchElements(
    const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
    const FdPatches<3>& fd_patches, const size_t patch,
    const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const size_t ghost_width, const size_t number_of_ghost_variables,
    const size_t number_of_packaged_variables)
    : patch_(patch),
      dg_mesh_(fd_patches.dg_mesh(patch)),
      subcell_mesh_(evolution::dg::subcell::fd::mesh(dg_mesh_)),
      element_ids_(fd_patches.element_ids(patch)),
      grid_to_inertial_map_(grid_to_inertial_map.get_clone()) {
  const size_t number_of_elements = element_ids_.size();

  // Elements and their maps
  elements_.reserve(number_of_elements);
  element_maps_.reserve(number_of_elements);
  for (const auto& element_id : element_ids_) {
    elements_.push_back(domain::create_initial_element(
        element_id, domain.blocks(), initial_refinement_levels));
    element_maps_.emplace_back(element_id,
                               domain.blocks()[element_id.block_id()]);
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
  host_received_ghost_data_ = Kokkos::create_mirror_view(
      Kokkos::WithoutInitializing, received_ghost_data_);
  host_received_mortar_data_ = Kokkos::create_mirror_view(
      Kokkos::WithoutInitializing, received_mortar_data_);
  host_send_data_ =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, send_data_);

  // Geometry
  logical_to_grid_ = logical_to_grid(exec, element_maps_, dg_mesh_);
  Index<3> face_extents = subcell_mesh_.extents();
  ++face_extents[0];
  face_geometry_ = BatchedVariables<inertial_face_tags>{
      exec, face_extents.product(), 3 * number_of_elements};
  cell_geometry_ = BatchedVariables<inertial_cell_tags>{
      exec, subcell_mesh_.number_of_grid_points(), number_of_elements};
  update_geometry(exec, time, functions_of_time);
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
  if (grid_to_inertial_.has_value() and grid_to_inertial_map_->is_identity()) {
    return;
  }
  grid_to_inertial_ =
      affine_grid_to_inertial(*grid_to_inertial_map_, time, functions_of_time,
                              logical_to_grid_.corners);
  inertial_geometry_on_device(exec, make_not_null(&face_geometry_),
                              make_not_null(&cell_geometry_), logical_to_grid_,
                              grid_to_inertial_.value());
}

void PatchElements::fill_ghost_zones(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<const double*>& ghost_variables) {
  ghost_zones_.fill_from_patch(exec, padded_ghost_variables_, ghost_variables);
}

std::vector<OutgoingBoundaryData<3>> PatchElements::boundary_data_to_send(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<const double*>& ghost_variables,
    const std::vector<RdmpTciData>& rdmp_tci_data,
    const TimeStepId& next_time_step_id, const int tci_status,
    const size_t integration_order) {
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::PatchElements::boundary_data_to_send"};
  ghost_zones_.slice_for_external(exec, send_data_, ghost_variables);
  const auto& host_send_data = host_send_data_;
  Kokkos::deep_copy(exec, host_send_data, send_data_);
  exec.fence(
      "evolution::dg::subcell::fd_patch::PatchElements::boundary_data_to_send");
  return pack_boundary_data(
      ghost_zones_,
      gsl::span<const double>{host_send_data.data(), host_send_data.size()},
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
  // The host buffers mustn't change until the copies are done
  exec.fence("evolution::dg::subcell::fd_patch::PatchElements::receive");
}
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
