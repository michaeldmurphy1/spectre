// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/FdPatchElementsKokkos.hpp"

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
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd_patch {
FdPatchElements::FdPatchElements(
    const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
    const evolution::patches::PatchTable<3>& patch_table, const size_t patch,
    const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
    const double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
    const size_t ghost_width, const size_t number_of_ghost_variables,
    const size_t number_of_packaged_variables, const bool compute_cell_geometry)
    : evolution::patches::PatchElements(exec, domain, patch_table, patch,
                                        initial_refinement_levels),
      subcell_mesh_(evolution::dg::subcell::fd::mesh(dg_mesh())),
      compute_cell_geometry_(compute_cell_geometry) {
  const size_t number_of_elements = this->number_of_elements();

  // Ghost zones and data from DG neighbors
  check_cells_coincide_across_blocks(domain, elements(), patch_table, patch,
                                     ghost_width);
  ghost_zones_ = GhostZones<3>{
      exec,
      elements(),
      std::vector<DirectionalIdMap<3, std::optional<intrp::Irregular<3>>>>(
          number_of_elements),
      patch_table,
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
                                                 elements(),
                                                 patch_table,
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
  logical_to_grid_ =
      logical_to_grid(exec, element_maps(), dg_mesh(), compute_cell_geometry);
  grid_to_inertial_ = Kokkos::View<evolution::patches::AffineGridToInertial*>(
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
  if (compute_cell_geometry_) {
    time_dependent_geometry_ =
        TimeDependentGeometry{exec, dg_mesh(), number_of_elements};
    cell_inertial_geometry_ = BatchedVariables<cell_geometry_tags>{
        exec, subcell_mesh_.number_of_grid_points(), number_of_elements};
    cell_mesh_velocity_ = BatchedVariables<cell_mesh_velocity_tags>{
        exec, subcell_mesh_.number_of_grid_points(), number_of_elements};
  }
  update_geometry(exec, time, functions_of_time);
}

Variables<host_logical_to_grid_cell_tags> FdPatchElements::host_cell_geometry(
    const size_t slot) const {
  // A copy of the handle to the same memory, which the result refers to
  auto host_cells = logical_to_grid_.host_cells;
  Variables<host_logical_to_grid_cell_tags> result{};
  set_data_ref_to_element(make_not_null(&result), host_cells, slot);
  return result;
}

void FdPatchElements::update_geometry(
    const Kokkos::DefaultExecutionSpace& exec, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  if (geometry_computed_ and not mesh_is_moving()) {
    return;
  }
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::FdPatchElements::update_geometry"};
  // The maps of the blocks for device kernels
  const bool all_affine = update_device_maps(exec, time, functions_of_time);
  if (all_affine) {
    std::vector<evolution::patches::AffineGridToInertial> maps(
        number_of_maps());
    for (size_t map = 0; map < maps.size(); ++map) {
      maps[map] = evolution::patches::affine_grid_to_inertial(
          this->map(map), time, functions_of_time, map_corners(map));
    }
    const size_t number_of_elements = this->number_of_elements();
    for (size_t slot = 0; slot < number_of_elements; ++slot) {
      const evolution::patches::AffineGridToInertial& map =
          maps[map_of_slot(slot)];
      for (size_t d = 0; d < 3; ++d) {
        host_grid_to_inertial_(d * number_of_elements + slot) = map;
      }
      host_div_mesh_velocity_(slot) = map.div_mesh_velocity;
    }
    Kokkos::deep_copy(exec, grid_to_inertial_, host_grid_to_inertial_);
    Kokkos::deep_copy(exec, div_mesh_velocity_, host_div_mesh_velocity_);
    inertial_geometry_on_device(exec, make_not_null(&face_geometry_),
                                make_not_null(&cell_geometry_),
                                logical_to_grid_, grid_to_inertial_);
  } else if (not compute_cell_geometry_) {
    ERROR("The grid to inertial map of an element of FD patch "
          << patch()
          << " isn't affine. Only patches that compute the cell geometry (e.g. "
             "of systems that evolve the spacetime) support such maps.");
  }
  if (compute_cell_geometry_) {
    time_dependent_geometry_.compute_cells(
        exec, make_not_null(&cell_inertial_geometry_),
        make_not_null(&cell_mesh_velocity_), logical_to_grid_, device_maps());
    if (not all_affine) {
      time_dependent_geometry_.compute_faces(exec,
                                             make_not_null(&face_geometry_));
      // The determinant at the cell centers, which the affine path computes
      // from the maps
      Kokkos::deep_copy(
          exec,
          get(get<::Tags::BatchView<Tags::CellDetInverseJacobian>>(
              cell_geometry_)),
          get(get<::Tags::BatchView<::domain::Tags::DetInvJacobian<
                  Frame::ElementLogical, Frame::Inertial>>>(
              cell_inertial_geometry_)));
    }
  }
  maps_are_affine_ = all_affine;
  geometry_computed_ = true;
}

void FdPatchElements::fill_ghost_zones(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<const double*>& ghost_variables) {
  ghost_zones_.fill_from_patch(exec, padded_ghost_variables_, ghost_variables);
}

void FdPatchElements::enqueue_boundary_data(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<const double*>& ghost_variables) {
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::FdPatchElements::enqueue_boundary_data"};
  ghost_zones_.slice_for_external(exec, send_data_, ghost_variables);
  Kokkos::deep_copy(exec, host_send_data_, send_data_);
}

std::vector<OutgoingBoundaryData<3>> FdPatchElements::boundary_data_to_send(
    const std::vector<RdmpTciData>& rdmp_tci_data,
    const TimeStepId& next_time_step_id, const int tci_status,
    const size_t integration_order) const {
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::FdPatchElements::boundary_data_to_send"};
  return pack_boundary_data(
      ghost_zones_,
      gsl::span<const double>{host_send_data_.data(), host_send_data_.size()},
      element_ids(), dg_mesh(), rdmp_tci_data, next_time_step_id, tci_status,
      integration_order);
}

std::vector<OutgoingInitialRdmpData<3>> FdPatchElements::initial_rdmp_data(
    const std::vector<RdmpTciData>& rdmp_tci_data) const {
  return initial_rdmp_data_for_dg_neighbors(ghost_zones_, element_ids(),
                                            rdmp_tci_data);
}

size_t FdPatchElements::number_of_dg_faces() const {
  return fd_patch::number_of_dg_faces(ghost_zones_);
}

bool FdPatchElements::received_all(
    const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
        messages) const {
  return received_all_boundary_data(ghost_zones_, element_ids(), messages);
}

void FdPatchElements::receive(
    const Kokkos::DefaultExecutionSpace& exec,
    const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
        messages,
    const size_t number_of_rdmp_values) {
  if (not has_external_faces()) {
    return;
  }
  const Kokkos::Profiling::ScopedRegion region{
      "fd_patch::FdPatchElements::receive"};
  const auto& host_ghost_data = host_received_ghost_data_;
  const auto& host_mortar_data = host_received_mortar_data_;
  unpack_boundary_data(
      gsl::span<double>{host_ghost_data.data(), host_ghost_data.size()},
      gsl::span<double>{host_mortar_data.data(), host_mortar_data.size()},
      ghost_zones_, packaged_data_from_dg_, dg_mesh(), element_ids(), messages,
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
