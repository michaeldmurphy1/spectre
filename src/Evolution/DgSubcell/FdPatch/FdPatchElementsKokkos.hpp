// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/Patches/InboxTags.hpp"
#include "Evolution/Patches/PatchElementsKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Domain;
class TimeStepId;
class TimeStepper;
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
namespace evolution::patches {
template <size_t Dim>
class PatchTable;
}  // namespace evolution::patches
/// \endcond

namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief The elements of a patch of FD elements: the parts of FD patches that
 * don't depend on the evolved system, i.e. the elements and their maps (see
 * `evolution::patches::PatchElements`), their FD geometry, and the exchange
 * of ghost data and packaged data with the neighbors outside the patch.
 *
 * \details System-specific FD patches (e.g.
 * `grmhd::ValenciaDivClean::fd_patch::Patch`) hold one and add the operations
 * of their system. In each substep they
 * 1. compute the ghost variables of all elements (a `BatchedVariables`, see
 *    `GhostZones`) and pass them to `fill_ghost_zones()`, and, if
 *    `has_external_faces()`, to `enqueue_boundary_data()`, and after
 *    fencing get the messages from `boundary_data_to_send()`,
 * 2. pass the received messages to `receive()`, which fills the remaining
 *    ghost zones of `padded_ghost_variables()` and the
 *    `received_mortar_data()`, laid out as described by
 *    `packaged_data_from_dg()`,
 * 3. compute the time derivative with the geometry (see `update_geometry()`).
 */
class FdPatchElements : public evolution::patches::PatchElements {
 public:
  FdPatchElements() = default;
  FdPatchElements(const FdPatchElements&) = delete;
  FdPatchElements& operator=(const FdPatchElements&) = delete;
  FdPatchElements(FdPatchElements&&) = default;
  FdPatchElements& operator=(FdPatchElements&&) = default;
  ~FdPatchElements() = default;

  /*!
   * \brief Set up the elements of the patch `patch` of `patch_table`.
   *
   * This builds the elements and their maps (see
   * `evolution::patches::PatchElements`), checks that their cells line up with
   * those of their neighbors in other blocks, builds the tables for the ghost
   * zones (of width `ghost_width`, for `number_of_ghost_variables` variables)
   * and for the packaged data from DG neighbors (with
   * `number_of_packaged_variables` independent components), and computes the
   * geometry at the `time`.
   *
   * With `compute_cell_geometry` the patch also computes the
   * `cell_geometry()` and `cell_mesh_velocity()` (see
   * `TimeDependentGeometry`), as systems that evolve the spacetime need, and
   * supports grid to inertial maps that aren't affine. Without it the maps
   * must be affine.
   */
  FdPatchElements(
      const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
      const evolution::patches::PatchTable<3>& patch_table, size_t patch,
      const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
      double time, const ::domain::FunctionsOfTimeMap& functions_of_time,
      size_t ghost_width, size_t number_of_ghost_variables,
      size_t number_of_packaged_variables, bool compute_cell_geometry = false);

  /// The subcell mesh of the elements
  const Mesh<3>& subcell_mesh() const { return subcell_mesh_; }

  /// The time-independent geometry at the cell centers of the element in
  /// `slot` on the host (see `LogicalToGrid::host_cells`), referring to the
  /// memory of the patch without copying it. It must not be modified.
  Variables<host_logical_to_grid_cell_tags> host_cell_geometry(
      size_t slot) const;

  /*!
   * \brief Prepare the inertial geometry at the cell centers at `time` on the
   * host for `host_inertial_cell_geometry()`, e.g. for observations.
   *
   * \details If the mesh moves, the geometry is evaluated on the device with
   * the maps at `time` (see `device_maps_at()`) and copied to the host on
   * `exec`, which is fenced. Otherwise the inertial frame is the grid frame,
   * so there is nothing to do.
   */
  void prepare_host_inertial_cell_geometry(
      const Kokkos::DefaultExecutionSpace& exec, double time,
      const ::domain::FunctionsOfTimeMap& functions_of_time) const;

  /// The inertial coordinates, the logical to inertial inverse Jacobian and
  /// its determinant at the cell centers of the element in `slot` on the host,
  /// at the time of the last `prepare_host_inertial_cell_geometry()`,
  /// referring to the memory of the patch without copying it. It must not be
  /// modified.
  Variables<host_inertial_cell_tags> host_inertial_cell_geometry(
      size_t slot) const;

  /*!
   * \brief Update the geometry to the `time`. Only needed if the mesh moves.
   *
   * \details The host evaluates the grid to inertial maps (one per block)
   * and copies their parameters (`evolution::patches::AffineGridToInertial`
   * for affine maps, and the `device_maps()` for the cell geometry) to the
   * device, from where the kernels read them, so the arguments of the kernels
   * don't change between substeps. The copy from the host buffer is only
   * enqueued, so the copy of the previous call must be done before the next
   * call, e.g. because the patch waited for the device in between (which it
   * does once per substep).
   */
  void update_geometry(const Kokkos::DefaultExecutionSpace& exec, double time,
                       const ::domain::FunctionsOfTimeMap& functions_of_time);

  /// @{
  /// The geometry at the last `update_geometry()`
  const BatchedVariables<inertial_face_tags>& face_geometry() const {
    return face_geometry_;
  }
  const BatchedVariables<inertial_cell_tags>& cell_det_inverse_jacobian()
      const {
    return cell_geometry_;
  }
  /// The divergence of the mesh velocity of each element, for affine maps
  Kokkos::View<const double*> div_mesh_velocity() const {
    return div_mesh_velocity_;
  }
  /// The inertial coordinates, inverse Jacobian and its determinant at the
  /// cell centers. Only with `compute_cell_geometry`.
  const BatchedVariables<cell_geometry_tags>& cell_geometry() const {
    return cell_inertial_geometry_;
  }
  /// The mesh velocity and its divergence at the cell centers. Only with
  /// `compute_cell_geometry`.
  const BatchedVariables<cell_mesh_velocity_tags>& cell_mesh_velocity() const {
    return cell_mesh_velocity_;
  }
  /// @}

  /// Fill the interior and the ghost zones between the elements of the patch
  /// of `padded_ghost_variables()` from the unpadded `ghost_variables` of the
  /// elements (see `GhostZones::fill_from_patch()`)
  void fill_ghost_zones(const Kokkos::DefaultExecutionSpace& exec,
                        const Kokkos::View<const double*>& ghost_variables);

  /// Whether any element of the patch has a neighbor outside the patch
  bool has_external_faces() const {
    return not ghost_zones_.external_faces().empty();
  }

  /// Enqueue computing the boundary data for the neighbors outside the patch
  /// from the unpadded `ghost_variables` and copying it to the host. Read it
  /// with `boundary_data_to_send()` after fencing `exec`.
  void enqueue_boundary_data(
      const Kokkos::DefaultExecutionSpace& exec,
      const Kokkos::View<const double*>& ghost_variables);

  /// The boundary data for the neighbors outside the patch, from the last
  /// `enqueue_boundary_data()` and the RDMP data of the elements, valid until
  /// `next_time_step_id` (see `pack_boundary_data`). The execution space
  /// instance of `enqueue_boundary_data()` must have been fenced since.
  std::vector<OutgoingBoundaryData<3>> boundary_data_to_send(
      const std::vector<RdmpTciData>& rdmp_tci_data,
      const TimeStepId& next_time_step_id, int tci_status,
      size_t integration_order) const;

  /// The RDMP data that the elements send to their DG neighbors during
  /// initialization (see `initial_rdmp_data_for_dg_neighbors`)
  std::vector<OutgoingInitialRdmpData<3>> initial_rdmp_data(
      const std::vector<RdmpTciData>& rdmp_tci_data) const;

  /// The number of faces of the elements with a DG neighbor
  size_t number_of_dg_faces() const;

  /// Whether `messages` hold all the data the patch receives in a substep
  bool received_all(
      const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
          messages) const;

  /// Fill the ghost zones from the received boundary data, whose ghost data
  /// ends with `number_of_rdmp_values` RDMP values, and keep the data from DG
  /// neighbors in `received_mortar_data()`. Must be called after
  /// `fill_ghost_zones()` in each substep. The copies to the device are only
  /// enqueued; their host buffers are reused in the next `receive()`, so the
  /// boundary data of the next substep must be sent (which waits for the
  /// device) before then.
  void receive(
      const Kokkos::DefaultExecutionSpace& exec,
      const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
          messages,
      size_t number_of_rdmp_values);

  /// The ghost variables of the elements padded with their ghost zones
  const ::fd::device::VolumeView& padded_ghost_variables() const {
    return padded_ghost_variables_;
  }
  /// Where the packaged data from DG neighbors goes on the faces
  const PackagedDataFromDg<3>& packaged_data_from_dg() const {
    return packaged_data_from_dg_;
  }
  /// The packaged data received from DG neighbors in the last `receive()`
  Kokkos::View<const double*> received_mortar_data() const {
    return received_mortar_data_;
  }

 private:
  Mesh<3> subcell_mesh_{};

  // Geometry
  bool geometry_computed_ = false;
  LogicalToGrid logical_to_grid_{};
  // The parameters of the maps on the device, per face set (see
  // `inertial_geometry_on_device()`), and the divergence of the mesh velocity
  // per element, with their host buffers
  Kokkos::View<evolution::patches::AffineGridToInertial*> grid_to_inertial_{};
  Kokkos::View<double*> div_mesh_velocity_{};
  Kokkos::View<evolution::patches::AffineGridToInertial*,
               Kokkos::SharedHostPinnedSpace>
      host_grid_to_inertial_{};
  Kokkos::View<double*, Kokkos::SharedHostPinnedSpace>
      host_div_mesh_velocity_{};
  BatchedVariables<inertial_face_tags> face_geometry_{};
  BatchedVariables<inertial_cell_tags> cell_geometry_{};
  bool compute_cell_geometry_ = false;
  TimeDependentGeometry time_dependent_geometry_{};
  BatchedVariables<cell_geometry_tags> cell_inertial_geometry_{};
  BatchedVariables<cell_mesh_velocity_tags> cell_mesh_velocity_{};
  // The inertial geometry at the cell centers for the host, if the mesh moves
  BatchedVariables<cell_geometry_tags> observation_cells_{};
  BatchedVariables<host_inertial_cell_tags, Kokkos::SharedHostPinnedSpace>
      host_observation_cells_{};

  // Communication
  GhostZones<3> ghost_zones_{};
  PackagedDataFromDg<3> packaged_data_from_dg_{};
  ::fd::device::VolumeView padded_ghost_variables_{};
  Kokkos::View<double*> received_ghost_data_{};
  Kokkos::View<double*> received_mortar_data_{};
  Kokkos::View<double*> send_data_{};
  // Host copies of the buffers, kept so they aren't allocated every substep,
  // and pinned so that copies don't wait for the device
  using HostBuffer = Kokkos::View<double*, Kokkos::SharedHostPinnedSpace>;
  HostBuffer host_received_ghost_data_{};
  HostBuffer host_received_mortar_data_{};
  HostBuffer host_send_data_{};
};

}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
