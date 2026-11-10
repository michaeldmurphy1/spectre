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
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
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
namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
class FdPatches;
}  // namespace evolution::dg::subcell::fd_patch
/// \endcond

namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief The parts of an FD patch that don't depend on the evolved system:
 * its elements, their geometry, and the exchange of ghost data and packaged
 * data with the neighbors outside the patch.
 *
 * \details System-specific patches (e.g.
 * `grmhd::ValenciaDivClean::fd_patch::Patch`) hold one and add the operations
 * of their system. In each substep they
 * 1. compute the ghost variables of all elements (a `BatchedVariables`, see
 *    `GhostZones`) and pass them to `fill_ghost_zones()`, and, if
 *    `has_external_faces()`, to `boundary_data_to_send()`,
 * 2. pass the received messages to `receive()`, which fills the remaining
 *    ghost zones of `padded_ghost_variables()` and the
 *    `received_mortar_data()`, laid out as described by
 *    `packaged_data_from_dg()`,
 * 3. compute the time derivative with the geometry (see `update_geometry()`).
 */
class PatchElements {
 public:
  PatchElements() = default;
  PatchElements(const PatchElements&) = delete;
  PatchElements& operator=(const PatchElements&) = delete;
  PatchElements(PatchElements&&) = default;
  PatchElements& operator=(PatchElements&&) = default;
  ~PatchElements() = default;

  /*!
   * \brief Set up the elements of the patch `patch` of `fd_patches`.
   *
   * This builds the elements and their maps, checks that their cells line up
   * with those of their neighbors in other blocks, builds the tables for the
   * ghost zones (of width `ghost_width`, for `number_of_ghost_variables`
   * variables) and for the packaged data from DG neighbors (with
   * `number_of_packaged_variables` independent components), and computes the
   * geometry at the `time`.
   */
  PatchElements(
      const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
      const FdPatches<3>& fd_patches, size_t patch,
      const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map,
      double time, const domain::FunctionsOfTimeMap& functions_of_time,
      size_t ghost_width, size_t number_of_ghost_variables,
      size_t number_of_packaged_variables);

  /// @{
  /// The elements of the patch, in slot order, and their maps
  size_t number_of_elements() const { return element_ids_.size(); }
  const std::vector<ElementId<3>>& element_ids() const { return element_ids_; }
  const std::vector<Element<3>>& elements() const { return elements_; }
  const std::vector<ElementMap<3, Frame::Grid>>& element_maps() const {
    return element_maps_;
  }
  /// @}

  /// The DG mesh of the elements
  const Mesh<3>& dg_mesh() const { return dg_mesh_; }
  /// The subcell mesh of the elements
  const Mesh<3>& subcell_mesh() const { return subcell_mesh_; }

  /// The map from the grid to the inertial frame of the block of the patch
  const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
  grid_to_inertial_map() const {
    return *grid_to_inertial_map_;
  }

  /// A description of the element in `slot` for error messages
  std::string element_description(size_t slot) const;

  /// The time stepper history of the elements for `number_of_values` evolved
  /// values per element, empty and at the starting order of the
  /// `time_stepper`, as `Initialization::TimeStepperHistory` sets it up for an
  /// element
  TimeSteppers::DeviceHistory<> initial_history(const TimeStepper& time_stepper,
                                                size_t number_of_values) const;

  /// Update the geometry to the `time`. Only needed if the grid to inertial
  /// map isn't the identity.
  void update_geometry(const Kokkos::DefaultExecutionSpace& exec, double time,
                       const domain::FunctionsOfTimeMap& functions_of_time);

  /// @{
  /// The geometry at the last `update_geometry()`
  const BatchedVariables<inertial_face_tags>& face_geometry() const {
    return face_geometry_;
  }
  const BatchedVariables<inertial_cell_tags>& cell_geometry() const {
    return cell_geometry_;
  }
  const AffineGridToInertial& grid_to_inertial() const {
    return grid_to_inertial_.value();
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

  /// The boundary data for the neighbors outside the patch from the unpadded
  /// `ghost_variables` and the RDMP data of the elements, valid until
  /// `next_time_step_id` (see `pack_boundary_data`). Waits for `exec`.
  std::vector<OutgoingBoundaryData<3>> boundary_data_to_send(
      const Kokkos::DefaultExecutionSpace& exec,
      const Kokkos::View<const double*>& ghost_variables,
      const std::vector<RdmpTciData>& rdmp_tci_data,
      const TimeStepId& next_time_step_id, int tci_status,
      size_t integration_order);

  /// The RDMP data that the elements send to their DG neighbors during
  /// initialization (see `initial_rdmp_data_for_dg_neighbors`)
  std::vector<OutgoingInitialRdmpData<3>> initial_rdmp_data(
      const std::vector<RdmpTciData>& rdmp_tci_data) const;

  /// The number of faces of the elements with a DG neighbor
  size_t number_of_dg_faces() const;

  /// Whether `messages` hold all the data the patch receives in a substep
  bool received_all(
      const Tags::BoundaryDataInbox<3>::messages_type& messages) const;

  /// Fill the ghost zones from the received boundary data, whose ghost data
  /// ends with `number_of_rdmp_values` RDMP values, and keep the data from DG
  /// neighbors in `received_mortar_data()`. Must be called after
  /// `fill_ghost_zones()` in each substep.
  void receive(const Kokkos::DefaultExecutionSpace& exec,
               const Tags::BoundaryDataInbox<3>::messages_type& messages,
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
  size_t patch_ = 0;
  Mesh<3> dg_mesh_{};
  Mesh<3> subcell_mesh_{};
  std::vector<ElementId<3>> element_ids_{};
  std::vector<Element<3>> elements_{};
  std::vector<ElementMap<3, Frame::Grid>> element_maps_{};

  // Geometry
  std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
      grid_to_inertial_map_{};
  LogicalToGrid logical_to_grid_{};
  std::optional<AffineGridToInertial> grid_to_inertial_{};
  BatchedVariables<inertial_face_tags> face_geometry_{};
  BatchedVariables<inertial_cell_tags> cell_geometry_{};

  // Communication
  GhostZones<3> ghost_zones_{};
  PackagedDataFromDg<3> packaged_data_from_dg_{};
  ::fd::device::VolumeView padded_ghost_variables_{};
  Kokkos::View<double*> received_ghost_data_{};
  Kokkos::View<double*> received_mortar_data_{};
  Kokkos::View<double*> send_data_{};
  // Host copies of the buffers, kept so they aren't allocated every substep
  Kokkos::View<double*>::host_mirror_type host_received_ghost_data_{};
  Kokkos::View<double*>::host_mirror_type host_received_mortar_data_{};
  Kokkos::View<double*>::host_mirror_type host_send_data_{};
};

/*!
 * \brief Copies of batches saved on the device, e.g. the variables of the
 * elements of a patch at the start of the self start.
 *
 * \details Copying a `BatchedVariables` doesn't copy its data (see
 * `BatchedVariables`), so this makes deep copies.
 */
template <typename... Batches>
class SavedBatches {
 public:
  /// Save copies of the `batches`
  void save(const Kokkos::DefaultExecutionSpace& exec,
            const Batches&... batches) {
    saved_ =
        std::tuple<Batches...>{Batches{exec, batches.number_of_grid_points(),
                                       batches.number_of_elements()}...};
    std::apply(
        [&](auto&... saved) {
          (deep_copy(exec, make_not_null(&saved), batches), ...);
        },
        saved_);
  }

  /// Copy the saved values into the `batches`
  void restore(const Kokkos::DefaultExecutionSpace& exec,
               const gsl::not_null<Batches*>... batches) const {
    std::apply(
        [&](const auto&... saved) { (deep_copy(exec, batches, saved), ...); },
        saved_);
  }

  /// Free the saved values
  void discard() { saved_ = {}; }

 private:
  std::tuple<Batches...> saved_{};
};
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
