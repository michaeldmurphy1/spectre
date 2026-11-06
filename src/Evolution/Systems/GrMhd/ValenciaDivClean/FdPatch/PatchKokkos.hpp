// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Domain;
namespace PUP {
class er;
}  // namespace PUP
class TimeDelta;
class TimeStepId;
class TimeStepper;
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
namespace evolution {
class BoundaryCorrection;
}  // namespace evolution
namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
class FdPatches;
}  // namespace evolution::dg::subcell::fd_patch
namespace grmhd::ValenciaDivClean::fd {
class Reconstructor;
}  // namespace grmhd::ValenciaDivClean::fd
/// \endcond

/// The ValenciaDivClean system on FD patches (see
/// `evolution::dg::subcell::fd_patch`)
namespace grmhd::ValenciaDivClean::fd_patch {
/// The variables that the initial data provides at a point: the primitive
/// variables and the spacetime variables
using initial_data_tags = tmpl::append<hydro::grmhd_tags<DataVector>,
                                       gr::tags_for_hydro<3, DataVector>>;

/// Evaluates the initial data at the inertial coordinates
using InitialDataFunction = std::function<Variables<initial_data_tags>(
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords)>;

/*!
 * \brief The ValenciaDivClean evolution of the FD elements of one patch, with
 * all their data on the device.
 *
 * \details A patch steps all its elements together with batched kernels. Each
 * substep is:
 * 1. `prepare_send()`: compute the ghost variables, fill the ghost zones
 *    between the elements of the patch, and pack the boundary data for the
 *    neighbors outside the patch, in the layout of an FD element on the host.
 * 2. `receive()`: fill the ghost zones from the received boundary data, and
 *    keep the data that DG neighbors send for the packaged data.
 * 3. `take_step()`: compute the time derivative (`subcell::TimeDerivative` on
 *    the host), record it in the history and update the evolved variables
 *    (`RecordTimeStepperData`, `UpdateU`, `CleanHistory`), then fix the
 *    conserved variables, recover the primitive variables, fix them to the
 *    atmosphere and recompute the conserved variables
 *    (`FixConservativesAndComputePrims`, `FixToAtmosphere`,
 *    `UpdateConservatives`).
 *
 * The background spacetime is fixed, as for `EvolveValenciaDivClean` with a
 * time-independent grid to inertial map. Supported are monotonised central
 * reconstruction (of the temperature), the `Rusanov` and `Hll` boundary
 * corrections, FD derivative order 2, global time stepping with time steppers
 * that need no past steps, and affine grid to inertial maps (see
 * `evolution::dg::subcell::fd_patch::AffineGridToInertial`). Anything else is
 * an error.
 *
 * The elements of the patch never switch to DG, so no TCI runs. They send the
 * TCI status `tci_status_sent` to their neighbors, which is what FD elements
 * send on the host with `AlwaysUseSubcells` (the only option FD patches
 * support), since their TCI decision is reset to 0 at initialization and
 * never recomputed.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
class Patch {
 public:
  /// The TCI status that the elements of the patch send to their neighbors
  static constexpr int tci_status_sent = 0;
  /// The number of RDMP values that every element sends and receives
  static constexpr size_t number_of_rdmp_values = 8;

  Patch() = default;
  Patch(const Patch&) = delete;
  Patch& operator=(const Patch&) = delete;
  Patch(Patch&&) = default;
  Patch& operator=(Patch&&) = default;
  ~Patch() = default;

  /*!
   * \brief Set up the patch `patch` of `fd_patches` and its initial data.
   *
   * This builds the elements and the tables for the ghost zones and the DG
   * faces, computes the geometry, evaluates the initial data at the cell
   * centers and the background spacetime on the faces, and computes the
   * conserved variables after fixing the primitive variables to the
   * atmosphere, as `EvolveValenciaDivClean` initializes subcell elements.
   */
  Patch(const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
        const evolution::dg::subcell::fd_patch::FdPatches<3>& fd_patches,
        size_t patch,
        const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
        const InitialDataFunction& initial_data,
        const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
            grid_to_inertial_map,
        double initial_time,
        const domain::FunctionsOfTimeMap& functions_of_time,
        const fd::Reconstructor& reconstructor,
        const evolution::BoundaryCorrection& boundary_correction,
        const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
        const VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
        const FixConservatives& fix_conservatives,
        const PrimitiveFromConservativeOptions&
            primitive_from_conservative_options,
        double constraint_damping_parameter, const TimeStepper& time_stepper);

  size_t number_of_elements() const { return element_ids_.size(); }
  const std::vector<ElementId<3>>& element_ids() const { return element_ids_; }

  /// The evolved variables of the elements
  const BatchedVariables<subcell::evolved_tags_on_device>& evolved_variables()
      const {
    return evolved_variables_;
  }
  /// The primitive variables of the elements
  const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives() const {
    return primitives_;
  }

  /// The subcell mesh of the elements
  const Mesh<3>& subcell_mesh() const { return subcell_mesh_; }

  /*!
   * \brief The data of each element that can be observed, by the name of each
   * tensor component as `dg::Events::ObserveFields` names it (e.g.
   * `TildeS_x`): the evolved variables, the primitive variables, the
   * spacetime variables, and the `InertialCoordinates` at `time`.
   *
   * The data is copied from the device on `exec`, which is fenced.
   */
  std::vector<std::unordered_map<std::string, DataVector>> observation_data(
      const Kokkos::DefaultExecutionSpace& exec, double time,
      const domain::FunctionsOfTimeMap& functions_of_time) const;

  /// The maximum and the root mean square of each component of the evolved
  /// variables over the elements of the patch, as `Events::ObserveNorms`
  /// computes them with `Max` and `L2Norm`, as text. A diagnostic until
  /// patches support observations.
  std::string norms(const Kokkos::DefaultExecutionSpace& exec) const;

  /// Compute the ghost variables and fill the ghost zones between the
  /// elements of the patch, and return the boundary data for the neighbors
  /// outside the patch, valid until `next_time_step_id`.
  std::vector<evolution::dg::subcell::fd_patch::OutgoingBoundaryData<3>>
  prepare_send(const Kokkos::DefaultExecutionSpace& exec,
               const TimeStepId& next_time_step_id);

  /// The RDMP data that the elements send to their DG neighbors during
  /// initialization (see `initial_rdmp_data_for_dg_neighbors`)
  std::vector<evolution::dg::subcell::fd_patch::OutgoingInitialRdmpData<3>>
  initial_rdmp_data(const Kokkos::DefaultExecutionSpace& exec) const;

  /// The number of faces of the elements with a DG neighbor
  size_t number_of_dg_faces() const;

  /// Whether `messages` hold all the data the patch receives in a substep
  bool received_all(
      const evolution::dg::subcell::fd_patch::Tags::BoundaryDataInbox<
          3>::messages_type& messages) const;

  /// Fill the ghost zones from the received boundary data and keep the data
  /// from DG neighbors for the packaged data. Must be called after
  /// `prepare_send()` in each substep.
  void receive(const Kokkos::DefaultExecutionSpace& exec,
               const evolution::dg::subcell::fd_patch::Tags::BoundaryDataInbox<
                   3>::messages_type& messages);

  /// Patches can't be serialized yet, e.g. for checkpoints, since their data
  /// is on the device. This is an error unless sizing.
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// Take the substep `time_step_id` (see the class documentation). The
  /// `functions_of_time` are those of the grid to inertial map.
  void take_step(const Kokkos::DefaultExecutionSpace& exec,
                 const TimeStepper& time_stepper,
                 const TimeStepId& time_step_id,
                 const TimeStepId& next_time_step_id,
                 const TimeDelta& time_step,
                 const domain::FunctionsOfTimeMap& functions_of_time);

 private:
  // A description of the element in `slot` for error messages
  std::string element_description(size_t slot) const;

  void update_geometry(const Kokkos::DefaultExecutionSpace& exec, double time,
                       const domain::FunctionsOfTimeMap& functions_of_time);

  void compute_conservatives(const Kokkos::DefaultExecutionSpace& exec);

  size_t patch_ = 0;
  Mesh<3> dg_mesh_{};
  Mesh<3> subcell_mesh_{};
  std::vector<ElementId<3>> element_ids_{};
  std::vector<Element<3>> elements_{};

  // Configuration
  std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>
      equation_of_state_{};
  VariableFixing::FixToAtmosphere<3>::DeviceView fix_to_atmosphere_{};
  FixConservatives fix_conservatives_{};
  PrimitiveFromConservativeOptions primitive_from_conservative_options_{};
  double constraint_damping_parameter_ = 0.0;
  std::variant<BoundaryCorrections::Rusanov, BoundaryCorrections::Hll>
      boundary_correction_{};
  std::variant<subcell::TimeDerivativeOnDevice<BoundaryCorrections::Rusanov>,
               subcell::TimeDerivativeOnDevice<BoundaryCorrections::Hll>>
      time_derivative_{};

  // Communication
  evolution::dg::subcell::fd_patch::GhostZones<3> ghost_zones_{};
  evolution::dg::subcell::fd_patch::PackagedDataFromDg<3>
      packaged_data_from_dg_{};
  BatchedVariables<subcell::PrimitiveGhostVariables::ghost_variables_tag_list>
      ghost_variables_{};
  ::fd::device::VolumeView padded_ghost_variables_{};
  Kokkos::View<double*> received_ghost_data_{};
  Kokkos::View<double*> received_mortar_data_{};
  Kokkos::View<double*> send_data_{};

  // Geometry
  std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
      grid_to_inertial_map_{};
  // The grid coordinates of the cell centers of each element, for
  // observations
  std::vector<tnsr::I<DataVector, 3, Frame::Grid>> cell_grid_coordinates_{};
  evolution::dg::subcell::fd_patch::LogicalToGrid logical_to_grid_{};
  std::optional<evolution::dg::subcell::fd_patch::AffineGridToInertial>
      grid_to_inertial_{};
  BatchedVariables<evolution::dg::subcell::fd_patch::inertial_face_tags>
      face_geometry_{};
  BatchedVariables<evolution::dg::subcell::fd_patch::inertial_cell_tags>
      cell_geometry_{};

  // Variables
  BatchedVariables<subcell::cell_spacetime_tags> cell_spacetime_variables_{};
  BatchedVariables<subcell::evolved_tags_on_device> evolved_variables_{};
  BatchedVariables<hydro::grmhd_tags<DataVector>> primitives_{};
  BatchedVariables<subcell::dt_tags_on_device> dt_variables_{};
  TimeSteppers::DeviceHistory<> history_{};
  std::vector<std::array<std::optional<StepperErrorEstimate>, 2>>
      stepper_errors_{};
};
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
