// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PatchElementsKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/TagsKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/RdmpTciDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
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
 * \details A patch steps all its elements together with batched kernels. The
 * parts that don't depend on the system, i.e. the elements, their geometry
 * and the exchange of boundary data, are in its
 * `evolution::dg::subcell::fd_patch::PatchElements`. Each substep is:
 * 1. `prepare_send()`: compute the ghost variables, fill the ghost zones
 *    between the elements of the patch, and pack the boundary data for the
 *    neighbors outside the patch, in the layout of an FD element on the host.
 * 2. `PatchElements::receive()`: fill the ghost zones from the received
 *    boundary data, and keep the data that DG neighbors send for the packaged
 *    data.
 * 3. `take_step()`: compute the time derivative (`subcell::TimeDerivative` on
 *    the host), record it in the history and update the evolved variables
 *    (`RecordTimeStepperData`, `UpdateU`, `CleanHistory`), then fix the
 *    conserved variables, recover the primitive variables, fix them to the
 *    atmosphere and recompute the conserved variables
 *    (`FixConservativesAndComputePrims`, `FixToAtmosphere`,
 *    `UpdateConservatives`).
 *
 * The state of the elements, i.e. their evolved, primitive and spacetime
 * variables (`state_tags`), is in the DataBox of the patch, and the methods
 * take it as arguments: those that change it take pointers to all of it, the
 * others references. The patch holds the operations on the state, which only
 * depend on the options and the domain, and scratch data.
 *
 * The background spacetime is fixed, as for `EvolveValenciaDivClean` with a
 * time-independent grid to inertial map. Supported are monotonised central
 * reconstruction (of the temperature), the `Rusanov` and `Hll` boundary
 * corrections, FD derivative order 2, global time stepping, and affine grid to
 * inertial maps (see
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

  /// @{
  /// The state of the elements, in the DataBox of the patch
  using evolved_variables_tag =
      evolution::dg::subcell::fd_patch::Tags::Batched<System::variables_tag>;
  using primitive_variables_tag =
      evolution::dg::subcell::fd_patch::Tags::Batched<
          System::primitive_variables_tag>;
  using cell_spacetime_variables_tag =
      evolution::dg::subcell::fd_patch::Tags::Batched<
          ::Tags::Variables<subcell::cell_spacetime_tags>>;
  using face_spacetime_variables_tag =
      evolution::dg::subcell::fd_patch::Tags::BatchedOnFaces<
          ::Tags::Variables<subcell::face_spacetime_tags>>;
  using state_tags =
      tmpl::list<evolved_variables_tag, primitive_variables_tag,
                 cell_spacetime_variables_tag, face_spacetime_variables_tag>;
  using EvolvedVariables = typename evolved_variables_tag::type;
  using PrimitiveVariables = typename primitive_variables_tag::type;
  using CellSpacetimeVariables = typename cell_spacetime_variables_tag::type;
  using FaceSpacetimeVariables = typename face_spacetime_variables_tag::type;
  /// @}

  Patch() = default;
  Patch(const Patch&) = delete;
  Patch& operator=(const Patch&) = delete;
  Patch(Patch&&) = default;
  Patch& operator=(Patch&&) = default;
  ~Patch() = default;

  /*!
   * \brief Set up the operations of the patch `patch` of `fd_patches`.
   *
   * This sets up the `patch_elements()` (with the geometry at the `time`) and
   * the time derivative. The patch needs the spacetime on the faces (see
   * `set_face_spacetime_variables()`) before it takes a step.
   */
  Patch(const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
        const evolution::dg::subcell::fd_patch::FdPatches<3>& fd_patches,
        size_t patch,
        const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
        const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
            grid_to_inertial_map,
        double time, const domain::FunctionsOfTimeMap& functions_of_time,
        const fd::Reconstructor& reconstructor,
        const evolution::BoundaryCorrection& boundary_correction,
        const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
        const VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
        const FixConservatives& fix_conservatives,
        const PrimitiveFromConservativeOptions&
            primitive_from_conservative_options,
        double constraint_damping_parameter);

  /*!
   * \brief The initial state of the elements.
   *
   * This evaluates the `initial_data` at the cell centers and the background
   * spacetime on the faces at the `initial_time`, and computes the conserved
   * variables after fixing the primitive variables to the atmosphere, as
   * `EvolveValenciaDivClean` initializes subcell elements.
   */
  tuples::tagged_tuple_from_typelist<state_tags> initial_state(
      const Kokkos::DefaultExecutionSpace& exec,
      const InitialDataFunction& initial_data, double initial_time,
      const domain::FunctionsOfTimeMap& functions_of_time) const;

  /// Copy the spacetime on the faces to the face data of the time
  /// derivative. Needed before the first step and whenever the spacetime on
  /// the faces changes.
  void set_face_spacetime_variables(
      const Kokkos::DefaultExecutionSpace& exec,
      const FaceSpacetimeVariables& face_spacetime_variables);

  /// @{
  /// The elements of the patch, their geometry and the exchange of boundary
  /// data
  const evolution::dg::subcell::fd_patch::PatchElements& patch_elements()
      const {
    return patch_elements_;
  }
  evolution::dg::subcell::fd_patch::PatchElements& patch_elements() {
    return patch_elements_;
  }
  /// @}

  /// Host copies of the variables of an element, as the DataBox of an FD
  /// element holds them (see
  /// `evolution::dg::subcell::fd_patch::make_element_box`)
  struct ElementVariables {
    typename System::variables_tag::type evolved{};
    typename System::primitive_variables_tag::type primitives{};
    typename System::spacetime_variables_tag::type spacetime{};
  };

  /// The variables of each element on the subcell grid, copied from the device
  /// on `exec`, which is fenced.
  std::vector<ElementVariables> element_variables(
      const Kokkos::DefaultExecutionSpace& exec,
      const EvolvedVariables& evolved_variables,
      const PrimitiveVariables& primitives,
      const CellSpacetimeVariables& cell_spacetime_variables,
      const FaceSpacetimeVariables& face_spacetime_variables) const;

  /// The number of evolved values of each element, i.e., the number of
  /// cells times the number of independent components of the evolved
  /// variables
  size_t number_of_evolved_values_per_element() const;

  /// The time stepper history of the elements, empty and at the starting
  /// order of the `time_stepper`, as `Initialization::TimeStepperHistory`
  /// sets it up for an element
  TimeSteppers::DeviceHistory<> initial_history(
      const TimeStepper& time_stepper) const;

  /// Compute the ghost variables and fill the ghost zones between the
  /// elements of the patch, and return the boundary data for the neighbors
  /// outside the patch, valid until `next_time_step_id`. The
  /// `integration_order` is that of the time stepper history. The received
  /// boundary data goes to `PatchElements::receive()`.
  std::vector<evolution::dg::subcell::fd_patch::OutgoingBoundaryData<3>>
  prepare_send(const Kokkos::DefaultExecutionSpace& exec,
               const EvolvedVariables& evolved_variables,
               const PrimitiveVariables& primitives,
               const CellSpacetimeVariables& cell_spacetime_variables,
               const FaceSpacetimeVariables& face_spacetime_variables,
               const TimeStepId& next_time_step_id, size_t integration_order);

  /// The RDMP data that the elements send to their DG neighbors during
  /// initialization (see `initial_rdmp_data_for_dg_neighbors`)
  std::vector<evolution::dg::subcell::fd_patch::OutgoingInitialRdmpData<3>>
  initial_rdmp_data(
      const Kokkos::DefaultExecutionSpace& exec,
      const EvolvedVariables& evolved_variables,
      const PrimitiveVariables& primitives,
      const CellSpacetimeVariables& cell_spacetime_variables,
      const FaceSpacetimeVariables& face_spacetime_variables) const;

  /// @{
  /// Save the evolved and primitive variables, and restore them, e.g., at
  /// the end of each order of the self start, as the self start actions do
  /// for an element (see `SelfStart`). `discard_initial_values()` frees the
  /// saved values.
  void save_initial_values(
      const Kokkos::DefaultExecutionSpace& exec,
      const EvolvedVariables& evolved_variables,
      const PrimitiveVariables& primitives,
      const CellSpacetimeVariables& cell_spacetime_variables,
      const FaceSpacetimeVariables& face_spacetime_variables);
  void restore_initial_values(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<EvolvedVariables*> evolved_variables,
      gsl::not_null<PrimitiveVariables*> primitives,
      gsl::not_null<CellSpacetimeVariables*> cell_spacetime_variables,
      gsl::not_null<FaceSpacetimeVariables*> face_spacetime_variables) const;
  void discard_initial_values();
  /// @}

  /// Serializes nothing: the state of the elements is in the DataBox, and the
  /// patch only holds operations and scratch data, which are rebuilt after a
  /// restart from a checkpoint (see `Actions::RebuildPatch`).
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// Take the substep `time_step_id` (see the class documentation), with the
  /// time stepper `history` of the elements. The `functions_of_time` are those
  /// of the grid to inertial map.
  ///
  /// The kernels are only enqueued on `exec`, without waiting for them. Their
  /// failures, e.g. of the primitive recovery, are recorded on the device and
  /// reported by `check_last_step()`.
  void take_step(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<EvolvedVariables*> evolved_variables,
      gsl::not_null<PrimitiveVariables*> primitives,
      gsl::not_null<CellSpacetimeVariables*> cell_spacetime_variables,
      gsl::not_null<FaceSpacetimeVariables*> face_spacetime_variables,
      gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
      const TimeStepper& time_stepper, const TimeStepId& time_step_id,
      const TimeStepId& next_time_step_id, const TimeDelta& time_step,
      const domain::FunctionsOfTimeMap& functions_of_time);

  /*!
   * \brief ERRORs if the last `take_step()` failed, e.g. in the primitive
   * recovery, with the variables at a failed grid point. Waits for `exec`.
   *
   * \details `prepare_send()` calls this after it waits for the device, so
   * that the host waits for the device only once per substep. Call it also
   * before anything that needs the state of the last substep to be valid,
   * e.g. observations and checkpoints. Calling it again before the next
   * `take_step()` checks the same substep again.
   */
  void check_last_step(const Kokkos::DefaultExecutionSpace& exec) const;

 private:
  size_t number_of_cells_of_all_elements() const;
  void reset_statuses(const Kokkos::DefaultExecutionSpace& exec) const;
  void copy_statuses_to_host(const Kokkos::DefaultExecutionSpace& exec) const;

  void compute_conservatives(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<EvolvedVariables*> evolved_variables,
      const PrimitiveVariables& primitives,
      const CellSpacetimeVariables& cell_spacetime_variables) const;

  evolution::dg::subcell::fd_patch::PatchElements patch_elements_{};

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

  // Scratch data
  BatchedVariables<subcell::PrimitiveGhostVariables::ghost_variables_tag_list>
      ghost_variables_{};
  BatchedVariables<subcell::dt_tags_on_device> dt_variables_{};
  // The values saved during the self start
  evolution::dg::subcell::fd_patch::SavedBatches<EvolvedVariables,
                                                 PrimitiveVariables>
      initial_values_{};
  std::vector<std::array<std::optional<StepperErrorEstimate>, 2>>
      stepper_errors_{};
  // Buffers of the per-element decisions and the RDMP data, kept so they
  // aren't allocated every substep
  Kokkos::View<bool*> fix_conservatives_scratch_{};
  Kokkos::View<bool*> primitive_recovery_scratch_{};
  subcell::RdmpTciDataBuffers rdmp_tci_data_buffers_{};
  // The failures of the last `take_step()` (see `check_last_step()`). Their
  // device data changes in const member functions, like that of other views.
  DeferredPointwiseStatus<fd::face_atmosphere_snapshot_tags> face_status_{};
  DeferredPointwiseStatus<fix_conservatives_snapshot_tags>
      fix_conservatives_status_{};
  DeferredPointwiseStatus<primitive_recovery_snapshot_tags> recovery_status_{};
  DeferredPointwiseStatus<VariableFixing::fix_to_atmosphere_snapshot_tags>
      atmosphere_status_{};
};
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
