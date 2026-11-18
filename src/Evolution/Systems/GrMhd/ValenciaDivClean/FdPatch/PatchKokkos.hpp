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
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/ElementBox.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatchElementsKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PatchBaseKokkos.hpp"
#include "Evolution/Patches/InboxTags.hpp"
#include "Evolution/Patches/SavedBatchesKokkos.hpp"
#include "Evolution/Patches/TagsKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/FixingAndRecoveryKokkos.hpp"
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
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
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

/// The variables of the DataBox of an element on the host that the DataBoxes
/// of the elements of patches for events hold (see
/// `evolution::dg::subcell::fd_patch::make_element_box`)
using element_variables_tags =
    tmpl::list<System::variables_tag, System::primitive_variables_tag,
               System::spacetime_variables_tag>;

/*!
 * \brief The ValenciaDivClean evolution of the FD elements of one patch, with
 * all their data on the device.
 *
 * \details A patch steps all its elements together with batched kernels. The
 * parts that don't depend on the system, i.e. the elements, their geometry
 * and the exchange of boundary data, are in its
 * `evolution::dg::subcell::fd_patch::FdPatchElements`. Each substep is:
 * 1. `prepare_send()`: compute the ghost variables, fill the ghost zones
 *    between the elements of the patch, and pack the boundary data for the
 *    neighbors outside the patch, in the layout of an FD element on the host.
 * 2. `FdPatchElements::receive()`: fill the ghost zones from the received
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
 * `evolution::patches::AffineGridToInertial`). Anything else is
 * an error.
 *
 * The fixing of the variables and the recovery of the primitive variables
 * are shared with other GRMHD systems (see `FixingAndRecovery`), and the
 * members that don't depend on the system with all FD patches (see
 * `evolution::dg::subcell::fd_patch::PatchBase`).
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
class Patch : public evolution::dg::subcell::fd_patch::PatchBase<
                  System, element_variables_tags> {
 public:
  using ElementVariables = typename evolution::dg::subcell::fd_patch::PatchBase<
      System, element_variables_tags>::ElementVariables;

  /// The number of RDMP values that every element sends and receives
  static constexpr size_t number_of_rdmp_values = 8;

  /// @{
  /// The state of the elements, in the DataBox of the patch
  using evolved_variables_tag =
      evolution::patches::Tags::Batched<System::variables_tag>;
  using primitive_variables_tag =
      evolution::patches::Tags::Batched<System::primitive_variables_tag>;
  using cell_spacetime_variables_tag = evolution::patches::Tags::Batched<
      ::Tags::Variables<subcell::cell_spacetime_tags>>;
  using face_spacetime_variables_tag = evolution::patches::Tags::BatchedOnFaces<
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
   * \brief Set up the operations of the patch `patch` of `patch_table`.
   *
   * This sets up the `patch_elements()` (with the geometry at the `time`) and
   * the time derivative. The patch needs the spacetime on the faces (see
   * `set_face_spacetime_variables()`) before it takes a step.
   */
  Patch(const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
        const evolution::patches::PatchTable<3>& patch_table, size_t patch,
        const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
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

  /*!
   * \brief The variables of each element on the subcell grid, copied from the
   * device on `exec`, which is fenced.
   *
   * \details The variables are copied into pinned host memory that the patch
   * allocates at the first call (see `evolution::patches::HostCopy`), and the
   * returned `Variables` refer to it without owning the data. So they are
   * valid until the next call (and while the patch exists), which is enough
   * for the element boxes of one observation.
   */
  std::vector<ElementVariables> element_variables(
      const Kokkos::DefaultExecutionSpace& exec,
      const EvolvedVariables& evolved_variables,
      const PrimitiveVariables& primitives,
      const CellSpacetimeVariables& cell_spacetime_variables,
      const FaceSpacetimeVariables& face_spacetime_variables);

  /// Compute the ghost variables and fill the ghost zones between the
  /// elements of the patch, and return the boundary data for the neighbors
  /// outside the patch, valid until `next_time_step_id`. The
  /// `integration_order` is that of the time stepper history. The received
  /// boundary data goes to `FdPatchElements::receive()`.
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
  FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes> fixing_{};

  // Configuration
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
  evolution::patches::SavedBatches<EvolvedVariables, PrimitiveVariables>
      initial_values_{};
  std::vector<std::array<std::optional<StepperErrorEstimate>, 2>>
      stepper_errors_{};
  // Buffers of the RDMP data, kept so they aren't allocated every substep
  subcell::RdmpTciDataBuffers rdmp_tci_data_buffers_{};
};
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
