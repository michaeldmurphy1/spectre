// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PatchBaseKokkos.hpp"
#include "Evolution/Patches/DeviceObservablesKokkos.hpp"
#include "Evolution/Patches/SavedBatchesKokkos.hpp"
#include "Evolution/Patches/TagsKokkos.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/ObservablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/FixingAndRecoveryKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/RdmpTciDataKokkos.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/LimitLorentzFactor.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/DeviceParametersKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
namespace PUP {
class er;
}  // namespace PUP
class TimeDelta;
class TimeStepper;
template <size_t Dim>
class Domain;
namespace evolution {
class BoundaryCorrection;
}  // namespace evolution
namespace evolution::patches {
template <size_t Dim>
class PatchTable;
}  // namespace evolution::patches
namespace grmhd::GhValenciaDivClean::fd {
template <typename System>
class Reconstructor;
}  // namespace grmhd::GhValenciaDivClean::fd
/// \endcond

/// The GhValenciaDivClean system on FD patches (see
/// `evolution::dg::subcell::fd_patch`)
namespace grmhd::GhValenciaDivClean::fd_patch {
/// The evolved system of the patches, without neutrinos
using System =
    grmhd::GhValenciaDivClean::System<RadiationTransport::NoNeutrinos::System>;

/// The compute tags of the DataBox of an element on the host that
/// `EvolveGhValenciaDivClean` adds
/// (`gh::Actions::InitializeGhAnd3Plus1Variables` and the square root of the
/// determinant of the spatial metric), so that the DataBoxes of the elements of
/// patches for events hold them too
using element_box_compute_tags = tmpl::push_back<
    typename gh::Actions::InitializeGhAnd3Plus1Variables<3>::compute_tags,
    gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3, Frame::Inertial>>;

/// The variables of the DataBox of an element on the host that the DataBoxes
/// of the elements of patches for events hold (see
/// `evolution::dg::subcell::fd_patch::make_element_box`): the evolved and
/// primitive variables and the time derivatives of the evolved variables
using element_variables_tags =
    tmpl::list<System::variables_tag, System::primitive_variables_tag,
               db::add_tag_prefix<::Tags::dt, System::variables_tag>>;

/// The variables of an element that the initial data gives on the host: the
/// GH variables and the primitive variables
using initial_data_tags =
    tmpl::append<subcell::gh_tags_on_device, hydro::grmhd_tags<DataVector>>;

/*!
 * \brief The GhValenciaDivClean evolution of the FD elements of one patch,
 * with all their data on the device.
 *
 * \details A patch steps all its elements together with batched kernels, as
 * `EvolveGhValenciaDivClean` steps its FD elements. The members that don't
 * depend on the system are in `evolution::dg::subcell::fd_patch::PatchBase`,
 * and the fixing of the GRMHD variables and the recovery of the primitive
 * variables are shared with ValenciaDivClean (`FixingAndRecovery`). Each
 * substep is:
 * 1. `prepare_send()`: compute the ghost variables (the primitive variables to
 *    reconstruct and the GH variables), fill the ghost zones between the
 *    elements of the patch, and pack the boundary data for the neighbors
 *    outside the patch, in the layout of an FD element on the host.
 * 2. `receive()`: fill the ghost zones from the received boundary data, and
 *    keep the packaged data of the GRMHD variables that DG neighbors send.
 * 3. `compute_time_derivative()`: compute the time derivative
 *    (`subcell::TimeDerivativeOnDevice`, as `subcell::TimeDerivative` on the
 *    host) and record it in the history.
 * 4. `update_variables()`: update the evolved variables, then fix the
 *    conserved variables, recover the primitive variables
 *    (`FixConservativesAndComputePrims`), fix them to the atmosphere, limit
 *    the Lorentz factor and recompute the conserved variables
 *    (`FixToAtmosphere`, `LimitLorentzFactor`, `UpdateConservatives`). The
 *    spatial metric of the fixing and the recovery is that of the updated
 *    spacetime metric, as the compute tags of an element on the host give it.
 *
 * Dense output, which needs the history of the substep but not its update,
 * goes between 3 and 4, as on the host.
 *
 * The host fixes the primitive variables again at the start of each step of
 * the evolution (not in the self start), which `fix_variables()` does.
 *
 * The state of the elements, i.e. their evolved and primitive variables
 * (`state_tags`), is in the DataBox of the patch, and the methods take it as
 * arguments: those that change it take pointers to all of it, the others
 * references. The patch holds the operations on the state, which only depend
 * on the options and the domain, and scratch data.
 *
 * Supported are the harmonic gauge, damping functions that are sums of
 * Gaussians (see `ConstraintDamping::DampingFunction::as_gaussian_sum`), the
 * `MonotonisedCentralPrim` and `PositivityPreservingAdaptiveOrderPrim`
 * reconstructions, products of a GH boundary correction with the `Rusanov` or
 * `Hll` correction of the GRMHD variables (FD uses only the latter), FD
 * derivative order 2 for the GRMHD fluxes, global time stepping, and grid to
 * inertial maps that are affine or have a device version (see
 * `evolution::dg::subcell::fd_patch::TimeDependentGeometry`). Anything else is
 * an error.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
class Patch : public evolution::dg::subcell::fd_patch::PatchBase<
                  System, element_variables_tags, element_box_compute_tags> {
 public:
  using ElementVariables = typename evolution::dg::subcell::fd_patch::PatchBase<
      System, element_variables_tags,
      element_box_compute_tags>::ElementVariables;

  /// The number of RDMP values that every element sends and receives, as for
  /// ValenciaDivClean
  static constexpr size_t number_of_rdmp_values = 8;

  /// @{
  /// The state of the elements, in the DataBox of the patch
  using evolved_variables_tag =
      evolution::patches::Tags::Batched<System::variables_tag>;
  using primitive_variables_tag =
      evolution::patches::Tags::Batched<System::primitive_variables_tag>;
  using state_tags = tmpl::list<evolved_variables_tag, primitive_variables_tag>;
  using EvolvedVariables = typename evolved_variables_tag::type;
  using PrimitiveVariables = typename primitive_variables_tag::type;
  /// @}
  static_assert(std::is_same_v<typename EvolvedVariables::tags_list,
                               subcell::evolved_tags_on_device>,
                "The GH variables come first in the evolved variables, in "
                "the order of the device time derivative.");

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
   * the time derivative. The `damping_functions` are those of \f$\gamma_0\f$,
   * \f$\gamma_1\f$ and \f$\gamma_2\f$. The `spacetime_dissipation` is that of
   * the Kreiss-Oliger filter of the GH variables, if they are filtered.
   */
  Patch(
      const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
      const evolution::patches::PatchTable<3>& patch_table, size_t patch,
      const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
      double time, const domain::FunctionsOfTimeMap& functions_of_time,
      const grmhd::GhValenciaDivClean::fd::Reconstructor<System>& reconstructor,
      const evolution::BoundaryCorrection& boundary_correction,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
      const VariableFixing::LimitLorentzFactor& limit_lorentz_factor,
      const ValenciaDivClean::FixConservatives& fix_conservatives,
      const ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options,
      double constraint_damping_parameter,
      const std::array<
          const ConstraintDamping::DampingFunction<3, Frame::Grid>*, 3>&
          damping_functions,
      std::optional<double> spacetime_dissipation);

  /*!
   * \brief The initial state of the elements from the `initial_data` of each
   * element on the host, in slot order.
   *
   * \details The initial data holds the GH and primitive variables at the
   * cell centers, as an element on the host has them after setting the
   * initial data and the constraints (e.g. `Actions::SetInitialData` and
   * `SetPiAndPhiFromConstraints`). The primitive variables are then fixed to
   * the atmosphere, their Lorentz factor is limited and the conserved
   * variables are computed, twice, as the host does before and after the
   * initial TCI.
   */
  tuples::tagged_tuple_from_typelist<state_tags> initial_state(
      const Kokkos::DefaultExecutionSpace& exec,
      const std::vector<Variables<initial_data_tags>>& initial_data);

  /*!
   * \brief The variables of each element on the subcell grid, copied from the
   * device on `exec`, which is fenced (see
   * `PatchBase::copy_element_variables`).
   *
   * \details The time derivatives are those of the last substep.
   */
  std::vector<ElementVariables> element_variables(
      const Kokkos::DefaultExecutionSpace& exec,
      const EvolvedVariables& evolved_variables,
      const PrimitiveVariables& primitives);

  /*!
   * \brief Compute the variables of the elements at `time` in the current
   * step by dense output (`TimeSteppers::dense_update_u_on_device`), then fix
   * the conserved variables and recover the primitive variables, as
   * `subcell::FixConservativesAndComputePrims` does after dense output on the
   * host. The primitive variables of the substep are the initial guess of the
   * recovery, as on the host.
   *
   * \details Enqueued on `exec` into scratch data of the patch, which is
   * allocated at the first dense output; the state isn't changed. Call it
   * between `compute_time_derivative()` and `update_variables()`, as the host
   * does dense output between recording the history and the update. Returns
   * false if dense output isn't possible yet in this substep (see
   * `TimeStepper::dense_update_u`). The results are read with
   * `dense_element_variables()`.
   */
  bool dense_output(const Kokkos::DefaultExecutionSpace& exec,
                    const EvolvedVariables& evolved_variables,
                    const PrimitiveVariables& primitives,
                    gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
                    const TimeStepper& time_stepper, double time);

  /// The variables of each element from the last `dense_output()` (at
  /// `time`), copied from the device on `exec`, which is fenced. ERRORs if
  /// the fixing or the recovery of the dense output failed. The time
  /// derivatives are those of the current substep, as on the host.
  std::vector<ElementVariables> dense_element_variables(
      const Kokkos::DefaultExecutionSpace& exec,
      const EvolvedVariables& evolved_variables,
      const PrimitiveVariables& primitives, double time);

  /// Whether `device_observables` can compute the observable tensor `name`
  /// (see `fd_patch::is_device_observable`)
  static bool is_device_observable(const std::string& name);

  /*!
   * \brief The observable tensors `tensor_names` of all elements, computed on
   * the device at `time` (see `fd_patch::device_observables`)
   *
   * \details From the state (`evolved_variables` and `primitives`) or, with
   * `dense_output`, from the last `dense_output()`, whose failures are reported
   * as `dense_element_variables()` does. The result refers to memory of the
   * patch that the next call overwrites.
   */
  const evolution::patches::DeviceObservables& device_observables(
      const Kokkos::DefaultExecutionSpace& exec,
      const std::vector<std::string>& tensor_names,
      const EvolvedVariables& evolved_variables,
      const PrimitiveVariables& primitives, bool dense_output, double time,
      const domain::FunctionsOfTimeMap& functions_of_time);

  /// Compute the ghost variables and fill the ghost zones between the
  /// elements of the patch, and return the boundary data for the neighbors
  /// outside the patch, valid until `next_time_step_id`. The
  /// `integration_order` is that of the time stepper history. The received
  /// boundary data goes to `receive()`.
  std::vector<evolution::dg::subcell::fd_patch::OutgoingBoundaryData<3>>
  prepare_send(const Kokkos::DefaultExecutionSpace& exec,
               const EvolvedVariables& evolved_variables,
               const PrimitiveVariables& primitives,
               const TimeStepId& next_time_step_id, size_t integration_order);

  /// The RDMP data that the elements send to their DG neighbors during
  /// initialization (see `initial_rdmp_data_for_dg_neighbors`)
  std::vector<evolution::dg::subcell::fd_patch::OutgoingInitialRdmpData<3>>
  initial_rdmp_data(const Kokkos::DefaultExecutionSpace& exec,
                    const EvolvedVariables& evolved_variables,
                    const PrimitiveVariables& primitives) const;

  /// @{
  /// Save the evolved and primitive variables, and restore them, e.g., at
  /// the end of each order of the self start, as the self start actions do
  /// for an element (see `SelfStart`). `discard_initial_values()` frees the
  /// saved values.
  void save_initial_values(const Kokkos::DefaultExecutionSpace& exec,
                           const EvolvedVariables& evolved_variables,
                           const PrimitiveVariables& primitives);
  void restore_initial_values(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<EvolvedVariables*> evolved_variables,
      gsl::not_null<PrimitiveVariables*> primitives) const;
  void discard_initial_values();
  /// @}

  /// Serializes nothing: the state of the elements is in the DataBox, and the
  /// patch only holds operations and scratch data, which are rebuilt after a
  /// restart from a checkpoint (see `Actions::RebuildPatch`).
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// @{
  /// Take the substep `time_step_id` (see the class documentation) in two
  /// parts: compute the time derivative and record it in the time stepper
  /// `history` of the elements, then update the variables. The
  /// `functions_of_time` are those of the grid to inertial maps and the
  /// damping functions.
  ///
  /// The kernels are only enqueued on `exec`, without waiting for them. Their
  /// failures, e.g. of the primitive recovery, are recorded on the device and
  /// reported by `check_last_step()`.
  void compute_time_derivative(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<EvolvedVariables*> evolved_variables,
      gsl::not_null<PrimitiveVariables*> primitives,
      gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
      const TimeStepId& time_step_id,
      const domain::FunctionsOfTimeMap& functions_of_time);
  void update_variables(const Kokkos::DefaultExecutionSpace& exec,
                        gsl::not_null<EvolvedVariables*> evolved_variables,
                        gsl::not_null<PrimitiveVariables*> primitives,
                        gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
                        const TimeStepper& time_stepper,
                        const TimeStepId& time_step_id,
                        const TimeStepId& next_time_step_id,
                        const TimeDelta& time_step);
  /// @}

  /// Fix the primitive variables to the atmosphere, limit their Lorentz
  /// factor and recompute the conserved variables, as the host does at the
  /// start of each step of the evolution. Enqueued on `exec`; failures are
  /// reported by `check_last_step()`.
  void fix_variables(const Kokkos::DefaultExecutionSpace& exec,
                     gsl::not_null<EvolvedVariables*> evolved_variables,
                     gsl::not_null<PrimitiveVariables*> primitives);

  /*!
   * \brief ERRORs if the last `update_variables()` or `fix_variables()`
   * failed, e.g. in the primitive recovery, with the variables at a failed
   * grid point.
   * Waits for `exec`.
   *
   * \details `prepare_send()` calls this after it waits for the device, so
   * that the host waits for the device only once per substep. Call it also
   * before anything that needs the state of the last substep to be valid,
   * e.g. observations and checkpoints.
   */
  void check_last_step(const Kokkos::DefaultExecutionSpace& exec) const;

 private:
  // Fix the primitive variables to the atmosphere, limit their Lorentz factor
  // and recompute the conserved variables, with the spatial metric of the
  // last `update_fixing_metric()`
  void fix_primitives_and_update_conservatives(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<EvolvedVariables*> evolved_variables,
      gsl::not_null<PrimitiveVariables*> primitives);
  // The spatial metric etc. of the fixing and the primitive recovery from the
  // spacetime metric of the `evolved_variables`
  void update_fixing_metric(const Kokkos::DefaultExecutionSpace& exec,
                            const EvolvedVariables& evolved_variables);
  size_t number_of_face_points() const;

  ValenciaDivClean::fd_patch::FixingAndRecovery<
      OrderedListOfPrimitiveRecoverySchemes>
      fixing_{};

  // Configuration
  VariableFixing::LimitLorentzFactor limit_lorentz_factor_{};
  double constraint_damping_parameter_ = 0.0;
  // Whether the reconstructed states on the faces are fixed to the atmosphere
  bool fix_reconstructed_states_ = false;
  std::array<
      std::unique_ptr<ConstraintDamping::DampingFunction<3, Frame::Grid>>, 3>
      damping_functions_{};
  std::variant<ValenciaDivClean::BoundaryCorrections::Rusanov,
               ValenciaDivClean::BoundaryCorrections::Hll>
      boundary_correction_{};
  std::variant<subcell::TimeDerivativeOnDevice<
                   ValenciaDivClean::BoundaryCorrections::Rusanov>,
               subcell::TimeDerivativeOnDevice<
                   ValenciaDivClean::BoundaryCorrections::Hll>>
      time_derivative_{};
  // The damping functions at the time of the substep, which the time
  // derivative reads on the device
  DeviceParameters<std::array<ConstraintDamping::GaussianSum<3>, 3>>
      gaussian_sums_{"GhValenciaDivClean damping functions"};

  // Scratch data
  // The hydro part of the ghost variables (the GH part is read from the
  // evolved variables)
  using hydro_ghost_tags =
      GhValenciaDivClean::Tags::primitive_grmhd_reconstruction_tags;
  BatchedVariables<hydro_ghost_tags> ghost_variables_{};
  // The time derivative in the latest record of the history (see
  // `TimeSteppers::derivative_batch`), for the element boxes
  BatchedVariables<subcell::dt_tags_on_device> dt_variables_{};
  BatchedVariables<subcell::fixing_metric_tags> fixing_metric_{};
  // The dense output, its fixing metric and the failures of its fixing,
  // allocated at the first dense output
  EvolvedVariables dense_evolved_variables_{};
  PrimitiveVariables dense_primitives_{};
  BatchedVariables<subcell::fixing_metric_tags> dense_fixing_metric_{};
  typename ValenciaDivClean::fd_patch::FixingAndRecovery<
      OrderedListOfPrimitiveRecoverySchemes>::Statuses dense_statuses_{};
  // The values saved during the self start
  evolution::patches::SavedBatches<EvolvedVariables, PrimitiveVariables>
      initial_values_{};
  std::vector<std::array<std::optional<StepperErrorEstimate>, 2>>
      stepper_errors_{};
  // Buffers of the RDMP data, kept so they aren't allocated every substep
  ValenciaDivClean::subcell::RdmpTciDataBuffers rdmp_tci_data_buffers_{};
  // The observables computed on the device and their buffers, kept so they
  // aren't allocated every observation
  evolution::patches::DeviceObservables device_observables_{};
  DeviceObservablesBuffers device_observables_buffers_{};
};
}  // namespace grmhd::GhValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
