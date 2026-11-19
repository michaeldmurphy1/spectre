// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
namespace evolution::dg::subcell::fd_patch {
class FdPatchElements;
}  // namespace evolution::dg::subcell::fd_patch
/// \endcond

namespace grmhd::ValenciaDivClean::fd_patch {
/// The spatial metric, its inverse and the square root of its determinant,
/// which the fixing and the primitive recovery use
using fixing_metric_tags =
    tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>>;

/*!
 * \brief The fixing of the GRMHD variables and the recovery of the primitive
 * variables of the elements of an FD patch on the device, shared by the
 * patches of the systems that evolve the GRMHD variables (e.g.
 * `grmhd::ValenciaDivClean::fd_patch::Patch`).
 *
 * \details This holds the options of the fixing and the recovery, the scratch
 * data they reuse between steps, and the `DeferredPointwiseStatus` of each
 * operation that can fail at a grid point, including the fixing of the
 * reconstructed states on the faces in the time derivative (`face_status()`).
 * A step of the patch resets the statuses (`reset_statuses()`), enqueues its
 * operations, and copies the statuses to the host (`copy_statuses_to_host()`)
 * without waiting. `check()` reports the failures after the patch waited for
 * the device.
 *
 * The functions take the evolved variables as any batch that holds the
 * ValenciaDivClean conservative variables (e.g. with the GH variables of
 * GhValenciaDivClean), and the spatial metric etc. as any batch that holds
 * the `fixing_metric_tags`. Their definitions are in
 * `FixingAndRecoveryKokkos.tpp`, which the patches of the systems include.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
class FixingAndRecovery {
 public:
  using Primitives = BatchedVariables<hydro::grmhd_tags<DataVector>>;

  /*!
   * \brief The failures of the operations since the last `reset()`, recorded
   * on the device.
   *
   * \details The patch holds those of its substeps (`reset_statuses()`,
   * `copy_statuses_to_host()`, `check()`), and may record the failures of
   * other operations, e.g. of dense output, separately, so that they are
   * reported with their own context and not as failures of the substep.
   */
  class Statuses {
   public:
    Statuses() = default;
    /// Allocated on `exec`, without failures
    explicit Statuses(const Kokkos::DefaultExecutionSpace& exec);

    /// Reset to no failures, on `exec`
    void reset(const Kokkos::DefaultExecutionSpace& exec) const;

    /// Enqueue copying the statuses to the host, for `check()`
    void copy_to_host(const Kokkos::DefaultExecutionSpace& exec) const;

    /// ERRORs with the variables at a failed grid point if an operation
    /// failed since the last `reset()`, in the order of a step. The
    /// `context`, e.g. "the dense output at time 1.5 of ", goes before the
    /// description of the element. Waits for `exec`. See
    /// `FixingAndRecovery::check()`.
    void check(
        const Kokkos::DefaultExecutionSpace& exec,
        const evolution::dg::subcell::fd_patch::FdPatchElements& patch_elements,
        size_t number_of_face_points, const std::string& context = "") const;

   private:
    friend class FixingAndRecovery;
    // Their device data changes in const member functions, like that of other
    // views.
    DeferredPointwiseStatus<fd::face_atmosphere_snapshot_tags> face_status_{};
    DeferredPointwiseStatus<fix_conservatives_snapshot_tags>
        fix_conservatives_status_{};
    DeferredPointwiseStatus<primitive_recovery_snapshot_tags>
        recovery_status_{};
    DeferredPointwiseStatus<::VariableFixing::fix_to_atmosphere_snapshot_tags>
        atmosphere_status_{};
  };

  FixingAndRecovery() = default;
  FixingAndRecovery(const FixingAndRecovery&) = delete;
  FixingAndRecovery& operator=(const FixingAndRecovery&) = delete;
  FixingAndRecovery(FixingAndRecovery&&) = default;
  FixingAndRecovery& operator=(FixingAndRecovery&&) = default;
  ~FixingAndRecovery() = default;

  /// The statuses are allocated on `exec` and have no failures.
  FixingAndRecovery(
      const Kokkos::DefaultExecutionSpace& exec,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const ::VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
      const FixConservatives& fix_conservatives,
      const PrimitiveFromConservativeOptions&
          primitive_from_conservative_options);

  const EquationsOfState::EquationOfState<true, 3>& equation_of_state() const {
    return *equation_of_state_;
  }
  const ::VariableFixing::FixToAtmosphere<3>::DeviceView& fix_to_atmosphere()
      const {
    return fix_to_atmosphere_;
  }

  /// Where the time derivative records the failures of fixing the
  /// reconstructed states on the faces to the atmosphere
  gsl::not_null<DeferredPointwiseStatus<fd::face_atmosphere_snapshot_tags>*>
  face_status() {
    return make_not_null(&statuses_.face_status_);
  }

  /// Reset the statuses of all operations to no failures, on `exec`
  void reset_statuses(const Kokkos::DefaultExecutionSpace& exec) const;

  /// Enqueue copying the statuses to the host, for `check()`
  void copy_statuses_to_host(const Kokkos::DefaultExecutionSpace& exec) const;

  /// @{
  /// Fix the conserved variables (`fix_conservatives_on_device`) and recover
  /// the primitive variables (`primitive_from_conservative_on_device`),
  /// enqueued on `exec`. The failures are recorded in the statuses of the
  /// substep, or in `statuses`.
  template <typename EvolvedTags, typename MetricTags>
  void fix_conservatives_and_recover_primitives(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<BatchedVariables<EvolvedTags>*> evolved_variables,
      gsl::not_null<Primitives*> primitives,
      const BatchedVariables<MetricTags>& metric);
  template <typename EvolvedTags, typename MetricTags>
  void fix_conservatives_and_recover_primitives(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<BatchedVariables<EvolvedTags>*> evolved_variables,
      gsl::not_null<Primitives*> primitives,
      const BatchedVariables<MetricTags>& metric,
      gsl::not_null<Statuses*> statuses);
  /// @}

  /// Fix the primitive variables to the atmosphere
  /// (`::VariableFixing::fix_to_atmosphere_on_device`), enqueued on `exec`
  template <typename MetricTags>
  void fix_to_atmosphere(const Kokkos::DefaultExecutionSpace& exec,
                         gsl::not_null<Primitives*> primitives,
                         const BatchedVariables<MetricTags>& metric);

  /// Fix the primitive variables to the atmosphere at once, e.g. of the
  /// initial data, and `ERROR` if that fails at a grid point of the
  /// `patch_elements`
  template <typename MetricTags>
  void fix_to_atmosphere_now(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<Primitives*> primitives,
      const BatchedVariables<MetricTags>& metric,
      const evolution::dg::subcell::fd_patch::FdPatchElements& patch_elements)
      const;

  /// Compute the conserved variables from the primitive variables
  /// (`ConservativeFromPrimitive`), enqueued on `exec`
  template <typename EvolvedTags, typename MetricTags>
  void compute_conservatives(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<BatchedVariables<EvolvedTags>*> evolved_variables,
      const Primitives& primitives,
      const BatchedVariables<MetricTags>& metric) const;

  /*!
   * \brief ERRORs with the variables at a failed grid point if an operation
   * failed since the last `reset_statuses()`, in the order of a step. Waits
   * for `exec`.
   *
   * \details The elements are those of `patch_elements`. The faces of the
   * time derivative have `number_of_face_points` points per face set, and
   * there are 3 face sets per element.
   */
  void check(
      const Kokkos::DefaultExecutionSpace& exec,
      const evolution::dg::subcell::fd_patch::FdPatchElements& patch_elements,
      size_t number_of_face_points) const;

 private:
  std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>
      equation_of_state_{};
  ::VariableFixing::FixToAtmosphere<3>::DeviceView fix_to_atmosphere_{};
  FixConservatives fix_conservatives_{};
  PrimitiveFromConservativeOptions primitive_from_conservative_options_{};
  // Buffers of the per-element decisions, kept so they aren't allocated every
  // step
  Kokkos::View<bool*> fix_conservatives_scratch_{};
  Kokkos::View<bool*> primitive_recovery_scratch_{};
  // The failures of the substep since the last `reset_statuses()`
  Statuses statuses_{};
};
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
