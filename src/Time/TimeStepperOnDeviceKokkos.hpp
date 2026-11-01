// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/TaggedVariant.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/SelfStart.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/HistoryCoefficientsKokkos.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace TimeSteppers {
namespace TimeStepperOnDevice_detail {
// The data of the batch as history data, which has the same layout
template <typename TagsList>
HistoryBatchView<> as_history_batch(const BatchedVariables<TagsList>& batch) {
  const size_t number_of_values =
      batch.number_of_grid_points() *
      BatchedVariables<TagsList>::number_of_independent_components;
  return HistoryBatchView<>(
      batch.view().data(),
      Kokkos::LayoutStride(number_of_values, 1, batch.number_of_elements(),
                           number_of_values));
}

template <typename TagsList>
void check_sizes(const DeviceHistory<>& history,
                 const BatchedVariables<TagsList>& batch) {
  ASSERT(history.number_of_values() ==
                 batch.number_of_grid_points() *
                     BatchedVariables<
                         TagsList>::number_of_independent_components and
             history.number_of_elements() == batch.number_of_elements(),
         "The history holds "
             << history.number_of_values() << " values of "
             << history.number_of_elements() << " elements, but the batch has "
             << batch.number_of_grid_points() *
                    BatchedVariables<TagsList>::number_of_independent_components
             << " values of " << batch.number_of_elements() << " elements.");
}
}  // namespace TimeStepperOnDevice_detail

/*!
 * \ingroup TimeSteppersGroup
 * \brief Records the variables `vars` and their time derivatives `dt_vars` of
 * a batch of elements in the time stepper history, like
 * `RecordTimeStepperData` does for each element.
 *
 * The `history` must hold the data of the elements of the batch (see
 * `DeviceHistory`). The data is copied on the device, asynchronously in `exec`.
 */
template <typename VarsTags, typename DtVarsTags>
void record_on_device(const Kokkos::DefaultExecutionSpace& exec,
                      const gsl::not_null<DeviceHistory<>*> history,
                      const TimeStepId& time_step_id,
                      const BatchedVariables<VarsTags>& vars,
                      const BatchedVariables<DtVarsTags>& dt_vars) {
  TimeStepperOnDevice_detail::check_sizes(*history, vars);
  TimeStepperOnDevice_detail::check_sizes(*history, dt_vars);
  const auto record = history->insert(exec, time_step_id, true);
  Kokkos::deep_copy(exec, record.value,
                    TimeStepperOnDevice_detail::as_history_batch(vars));
  Kokkos::deep_copy(exec, record.derivative,
                    TimeStepperOnDevice_detail::as_history_batch(dt_vars));
}

/*!
 * \ingroup TimeSteppersGroup
 * \brief Updates the variables `vars` of a batch of elements with the time
 * stepper history, like `UpdateU` does for each element.
 *
 * As in `UpdateU`, nothing is done on a step that is unused in the self-start
 * procedure, and the `errors` of each element hold the error estimates of the
 * current and the previous step (if `error_estimates_enabled`). The `history`
 * must hold the data of the elements of the batch (see `DeviceHistory`).
 *
 * The update is written to `vars` on the device in `exec`. With error
 * estimates, this waits for `exec` to complete, since the estimates are needed
 * on the host.
 */
template <typename VarsTags>
void update_u_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<VarsTags>*> vars,
    const gsl::not_null<
        std::vector<std::array<std::optional<StepperErrorEstimate>, 2>>*>
        errors,
    const DeviceHistory<>& history, const TimeStepper& time_stepper,
    const TimeStepId& time_step_id, const TimeStepId& next_time_step_id,
    const TimeDelta& time_step, const bool error_estimates_enabled,
    const StepperErrorTolerances& stepper_tolerances) {
  if (::SelfStart::step_unused(time_step_id, next_time_step_id)) {
    return;
  }
  TimeStepperOnDevice_detail::check_sizes(history, *vars);
  const auto result = TimeStepperOnDevice_detail::as_history_batch(*vars);
  if (not error_estimates_enabled) {
    apply_step_coefficients_on_device(
        exec, result, history.device_view(),
        time_stepper.step_coefficients(history.times(), time_step),
        StepperErrorTolerances{});
    return;
  }
  // As in `UpdateU`, variable-order h-refinement needs low-order error
  // estimates.
  StepperErrorTolerances tolerances = stepper_tolerances;
  if (tolerances.estimates != StepperErrorTolerances::Estimates::None and
      variants::holds_alternative<TimeSteppers::Tags::VariableOrder>(
          time_stepper.order()) and
      (history.bookkeeping().back().time_step_id.step_time() + time_step)
          .is_at_slab_boundary()) {
    tolerances.estimates = StepperErrorTolerances::Estimates::AllOrders;
  }
  const auto new_errors = apply_step_coefficients_on_device(
      exec, result, history.device_view(),
      time_stepper.step_coefficients(history.times(), time_step, tolerances),
      tolerances);
  errors->resize(vars->number_of_elements());
  for (size_t e = 0; e < new_errors.size(); ++e) {
    const auto& new_error = new_errors[e];
    auto& element_errors = (*errors)[e];
    if (new_error.has_value()) {
      // Save the previous error if there was one, but not if this is a retry
      // of the same step.
      if (element_errors[1].has_value() and
          element_errors[1]->step_time != new_error->step_time) {
        element_errors[0] = element_errors[1];
      }
      element_errors[1].emplace(*new_error);
    }
  }
}

/*!
 * \ingroup TimeSteppersGroup
 * \brief Dense output of the variables of a batch of elements at `time`, like
 * `TimeStepper::dense_update_u` for each element.
 *
 * Returns false if dense output is not possible yet, as `dense_update_u`
 * does. Unlike `dense_update_u`, this also sets `vars` to the value at the
 * start of the step. The result is written to `vars` on the device in `exec`.
 */
template <typename VarsTags>
bool dense_update_u_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<VarsTags>*> vars,
    const DeviceHistory<>& history, const TimeStepper& time_stepper,
    const double time) {
  TimeStepperOnDevice_detail::check_sizes(history, *vars);
  const auto coefficients =
      time_stepper.dense_output_coefficients(history.times(), time);
  if (not coefficients.has_value()) {
    return false;
  }
  const auto result = TimeStepperOnDevice_detail::as_history_batch(*vars);
  Kokkos::deep_copy(exec, result, history.step_start_value(time));
  apply_history_linear_combination_on_device(
      exec, result, history.device_view(), *coefficients);
  return true;
}
}  // namespace TimeSteppers
#endif  // SPECTRE_KOKKOS
