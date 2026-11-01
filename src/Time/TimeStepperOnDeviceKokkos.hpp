// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedVariant.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/SelfStart.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/DeviceHistoryEvolvedVariablesKokkos.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/LtsStepChoosers.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerancesCompute.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Tags/VariableOrderAlgorithm.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/HistoryCoefficientsKokkos.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
namespace Tags {
template <typename Tag>
struct Next;
}  // namespace Tags
/// \endcond

namespace TimeSteppers {
/// \ingroup TimeSteppersGroup
/// Device version of `TimeStepper::dense_update_u` for a host `vars`
/// and a history with its data on the device.  Returns false if dense
/// output is not possible yet, as `dense_update_u` does.  Unlike
/// `dense_update_u`, this also sets \p vars to the value at the start of
/// the step.
template <typename ExecutionSpace, typename Vars>
bool dense_update_u_on_device(
    const ExecutionSpace& exec, const gsl::not_null<Vars*> vars,
    const gsl::not_null<DeviceHistory<typename ExecutionSpace::memory_space>*>
        history,
    const TimeStepper& time_stepper, const double time) {
  const auto coefficients =
      time_stepper.dense_output_coefficients(history->times(), time);
  if (not coefficients.has_value()) {
    return false;
  }
  const auto result = history->scratch();
  Kokkos::deep_copy(exec, result, history->step_start_value(time));
  apply_history_linear_combination_on_device(
      exec, result, history->device_view(), *coefficients);
  copy_to_host(exec, gsl::span<double>(vars->data(), vars->size()), result);
  return true;
}
}  // namespace TimeSteppers

/*!
 * \ingroup TimeGroup
 * \brief Records the variables and their time derivatives in the time
 * stepper history on the device.
 *
 * This is the device version of `RecordTimeStepperData`.  The data is
 * copied from the host, since the variables are still stored there.
 * Only a single variables tag is supported.
 */
template <typename System>
struct RecordTimeStepperDataOnDevice {
  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  using return_tags =
      tmpl::list<::Tags::DeviceHistoryEvolvedVariables<variables_tag>>;
  using argument_tags =
      tmpl::list<::Tags::TimeStepId, variables_tag, dt_variables_tag,
                 ::Tags::HistoryEvolvedVariables<variables_tag>>;

  static void apply(
      const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
      const TimeStepId& time_step_id, const typename variables_tag::type& vars,
      const typename dt_variables_tag::type& dt_vars,
      const TimeSteppers::History<typename variables_tag::type>& host_history) {
    if (not host_history.empty()) {
      ERROR("The host time stepper history has "
            << host_history.size()
            << " entries, but the history is being recorded on the device.  "
               "Moving the history between the host and the device (e.g., when "
               "switching between DG and FD) is not supported yet.");
    }
    const Kokkos::DefaultExecutionSpace exec{};
    if (history->empty()) {
      history->set_number_of_values(vars.size());
    }
    const auto record = history->insert(time_step_id, true);
    TimeSteppers::copy_to_device(
        exec, record.value, gsl::span<const double>(vars.data(), vars.size()));
    TimeSteppers::copy_to_device(
        exec, record.derivative,
        gsl::span<const double>(dt_vars.data(), dt_vars.size()));
    // The host data must stay valid until the copies are done.
    exec.fence();
  }
};

/*!
 * \ingroup TimeGroup
 * \brief Updates the variables using the time stepper history on the
 * device.
 *
 * This is the device version of `UpdateU`, with the same error
 * estimates.  The result is copied to the host, since the variables
 * are still stored there.  Only a single variables tag is supported.
 */
template <typename System>
struct UpdateUOnDevice {
  using variables_tag = typename System::variables_tag;

  using const_global_cache_tags =
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
                 ::Tags::LtsStepChoosers, ::Tags::VariableOrderAlgorithm>;
  using simple_tags = tmpl::list<::Tags::StepperErrors<variables_tag>>;
  using compute_tags =
      tmpl::list<Tags::StepperErrorEstimatesEnabledCompute<>,
                 Tags::StepperErrorTolerancesCompute<variables_tag>>;

  using return_tags =
      tmpl::list<variables_tag, ::Tags::StepperErrors<variables_tag>,
                 ::Tags::DeviceHistoryEvolvedVariables<variables_tag>>;
  using argument_tags =
      tmpl::list<::Tags::TimeStepper<TimeStepper>, ::Tags::TimeStepId,
                 ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
                 ::Tags::StepperErrorEstimatesEnabled,
                 ::Tags::StepperErrorTolerances<variables_tag>>;

  static void apply(
      const gsl::not_null<typename variables_tag::type*> vars,
      const gsl::not_null<std::array<std::optional<StepperErrorEstimate>, 2>*>
          errors,
      const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
      const TimeStepper& time_stepper, const TimeStepId& time_step_id,
      const TimeStepId& next_time_step_id, const TimeDelta& time_step,
      const bool error_estimates_enabled,
      const StepperErrorTolerances& stepper_tolerances) {
    if (::SelfStart::step_unused(time_step_id, next_time_step_id)) {
      return;
    }
    const Kokkos::DefaultExecutionSpace exec{};
    const auto result = history->scratch();
    if (error_estimates_enabled) {
      // As in `UpdateU`, variable-order h-refinement needs low-order
      // error estimates.
      StepperErrorTolerances tolerances = stepper_tolerances;
      if (tolerances.estimates != StepperErrorTolerances::Estimates::None and
          variants::holds_alternative<TimeSteppers::Tags::VariableOrder>(
              time_stepper.order()) and
          (history->bookkeeping().back().time_step_id.step_time() + time_step)
              .is_at_slab_boundary()) {
        tolerances.estimates = StepperErrorTolerances::Estimates::AllOrders;
      }
      const auto new_errors = TimeSteppers::apply_step_coefficients_on_device(
          exec, result, history->device_view(),
          time_stepper.step_coefficients(history->times(), time_step,
                                         tolerances),
          tolerances);
      const auto& new_error = new_errors.front();
      if (new_error.has_value()) {
        // Save the previous error if there was one, but not if this is a
        // retry of the same step.
        if ((*errors)[1].has_value() and
            (*errors)[1]->step_time != new_error->step_time) {
          (*errors)[0] = (*errors)[1];
        }
        (*errors)[1].emplace(*new_error);
      }
    } else {
      TimeSteppers::apply_step_coefficients_on_device(
          exec, result, history->device_view(),
          time_stepper.step_coefficients(history->times(), time_step),
          StepperErrorTolerances{});
    }
    TimeSteppers::copy_to_host(
        exec, gsl::span<double>(vars->data(), vars->size()), result);
  }
};

/// \ingroup TimeGroup
/// \brief Removes old entries from the time stepper history on the
/// device.  This is the device version of `CleanHistory`.
template <typename System>
struct CleanHistoryOnDevice {
  using variables_tag = typename System::variables_tag;

  using return_tags =
      tmpl::list<::Tags::DeviceHistoryEvolvedVariables<variables_tag>>;
  using argument_tags = tmpl::list<::Tags::TimeStepper<TimeStepper>>;

  static void apply(const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
                    const TimeStepper& time_stepper) {
    history->clean(time_stepper);
  }
};

namespace Initialization {
/*!
 * \ingroup InitializationGroup
 * \brief Adds the time stepper history on the device.
 *
 * The history starts at the same integration order as the one added by
 * `Initialization::TimeStepperHistory`.  The size of its data is set
 * when the first entry is recorded.
 */
template <typename System>
struct DeviceTimeStepperHistory {
  using variables_tag = typename System::variables_tag;
  using history_tag = ::Tags::DeviceHistoryEvolvedVariables<variables_tag>;

  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<history_tag>;
  using compute_tags = tmpl::list<>;

  using argument_tags =
      tmpl::list<::Tags::HistoryEvolvedVariables<variables_tag>>;
  using return_tags = simple_tags;

  static void apply(
      const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
      const TimeSteppers::History<typename variables_tag::type>& host_history) {
    *history =
        TimeSteppers::DeviceHistory<>(host_history.integration_order(), 0);
  }
};
}  // namespace Initialization
#endif  // SPECTRE_KOKKOS
