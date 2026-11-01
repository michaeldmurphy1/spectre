// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

#include "DataStructures/MathWrapper.hpp"
#include "Time/History.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
struct StepperErrorTolerances;
/// \endcond

namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * \brief The times stored in a time-stepper `History`, without any data.
 *
 * The coefficients a `TimeStepper` applies to the history data depend
 * only on these times and on the integration order, so they can be
 * computed on the host without access to the data.  This allows the
 * data to be stored elsewhere, e.g., in device memory.
 */
struct HistoryTimes {
  HistoryTimes() = default;

  /// Construct from a `History` or a `ConstUntypedHistory`.
  template <typename HistoryType>
  explicit HistoryTimes(const HistoryType& history)
      : integration_order(history.integration_order()) {
    for (const auto& record : history) {
      steps.push_back(record.time_step_id);
    }
    for (const auto& record : history.substeps()) {
      substeps.push_back(record.time_step_id);
    }
  }

  size_t integration_order{};
  /// The `TimeStepId`s of the steps, earliest first.
  boost::container::static_vector<TimeStepId, history_max_past_steps + 2>
      steps{};
  /// The `TimeStepId`s of the substeps.  See `History::substeps()`.
  boost::container::static_vector<TimeStepId, history_max_substeps> substeps{};

  /// Whether the most recent entry is a step, not a substep.
  bool at_step_start() const;
};

/*!
 * \ingroup TimeSteppersGroup
 * \brief One term \f$c\,y\f$ of a `HistoryLinearCombination`, where
 * \f$y\f$ is the value or derivative of a record in the history.
 */
struct HistoryTerm {
  /// Whether the record is one of the steps or one of the substeps.
  enum class Record : uint8_t { Step, Substep };
  /// Whether the term uses the value or the derivative of the record.
  enum class Field : uint8_t { Value, Derivative };

  Record record{};
  /// Index into `HistoryTimes::steps` or `HistoryTimes::substeps`.
  uint8_t index{};
  Field field{};
  double coefficient{};
};

/// Largest number of terms in a `HistoryLinearCombination`.  An
/// order-8 Adams step uses the step-start value and eight
/// derivatives.
constexpr size_t max_history_terms = adams_coefficients::maximum_order + 1;
// A Runge-Kutta step uses the step-start value, the step-start
// derivative, and the derivatives of all the substeps.  Dense output
// can also use the derivative at the end of the step.
static_assert(history_max_substeps + 2 <= max_history_terms);

/*!
 * \ingroup TimeSteppersGroup
 * \brief A linear combination of history values and derivatives.
 *
 * The result is
 *
 * \f{equation}{
 *   u \leftarrow [u] + \sum_i c_i y_i,
 * \f}
 *
 * where the \f$y_i\f$ and \f$c_i\f$ are given by the `terms`, and the
 * initial \f$u\f$ is included only if `accumulate` is true.  Any
 * factors of the step size are included in the coefficients.  The
 * terms must be applied in order to reproduce the results of the
 * `TimeStepper` methods exactly.
 *
 * The terms only reference history records by index, so a generic
 * kernel can apply them to history data stored anywhere.
 *
 * \note On the device, `DeviceHistory` stores the history data of a
 * batch of elements that share their record times, e.g., the FD
 * elements of a patch, so the same coefficients apply to all of them.
 */
struct HistoryLinearCombination {
  /// Whether to add to the existing value of the result instead of
  /// overwriting it.
  bool accumulate = false;
  size_t size = 0;
  std::array<HistoryTerm, max_history_terms> terms{};

  void add_term(HistoryTerm::Record record, size_t index,
                HistoryTerm::Field field, double coefficient);
};

/*!
 * \ingroup TimeSteppersGroup
 * \brief The linear combinations needed to compute a
 * `StepperErrorEstimate`.
 *
 * Each entry of `errors` mirrors the corresponding entry of
 * `StepperErrorEstimate::errors`.  The error is measured relative to
 * the value of the latest step, i.e., the error ratio is computed by
 * `largest_stepper_error` with that value and the result of the
 * linear combination.
 */
struct ErrorCoefficients {
  Time step_time{};
  TimeDelta step_size{};
  size_t order{};
  std::array<std::optional<HistoryLinearCombination>,
             std::tuple_size_v<decltype(StepperErrorEstimate::errors)>>
      errors{};
};

/// \ingroup TimeSteppersGroup
/// The coefficients for a call to `TimeStepper::update_u`.
struct StepCoefficients {
  HistoryLinearCombination update{};
  /// Present only if the substep produces an error estimate.
  std::optional<ErrorCoefficients> error{};
};

/*!
 * \ingroup TimeSteppersGroup
 * \brief The coefficients for an implicit substep of an
 * `ImexTimeStepper`.
 *
 * `inhomogeneous` accumulates the inhomogeneous terms into the
 * explicit result, and `weight` is the implicit weight.  See
 * `ImexTimeStepper` for details.
 */
struct ImplicitCoefficients {
  HistoryLinearCombination inhomogeneous{};
  double weight{};
};

/// \ingroup TimeSteppersGroup
/// Apply \p combination to the data in \p history, storing the result
/// in \p result.
/// @{
template <typename T>
void apply_history_linear_combination(
    gsl::not_null<T*> result, const ConstUntypedHistory<T>& history,
    const HistoryLinearCombination& combination);

template <typename Vars>
void apply_history_linear_combination(
    const gsl::not_null<Vars*> result, const History<Vars>& history,
    const HistoryLinearCombination& combination) {
  apply_history_linear_combination(make_not_null(&*make_math_wrapper(result)),
                                   history.untyped(), combination);
}
/// @}

/// \ingroup TimeSteppersGroup
/// Apply \p coefficients to the data in \p history, storing the
/// updated value in \p u and returning the error estimate if one is
/// requested.  This performs the same calculation as
/// `TimeStepper::update_u`.
/// @{
template <typename T>
std::optional<StepperErrorEstimate> apply_step_coefficients(
    gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
    const StepCoefficients& coefficients,
    const StepperErrorTolerances& tolerances);

template <typename Vars>
std::optional<StepperErrorEstimate> apply_step_coefficients(
    const gsl::not_null<Vars*> u, const History<Vars>& history,
    const StepCoefficients& coefficients,
    const StepperErrorTolerances& tolerances) {
  return apply_step_coefficients(make_not_null(&*make_math_wrapper(u)),
                                 history.untyped(), coefficients, tolerances);
}
/// @}
}  // namespace TimeSteppers
