// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/HistoryCoefficients.hpp"

#include <complex>
#include <cstddef>
#include <optional>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Time/History.hpp"
#include "Time/LargestStepperError.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {
bool HistoryTimes::at_step_start() const {
  ASSERT(not steps.empty(), "History is empty");
  return substeps.empty() or substeps.back() < steps.back();
}

void HistoryLinearCombination::add_term(const HistoryTerm::Record record,
                                        const size_t index,
                                        const HistoryTerm::Field field,
                                        const double coefficient) {
  ASSERT(size < terms.size(),
         "Cannot add more than " << terms.size() << " terms.");
  ASSERT(
      index < (record == HistoryTerm::Record::Step ? history_max_past_steps + 2
                                                   : history_max_substeps),
      "History index " << index << " is out of range.");
  gsl::at(terms, size) = HistoryTerm{.record = record,
                                     .index = static_cast<uint8_t>(index),
                                     .field = field,
                                     .coefficient = coefficient};
  ++size;
}

namespace {
template <typename T>
const T& history_data(const ConstUntypedHistory<T>& history,
                      const HistoryTerm& term) {
  const auto& record = term.record == HistoryTerm::Record::Step
                           ? history[term.index]
                           : history.substeps()[term.index];
  if (term.field == HistoryTerm::Field::Derivative) {
    return record.derivative;
  }
  ASSERT(record.value.has_value(),
         "The value at " << record.time_step_id << " has been discarded.");
  return *record.value;
}
}  // namespace

template <typename T>
void apply_history_linear_combination(
    const gsl::not_null<T*> result, const ConstUntypedHistory<T>& history,
    const HistoryLinearCombination& combination) {
  ASSERT(combination.accumulate or combination.size > 0,
         "Cannot overwrite the result with an empty linear combination.");
  for (size_t i = 0; i < combination.size; ++i) {
    const HistoryTerm& term = gsl::at(combination.terms, i);
    const T& data = history_data(history, term);
    if (i == 0 and not combination.accumulate) {
      if (term.coefficient == 1.0) {
        *result = data;
      } else {
        *result = term.coefficient * data;
      }
    } else {
      if (term.coefficient == 1.0) {
        *result += data;
      } else {
        *result += term.coefficient * data;
      }
    }
  }
}

template <typename T>
std::optional<StepperErrorEstimate> apply_step_coefficients(
    const gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
    const StepCoefficients& coefficients,
    const StepperErrorTolerances& tolerances) {
  std::optional<StepperErrorEstimate> error{};
  if (coefficients.error.has_value()) {
    const ErrorCoefficients& error_coefficients = *coefficients.error;
    error.emplace();
    error->step_time = error_coefficients.step_time;
    error->step_size = error_coefficients.step_size;
    error->order = error_coefficients.order;
    // Use u as scratch space for the errors.  It is overwritten by the
    // update below.
    for (size_t i = 0; i < error_coefficients.errors.size(); ++i) {
      if (gsl::at(error_coefficients.errors, i).has_value()) {
        apply_history_linear_combination(
            u, history, *gsl::at(error_coefficients.errors, i));
        gsl::at(error->errors, i) =
            largest_stepper_error(*history.back().value, *u, tolerances);
      }
    }
    ASSERT(gsl::at(error->errors, error->order).has_value(),
           "No error estimate at the stepper order " << error->order);
  }
  apply_history_linear_combination(u, history, coefficients.update);
  return error;
}

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template void apply_history_linear_combination(                       \
      gsl::not_null<TYPE(data)*> result,                                \
      const ConstUntypedHistory<TYPE(data)>& history,                   \
      const HistoryLinearCombination& combination);                     \
  template std::optional<StepperErrorEstimate> apply_step_coefficients( \
      gsl::not_null<TYPE(data)*> u,                                     \
      const ConstUntypedHistory<TYPE(data)>& history,                   \
      const StepCoefficients& coefficients,                             \
      const StepperErrorTolerances& tolerances);

GENERATE_INSTANTIATIONS(INSTANTIATE, (MATH_WRAPPER_TYPES))

#undef INSTANTIATE
#undef TYPE
}  // namespace TimeSteppers
