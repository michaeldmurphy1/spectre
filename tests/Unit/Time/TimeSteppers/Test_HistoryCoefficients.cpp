// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/HistoryCoefficients.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using Record = TimeSteppers::HistoryTerm::Record;
using Field = TimeSteppers::HistoryTerm::Field;

// A history with two steps, the second of which has a substep.
TimeSteppers::History<DataVector> make_history() {
  const Slab slab(0.0, 1.0);
  const auto step = slab.duration() / 2;
  TimeSteppers::History<DataVector> history{3};
  const TimeStepId first(true, 0, slab.start());
  const TimeStepId second = first.next_step(step);
  history.insert(first, DataVector{1.0, 2.0}, DataVector{3.0, 4.0});
  history.insert(second, DataVector{5.0, 6.0}, DataVector{7.0, 8.0});
  history.insert(second.next_substep(step, 1.0), DataVector{9.0, 10.0},
                 DataVector{11.0, 12.0});
  return history;
}

void test_history_times() {
  auto history = make_history();
  const TimeSteppers::HistoryTimes times(history);
  CHECK(times.integration_order == 3);
  REQUIRE(times.steps.size() == 2);
  CHECK(times.steps[0] == history[0].time_step_id);
  CHECK(times.steps[1] == history[1].time_step_id);
  REQUIRE(times.substeps.size() == 1);
  CHECK(times.substeps[0] == history.substeps()[0].time_step_id);
  CHECK(not times.at_step_start());
  CHECK(TimeSteppers::HistoryTimes(history.untyped()).substeps ==
        times.substeps);

  history.clear_substeps();
  CHECK(TimeSteppers::HistoryTimes(history).at_step_start());
}

void test_linear_combination() {
  const auto history = make_history();

  TimeSteppers::HistoryLinearCombination combination{};
  combination.add_term(Record::Step, 1, Field::Value, 1.0);
  combination.add_term(Record::Step, 0, Field::Derivative, 2.0);
  combination.add_term(Record::Substep, 0, Field::Value, -1.0);
  combination.add_term(Record::Substep, 0, Field::Derivative, 0.5);
  CHECK(combination.size == 4);
  CHECK(combination.terms[2].record == Record::Substep);
  CHECK(combination.terms[2].index == 0);
  CHECK(combination.terms[2].field == Field::Value);
  CHECK(combination.terms[2].coefficient == -1.0);

  const DataVector expected =
      DataVector{5.0, 6.0} + 2.0 * DataVector{3.0, 4.0} -
      DataVector{9.0, 10.0} + 0.5 * DataVector{11.0, 12.0};
  DataVector result{std::numeric_limits<double>::signaling_NaN(),
                    std::numeric_limits<double>::signaling_NaN()};
  TimeSteppers::apply_history_linear_combination(make_not_null(&result),
                                                 history, combination);
  CHECK_ITERABLE_APPROX(result, expected);

  combination.accumulate = true;
  result = DataVector{100.0, 200.0};
  TimeSteppers::apply_history_linear_combination(make_not_null(&result),
                                                 history, combination);
  CHECK_ITERABLE_APPROX(result, (expected + DataVector{100.0, 200.0}));

  // An empty accumulating combination does nothing.
  result = DataVector{100.0, 200.0};
  TimeSteppers::apply_history_linear_combination(
      make_not_null(&result), history,
      TimeSteppers::HistoryLinearCombination{.accumulate = true});
  CHECK(result == DataVector{100.0, 200.0});

#ifdef SPECTRE_DEBUG
  TimeSteppers::HistoryLinearCombination full{};
  for (size_t i = 0; i < TimeSteppers::max_history_terms; ++i) {
    full.add_term(Record::Step, 0, Field::Value, 1.0);
  }
  CHECK_THROWS_WITH(full.add_term(Record::Step, 0, Field::Value, 1.0),
                    Catch::Matchers::ContainsSubstring("Cannot add more than"));
#endif  // SPECTRE_DEBUG
}

void test_step_coefficients() {
  const auto history = make_history();
  const StepperErrorTolerances tolerances{
      .estimates = StepperErrorTolerances::Estimates::AllOrders,
      .absolute = 1.0,
      .relative = 0.5};

  TimeSteppers::StepCoefficients coefficients{};
  coefficients.update.add_term(Record::Step, 1, Field::Value, 1.0);
  coefficients.update.add_term(Record::Substep, 0, Field::Derivative, 0.25);
  DataVector u{std::numeric_limits<double>::signaling_NaN(),
               std::numeric_limits<double>::signaling_NaN()};
  CHECK(not TimeSteppers::apply_step_coefficients(make_not_null(&u), history,
                                                  coefficients, tolerances)
                .has_value());
  const DataVector expected_u =
      DataVector{5.0, 6.0} + 0.25 * DataVector{11.0, 12.0};
  CHECK_ITERABLE_APPROX(u, expected_u);

  const Slab slab(0.0, 1.0);
  auto& error = coefficients.error.emplace();
  error.step_time = slab.start();
  error.step_size = slab.duration();
  error.order = 2;
  gsl::at(error.errors, 2)
      .emplace()
      .add_term(Record::Step, 0, Field::Derivative, 0.1);
  gsl::at(error.errors, 0)
      .emplace()
      .add_term(Record::Substep, 0, Field::Derivative, -0.01);
  const auto estimate = TimeSteppers::apply_step_coefficients(
      make_not_null(&u), history, coefficients, tolerances);
  CHECK_ITERABLE_APPROX(u, expected_u);
  REQUIRE(estimate.has_value());
  CHECK(estimate->step_time == slab.start());
  CHECK(estimate->step_size == slab.duration());
  CHECK(estimate->order == 2);
  // The errors are measured relative to the value of the latest step.
  const auto expected_error = [](const DataVector& errors) {
    const DataVector values{5.0, 6.0};
    return max(abs(errors) /
               (1.0 + 0.5 * max(abs(values), abs(values + errors))));
  };
  CHECK(estimate->step_error() ==
        approx(expected_error(0.1 * DataVector{3.0, 4.0})));
  REQUIRE(estimate->errors[0].has_value());
  CHECK(*estimate->errors[0] ==
        approx(expected_error(-0.01 * DataVector{11.0, 12.0})));
  CHECK(not estimate->errors[1].has_value());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.HistoryCoefficients",
                  "[Unit][Time]") {
  test_history_times();
  test_linear_combination();
  test_step_coefficients();
}
