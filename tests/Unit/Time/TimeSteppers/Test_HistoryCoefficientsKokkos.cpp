// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Time/TimeSteppers/HistoryCoefficients.hpp"
#include "Time/TimeSteppers/HistoryCoefficientsKokkos.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

namespace {
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using DeviceBatch = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>;
using Histories = std::vector<TimeSteppers::History<DataVector>>;
constexpr size_t number_of_points = 23;

// Copies the data of the elements of a batch to the device, indexed
// `(i, e)`.
DeviceBatch to_device(const std::vector<DataVector>& host) {
  DeviceBatch result("batch", number_of_points, host.size());
  const auto host_mirror = Kokkos::create_mirror_view(result);
  for (size_t e = 0; e < host.size(); ++e) {
    for (size_t i = 0; i < number_of_points; ++i) {
      host_mirror(i, e) = host[e][i];
    }
  }
  Kokkos::deep_copy(result, host_mirror);
  return result;
}

std::vector<DataVector> to_host(const DeviceBatch& device) {
  const auto host_mirror =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device);
  std::vector<DataVector> result(device.extent(1),
                                 DataVector(number_of_points));
  for (size_t e = 0; e < result.size(); ++e) {
    for (size_t i = 0; i < number_of_points; ++i) {
      result[e][i] = host_mirror(i, e);
    }
  }
  return result;
}

// Mirrors the host histories of a batch of elements on the device.
// The device data of a history would normally stay on the device;
// here it is copied from the host so the device results can be
// compared with the host ones.
class MirroredHistory {
 public:
  explicit MirroredHistory(const Histories& histories) {
    const auto& first = histories.front();
    for (size_t step = 0; step < first.size(); ++step) {
      device_history_.steps.push_back(mirror(
          histories,
          [&](const auto& history) -> const auto& { return history[step]; }));
    }
    for (size_t substep = 0; substep < first.substeps().size(); ++substep) {
      device_history_.substeps.push_back(
          mirror(histories, [&](const auto& history) -> const auto& {
            return history.substeps()[substep];
          }));
    }
  }

  const TimeSteppers::DeviceHistoryView<MemorySpace>& device_history() const {
    return device_history_;
  }

 private:
  template <typename GetRecord>
  TimeSteppers::DeviceHistoryRecordView<MemorySpace> mirror(
      const Histories& histories, const GetRecord& get_record) {
    TimeSteppers::DeviceHistoryRecordView<MemorySpace> result{};
    std::vector<DataVector> values{};
    std::vector<DataVector> derivatives{};
    for (const auto& history : histories) {
      const auto& record = get_record(history);
      CHECK(record.time_step_id == get_record(histories.front()).time_step_id);
      if (record.value.has_value()) {
        values.push_back(*record.value);
      }
      derivatives.push_back(record.derivative);
    }
    if (not values.empty()) {
      REQUIRE(values.size() == histories.size());
      result.value = storage_.emplace_back(to_device(values));
    }
    result.derivative = storage_.emplace_back(to_device(derivatives));
    return result;
  }

  std::deque<DeviceBatch> storage_{};
  TimeSteppers::DeviceHistoryView<MemorySpace> device_history_{};
};

// Applies the step coefficients on the device and checks the result
// against `update_u` on the host for each element.
void check_step(const TimeStepper& stepper, const Histories& histories,
                const TimeDelta& time_step,
                const StepperErrorTolerances& tolerances) {
  const MirroredHistory mirrored(histories);
  const DeviceBatch u("u", number_of_points, histories.size());
  const auto errors = TimeSteppers::apply_step_coefficients_on_device(
      Kokkos::DefaultExecutionSpace{}, u, mirrored.device_history(),
      stepper.step_coefficients(TimeSteppers::HistoryTimes(histories.front()),
                                time_step, tolerances),
      tolerances);
  const auto device_u = to_host(u);
  REQUIRE(errors.size() == histories.size());

  for (size_t e = 0; e < histories.size(); ++e) {
    CAPTURE(e);
    DataVector expected_u(number_of_points);
    const auto expected_error = stepper.update_u(
        make_not_null(&expected_u), histories[e], time_step, tolerances);
    CHECK_ITERABLE_APPROX(device_u[e], expected_u);
    REQUIRE(errors[e].has_value() == expected_error.has_value());
    if (errors[e].has_value()) {
      CHECK(errors[e]->step_time == expected_error->step_time);
      CHECK(errors[e]->step_size == expected_error->step_size);
      CHECK(errors[e]->order == expected_error->order);
      for (size_t order = 0; order < errors[e]->errors.size(); ++order) {
        CAPTURE(order);
        REQUIRE(gsl::at(errors[e]->errors, order).has_value() ==
                gsl::at(expected_error->errors, order).has_value());
        if (gsl::at(errors[e]->errors, order).has_value()) {
          CHECK(*gsl::at(errors[e]->errors, order) ==
                approx(*gsl::at(expected_error->errors, order)));
        }
      }
    }
  }
}

// Applies the dense-output coefficients on the device and checks the
// result against `dense_update_u` on the host for each element.
// Returns whether dense output was possible.
bool check_dense_output(const TimeStepper& stepper, const Histories& histories,
                        const double time) {
  const auto coefficients = stepper.dense_output_coefficients(
      TimeSteppers::HistoryTimes(histories.front()), time);
  std::vector<DataVector> initial_u{};
  std::vector<DataVector> expected_u{};
  for (const auto& history : histories) {
    const auto& step_start = history.step_start(time);
    if (not step_start.value.has_value()) {
      return false;
    }
    initial_u.push_back(*step_start.value);
    expected_u.push_back(*step_start.value);
    const bool success = stepper.dense_update_u(
        make_not_null(&expected_u.back()), history, time);
    REQUIRE(success == coefficients.has_value());
  }
  if (not coefficients.has_value()) {
    return false;
  }

  const MirroredHistory mirrored(histories);
  const DeviceBatch u = to_device(initial_u);
  TimeSteppers::apply_history_linear_combination_on_device(
      Kokkos::DefaultExecutionSpace{}, u, mirrored.device_history(),
      *coefficients);
  const auto device_u = to_host(u);
  for (size_t e = 0; e < histories.size(); ++e) {
    CAPTURE(e);
    CHECK_ITERABLE_APPROX(device_u[e], expected_u[e]);
  }
  return true;
}

// Applies the implicit coefficients on the device and checks the
// result against `add_inhomogeneous_implicit_terms` on the host for
// each element.
void check_implicit_terms(const ImexTimeStepper& stepper,
                          const Histories& implicit_histories,
                          const TimeDelta& time_step,
                          const std::vector<DataVector>& explicit_u) {
  const auto coefficients = stepper.implicit_coefficients(
      TimeSteppers::HistoryTimes(implicit_histories.front()), time_step);
  const MirroredHistory mirrored(implicit_histories);
  const DeviceBatch u = to_device(explicit_u);
  TimeSteppers::apply_history_linear_combination_on_device(
      Kokkos::DefaultExecutionSpace{}, u, mirrored.device_history(),
      coefficients.inhomogeneous);
  const auto device_u = to_host(u);

  for (size_t e = 0; e < implicit_histories.size(); ++e) {
    CAPTURE(e);
    DataVector expected_u = explicit_u[e];
    stepper.add_inhomogeneous_implicit_terms(make_not_null(&expected_u),
                                             implicit_histories[e], time_step);
    CHECK_ITERABLE_APPROX(device_u[e], expected_u);
    CHECK(coefficients.weight ==
          stepper.implicit_weight(implicit_histories[e], time_step));
  }
}

// Takes a few steps with random history data for a batch of elements,
// checking every update, error estimate, dense output, and implicit
// update on the device.
void test_stepper(const TimeStepper& stepper, const size_t order,
                  const size_t number_of_elements,
                  const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto random_data = [&]() {
    return make_with_random_values<DataVector>(generator, make_not_null(&dist),
                                               DataVector(number_of_points));
  };
  const auto random_batch = [&]() {
    std::vector<DataVector> result(number_of_elements);
    for (auto& data : result) {
      data = random_data();
    }
    return result;
  };
  const auto* const imex_stepper =
      dynamic_cast<const ImexTimeStepper*>(&stepper);

  const Slab slab(0.0, 1.0);
  const TimeDelta time_step = slab.duration() / 8;
  const StepperErrorTolerances tolerances{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 1.0e-3,
      .relative = 1.0e-2};

  Histories histories(number_of_elements,
                      TimeSteppers::History<DataVector>{order});
  Histories implicit_histories(number_of_elements,
                               TimeSteppers::History<DataVector>{order});
  for (auto& history : histories) {
    TimeStepperTestUtils::initialize_history(
        slab.start(), make_not_null(&history),
        [&](const double /*t*/) { return random_data(); },
        [&](const DataVector& /*y*/, const double /*t*/) {
          return random_data();
        },
        time_step, stepper.number_of_past_steps());
  }

  size_t dense_outputs = 0;
  TimeStepId time_step_id(true, 0, slab.start());
  while (time_step_id.step_time() < slab.start() + 4 * time_step) {
    CAPTURE(time_step_id);
    for (auto& history : histories) {
      history.insert(time_step_id, random_data(), random_data());
    }
    if (imex_stepper != nullptr) {
      for (auto& implicit_history : implicit_histories) {
        implicit_history.insert(time_step_id,
                                TimeSteppers::History<DataVector>::no_value,
                                random_data());
      }
    }

    for (const double fraction : {0.3, -0.4}) {
      const double time =
          time_step_id.step_time().value() + fraction * time_step.value();
      if (time >= histories.front().front().time_step_id.step_time().value()) {
        dense_outputs += check_dense_output(stepper, histories, time) ? 1 : 0;
      }
    }

    check_step(stepper, histories, time_step, tolerances);
    if (imex_stepper != nullptr) {
      check_implicit_terms(*imex_stepper, implicit_histories, time_step,
                           random_batch());
    }

    for (auto& history : histories) {
      stepper.clean_history(make_not_null(&history));
    }
    if (imex_stepper != nullptr) {
      for (auto& implicit_history : implicit_histories) {
        stepper.clean_history(make_not_null(&implicit_history));
      }
    }
    time_step_id = stepper.next_time_id_for_error(time_step_id, time_step);
  }
  CHECK(dense_outputs > 0);
}

// Checks the estimates at all orders of variable-order steppers, as
// used for variable-order h-refinement.
void test_all_orders(const TimeStepper& stepper, const size_t order,
                     const bool corrector,
                     const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto random_data = [&]() {
    return make_with_random_values<DataVector>(generator, make_not_null(&dist),
                                               DataVector(number_of_points));
  };
  const Slab slab(0.0, 1.0);
  const TimeDelta time_step = slab.duration() / 8;
  const TimeStepId step_id(true, 0, slab.start());
  // Adams-Bashforth uses `order` steps.  The Adams-Moulton corrector
  // uses `order - 1` steps and the predictor substep.
  const size_t number_of_steps = corrector ? order - 1 : order;
  Histories histories(3, TimeSteppers::History<DataVector>{order});
  for (auto& history : histories) {
    TimeStepperTestUtils::initialize_history(
        slab.start(), make_not_null(&history),
        [&](const double /*t*/) { return random_data(); },
        [&](const DataVector& /*y*/, const double /*t*/) {
          return random_data();
        },
        time_step, number_of_steps - 1);
    history.insert(step_id, random_data(), random_data());
    if (corrector) {
      history.insert(step_id.next_substep(time_step, 1.0), random_data(),
                     random_data());
    }
  }
  const StepperErrorTolerances tolerances{
      .estimates = StepperErrorTolerances::Estimates::AllOrders,
      .absolute = 1.0e-3,
      .relative = 1.0e-2};
  check_step(stepper, histories, time_step, tolerances);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.HistoryCoefficients.Kokkos",
                  "[Unit][Time]") {
  MAKE_GENERATOR(generator);
  // A single element is a batch of one.
  for (const size_t number_of_elements : {1_st, 3_st}) {
    CAPTURE(number_of_elements);
    {
      INFO("AdamsBashforth");
      test_stepper(TimeSteppers::AdamsBashforth(3), 3, number_of_elements,
                   make_not_null(&generator));
    }
    {
      INFO("AdamsMoultonPc");
      test_stepper(TimeSteppers::AdamsMoultonPc<false>(4), 4,
                   number_of_elements, make_not_null(&generator));
    }
    {
      INFO("AdamsMoultonPcMonotonic");
      test_stepper(TimeSteppers::AdamsMoultonPc<true>(3), 3, number_of_elements,
                   make_not_null(&generator));
    }
    {
      INFO("ClassicalRungeKutta4");
      test_stepper(TimeSteppers::ClassicalRungeKutta4{}, 4, number_of_elements,
                   make_not_null(&generator));
    }
    {
      INFO("DormandPrince5");
      test_stepper(TimeSteppers::DormandPrince5{}, 5, number_of_elements,
                   make_not_null(&generator));
    }
    {
      INFO("Heun2");
      test_stepper(TimeSteppers::Heun2{}, 2, number_of_elements,
                   make_not_null(&generator));
    }
    {
      INFO("Rk3HesthavenSsp");
      test_stepper(TimeSteppers::Rk3HesthavenSsp{}, 3, number_of_elements,
                   make_not_null(&generator));
    }
  }
  {
    INFO("AdamsBashforth all orders");
    test_all_orders(TimeSteppers::AdamsBashforth(std::nullopt), 4, false,
                    make_not_null(&generator));
  }
  {
    INFO("AdamsMoultonPc all orders");
    test_all_orders(TimeSteppers::AdamsMoultonPc<false>(std::nullopt), 4, true,
                    make_not_null(&generator));
  }
}
