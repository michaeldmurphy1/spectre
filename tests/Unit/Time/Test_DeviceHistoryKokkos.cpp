// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
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
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using DeviceHistory = TimeSteppers::DeviceHistory<MemorySpace>;
using DeviceBatch = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>;
constexpr size_t number_of_points = 17;

void copy_to_device(const TimeSteppers::HistoryBatchView<MemorySpace>& device,
                    const DataVector& host) {
  const Kokkos::View<const double**, Kokkos::LayoutStride, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>
      host_view(host.data(), device.layout());
  Kokkos::deep_copy(device, host_view);
}

DataVector to_host(
    const Kokkos::View<const double**, Kokkos::LayoutStride, MemorySpace,
                       Kokkos::MemoryUnmanaged>& device) {
  DataVector result(device.extent(0));
  const Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>
      host_view(result.data(), device.layout());
  Kokkos::deep_copy(host_view, device);
  return result;
}

// The device data of a history should match the host history.  The
// data only agrees to roundoff, because device kernels can round
// differently than the host.
void check_matches(const DeviceHistory& device_history,
                   const TimeSteppers::History<DataVector>& history) {
  const TimeSteppers::HistoryTimes times = device_history.times();
  const TimeSteppers::HistoryTimes expected_times(history);
  CHECK(times.integration_order == expected_times.integration_order);
  CHECK(times.steps == expected_times.steps);
  CHECK(times.substeps == expected_times.substeps);
  const auto views = device_history.device_view();
  const auto check_record = [](const auto& view, const auto& record) {
    CAPTURE(record.time_step_id);
    REQUIRE((view.value.data() != nullptr) == record.value.has_value());
    if (record.value.has_value()) {
      CHECK_ITERABLE_APPROX(to_host(view.value), *record.value);
    }
    CHECK_ITERABLE_APPROX(to_host(view.derivative), record.derivative);
  };
  REQUIRE(views.steps.size() == history.size());
  for (size_t i = 0; i < history.size(); ++i) {
    check_record(views.steps[i], history[i]);
  }
  REQUIRE(views.substeps.size() == history.substeps().size());
  for (size_t i = 0; i < history.substeps().size(); ++i) {
    check_record(views.substeps[i], history.substeps()[i]);
  }
}

// Integrates y' = a y pointwise with the host `History` and the
// `DeviceHistory` side by side, with the derivative computed in a
// kernel writing into the history, and compares the updates, error
// estimates, dense output, and stored data.
void test_integration(const TimeStepper& stepper, const size_t order,
                      const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto rates = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), DataVector(number_of_points));
  const auto initial_values = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), DataVector(number_of_points));
  const auto analytic = [&](const double t) {
    return DataVector(initial_values * exp(rates * t));
  };
  const auto rhs = [&](const DataVector& y, const double /*t*/) {
    return DataVector(rates * y);
  };
  const DeviceBatch device_rates("rates", number_of_points, 1);
  copy_to_device(device_rates, rates);

  const Slab slab(0.0, 1.0);
  const TimeDelta time_step = slab.duration() / 16;
  const StepperErrorTolerances tolerances{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 1.0e-8,
      .relative = 1.0e-6};

  TimeSteppers::History<DataVector> history{order};
  TimeStepperTestUtils::initialize_history(
      slab.start(), make_not_null(&history), analytic, rhs, time_step,
      stepper.number_of_past_steps());
  DeviceHistory device_history{order, number_of_points};
  // `insert` with `initial` adds records before the existing ones.
  for (size_t i = history.size(); i > 0; --i) {
    const auto& record = history[i - 1];
    const auto inserted =
        device_history.insert(record.time_step_id, true, true);
    copy_to_device(inserted.value, *record.value);
    copy_to_device(inserted.derivative, record.derivative);
  }
  check_matches(device_history, history);

  DataVector u = analytic(0.0);
  const DeviceBatch device_u("u", number_of_points, 1);
  copy_to_device(device_u, u);

  size_t dense_outputs = 0;
  size_t capacity_after_first_step = 0;
  TimeStepId time_step_id(true, 0, slab.start());
  while (time_step_id.step_time() < slab.end()) {
    CAPTURE(time_step_id);
    history.insert(time_step_id, u, rhs(u, time_step_id.substep_time()));
    {
      const auto inserted = device_history.insert(time_step_id, true);
      Kokkos::deep_copy(inserted.value, device_u);
      const auto derivative = inserted.derivative;
      Kokkos::parallel_for(
          "rhs", Kokkos::RangePolicy<>(0, number_of_points),
          KOKKOS_LAMBDA(const size_t i) {
            derivative(i, 0) = device_rates(i, 0) * device_u(i, 0);
          });
    }
    check_matches(device_history, history);

    // Dense output in the current and the previous step.  Steppers
    // using the derivative at the end of a step for dense output can
    // only produce it once the next step has started.
    for (const double fraction : {0.5, -0.5}) {
      const double dense_time =
          time_step_id.step_time().value() + fraction * time_step.value();
      if (dense_time < history.front().time_step_id.step_time().value() or
          not history.step_start(dense_time).value.has_value()) {
        continue;
      }
      DataVector dense_u = *history.step_start(dense_time).value;
      if (not stepper.dense_update_u(make_not_null(&dense_u), history,
                                     dense_time)) {
        continue;
      }
      const auto coefficients =
          stepper.dense_output_coefficients(device_history.times(), dense_time);
      REQUIRE(coefficients.has_value());
      const DeviceBatch device_dense_u("dense_u", number_of_points, 1);
      Kokkos::deep_copy(device_dense_u,
                        device_history.step_start_value(dense_time));
      TimeSteppers::apply_history_linear_combination_on_device(
          Kokkos::DefaultExecutionSpace{}, device_dense_u,
          device_history.device_view(), *coefficients);
      CHECK_ITERABLE_APPROX(to_host(device_dense_u), dense_u);
      ++dense_outputs;
    }

    const auto error =
        stepper.update_u(make_not_null(&u), history, time_step, tolerances);
    const auto device_errors = TimeSteppers::apply_step_coefficients_on_device(
        Kokkos::DefaultExecutionSpace{}, device_u, device_history.device_view(),
        stepper.step_coefficients(device_history.times(), time_step,
                                  tolerances),
        tolerances);
    CHECK_ITERABLE_APPROX(to_host(device_u), u);
    REQUIRE(device_errors.size() == 1);
    REQUIRE(device_errors[0].has_value() == error.has_value());
    if (error.has_value()) {
      // The error estimate is a difference of nearly equal values, so
      // it amplifies the roundoff differences between the host and
      // device evolutions.  Those are of order roundoff divided by the
      // absolute tolerance.
      CHECK(device_errors[0]->step_error() ==
            Approx::custom().epsilon(1.0e-10).margin(1.0e-6)(
                error->step_error()));
    }

    stepper.clean_history(make_not_null(&history));
    device_history.clean(stepper);
    check_matches(device_history, history);

    time_step_id = stepper.next_time_id_for_error(time_step_id, time_step);
    if (capacity_after_first_step == 0 and time_step_id.substep() == 0) {
      capacity_after_first_step = device_history.capacity();
    }
  }
  CHECK(dense_outputs > 0);
  // The evolution reaches a steady state, so the history allocates
  // no more slots after the first step.
  CHECK(device_history.capacity() == capacity_after_first_step);
  CHECK_ITERABLE_CUSTOM_APPROX(u, analytic(slab.end().value()),
                               Approx::custom().epsilon(1.0e-2).scale(1.0));

  {
    INFO("Serialization");
    const auto deserialized = serialize_and_deserialize(device_history);
    check_matches(deserialized, history);
    CHECK(deserialized.capacity() == device_history.capacity());
  }
}

void test_bookkeeping() {
  const Slab slab(0.0, 1.0);
  const TimeDelta time_step = slab.duration() / 4;
  DeviceHistory device_history{2, number_of_points};
  CHECK(device_history.empty());
  CHECK(device_history.capacity() == 0);
  CHECK(device_history.number_of_values() == number_of_points);
  CHECK(device_history.integration_order() == 2);
  device_history.integration_order(3);
  CHECK(device_history.integration_order() == 3);

  const TimeStepId first(true, 0, slab.start());
  const TimeStepId second = first.next_step(time_step);
  const auto fill = [](const auto& view, const double fill_value) {
    Kokkos::deep_copy(view, fill_value);
  };
  {
    const auto record = device_history.insert(first, true);
    fill(record.value, 1.0);
    fill(record.derivative, 2.0);
  }
  {
    const auto record = device_history.insert(second, false);
    CHECK(record.value.data() == nullptr);
    fill(record.derivative, 3.0);
  }
  {
    const auto record =
        device_history.insert(second.next_substep(time_step, 1.0), true);
    fill(record.value, 4.0);
    fill(record.derivative, 5.0);
  }
  // The allocation has grown, keeping the data of the existing records.
  CHECK(device_history.capacity() >= 5);
  CHECK(device_history.size() == 2);
  CHECK(device_history.number_of_substeps() == 1);
  CHECK(not device_history.at_step_start());
  CHECK(device_history.step_id(1) == second);
  {
    const auto views = device_history.device_view();
    CHECK(to_host(views.steps[0].value) == DataVector(number_of_points, 1.0));
    CHECK(to_host(views.steps[0].derivative) ==
          DataVector(number_of_points, 2.0));
    CHECK(views.steps[1].value.data() == nullptr);
    CHECK(to_host(views.steps[1].derivative) ==
          DataVector(number_of_points, 3.0));
    CHECK(to_host(views.substeps[0].value) ==
          DataVector(number_of_points, 4.0));
    CHECK(to_host(views.substeps[0].derivative) ==
          DataVector(number_of_points, 5.0));
  }
  CHECK(to_host(device_history.step_start_value(0.1)) ==
        DataVector(number_of_points, 1.0));

  // Removed records free their slots for reuse.
  const size_t capacity = device_history.capacity();
  device_history.discard_value(first);
  CHECK(device_history.device_view().steps[0].value.data() == nullptr);
  device_history.clear_substeps();
  CHECK(device_history.number_of_substeps() == 0);
  device_history.pop_front();
  CHECK(device_history.size() == 1);
  for (size_t i = 0; i < 3; ++i) {
    device_history.insert(second.next_substep(time_step, 1.0), true);
    device_history.clear_substeps();
  }
  CHECK(device_history.capacity() == capacity);

  // The bookkeeping has the same times and can be modified.
  CHECK(device_history.bookkeeping().size() == 1);
  CHECK(device_history.bookkeeping().back().time_step_id == second);
  const TimeStepId renamed(true, 1, second.step_time());
  device_history.bookkeeping().back().time_step_id = renamed;
  CHECK(device_history.step_id(0) == renamed);
  CHECK(to_host(device_history.device_view().steps[0].derivative) ==
        DataVector(number_of_points, 3.0));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      device_history.set_number_of_values(number_of_points + 1),
      Catch::Matchers::ContainsSubstring(
          "Cannot change the number of values of a non-empty DeviceHistory"));
#endif  // SPECTRE_DEBUG
  device_history.clear();
  CHECK(device_history.empty());
  device_history.set_number_of_values(number_of_points + 1);
  CHECK(device_history.number_of_values() == number_of_points + 1);
  CHECK(device_history.capacity() == 0);
  CHECK(device_history.insert(first, true).value.extent(0) ==
        number_of_points + 1);
}

void test_copies_and_scratch() {
  const Kokkos::DefaultExecutionSpace exec{};
  DeviceHistory device_history{1, number_of_points};
  const auto scratch = device_history.scratch();
  CHECK(scratch.extent(0) == number_of_points);
  CHECK(scratch.extent(1) == 1);
  // The scratch storage is reused.
  CHECK(device_history.scratch().data() == scratch.data());

  const DataVector host_data = [] {
    DataVector result(number_of_points);
    for (size_t i = 0; i < number_of_points; ++i) {
      result[i] = static_cast<double>(i) + 0.5;
    }
    return result;
  }();
  TimeSteppers::copy_to_device(
      exec, scratch,
      gsl::span<const double>(host_data.data(), host_data.size()));
  DataVector copied(number_of_points, 0.0);
  TimeSteppers::copy_to_host(
      exec, gsl::span<double>(copied.data(), copied.size()), scratch);
  CHECK(copied == host_data);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.DeviceHistory.Kokkos", "[Unit][Time]") {
  MAKE_GENERATOR(generator);
  test_bookkeeping();
  test_copies_and_scratch();
  {
    INFO("AdamsBashforth");
    test_integration(TimeSteppers::AdamsBashforth(3), 3,
                     make_not_null(&generator));
  }
  {
    INFO("AdamsMoultonPc");
    test_integration(TimeSteppers::AdamsMoultonPc<false>(4), 4,
                     make_not_null(&generator));
  }
  {
    INFO("AdamsMoultonPcMonotonic");
    test_integration(TimeSteppers::AdamsMoultonPc<true>(3), 3,
                     make_not_null(&generator));
  }
  {
    INFO("ClassicalRungeKutta4");
    test_integration(TimeSteppers::ClassicalRungeKutta4{}, 4,
                     make_not_null(&generator));
  }
  {
    INFO("DormandPrince5");
    test_integration(TimeSteppers::DormandPrince5{}, 5,
                     make_not_null(&generator));
  }
  {
    INFO("Heun2");
    test_integration(TimeSteppers::Heun2{}, 2, make_not_null(&generator));
  }
  {
    INFO("Rk3HesthavenSsp");
    test_integration(TimeSteppers::Rk3HesthavenSsp{}, 3,
                     make_not_null(&generator));
  }
}
