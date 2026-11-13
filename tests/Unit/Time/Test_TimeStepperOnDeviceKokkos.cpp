// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Time/CleanHistory.tpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/History.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/Slab.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepperOnDeviceKokkos.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/UpdateU.tpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using variables_tag = ::Tags::Variables<tmpl::list<Var>>;
};

using Vars = Variables<tmpl::list<Var>>;
using DtVars = Variables<tmpl::list<::Tags::dt<Var>>>;
constexpr size_t number_of_points = 11;
constexpr size_t number_of_elements = 3;

template <typename TagsList>
void to_device(const gsl::not_null<BatchedVariables<TagsList>*> batch,
               const std::vector<Variables<TagsList>>& elements) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{number_of_points,
                                                           elements.size()};
  for (size_t e = 0; e < elements.size(); ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    element = elements[e];
  }
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, batch, host_batch);
  exec.fence();
}

template <typename TagsList>
std::vector<Variables<TagsList>> to_host(
    const BatchedVariables<TagsList>& batch) {
  BatchedVariables<TagsList, Kokkos::HostSpace> host_batch{
      number_of_points, batch.number_of_elements()};
  const Kokkos::DefaultExecutionSpace exec{};
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();
  std::vector<Variables<TagsList>> result{};
  for (size_t e = 0; e < batch.number_of_elements(); ++e) {
    Variables<TagsList> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    result.push_back(element);
  }
  return result;
}

// Evolves y' = a y pointwise for several elements, each with the host
// mutators, and all as a batch with the device functions, and compares the
// variables, the error estimates of each element, and dense output.
void test_stepper(const TimeStepper& stepper,
                  const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<DataVector> rates{};
  std::vector<Vars> vars{};
  for (size_t e = 0; e < number_of_elements; ++e) {
    rates.push_back(make_with_random_values<DataVector>(
        generator, make_not_null(&dist), DataVector(number_of_points)));
    vars.emplace_back(number_of_points);
    get(get<Var>(vars.back())) = make_with_random_values<DataVector>(
        generator, make_not_null(&dist), DataVector(number_of_points));
  }
  const auto rhs = [&rates](const Vars& local_vars, const size_t e) {
    DtVars result(number_of_points);
    get(get<::Tags::dt<Var>>(result)) = rates[e] * get(get<Var>(local_vars));
    return result;
  };
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<tmpl::list<Var>> device_vars{exec, number_of_points,
                                                number_of_elements};
  to_device(make_not_null(&device_vars), vars);
  BatchedVariables<tmpl::list<::Tags::dt<Var>>> device_dt_vars{
      exec, number_of_points, number_of_elements};

  // The order requiring no past steps, as used to start an evolution.
  const size_t order = get<TimeSteppers::Tags::FixedOrder>(stepper.order()) -
                       stepper.number_of_past_steps();
  std::vector<TimeSteppers::History<Vars>> histories(
      number_of_elements, TimeSteppers::History<Vars>{order});
  TimeSteppers::DeviceHistory<> device_history{order, number_of_points,
                                               number_of_elements};

  const Slab slab(0.0, 1.0);
  const TimeDelta time_step = slab.duration() / 16;
  const StepperErrorTolerances tolerances{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 1.0e-8,
      .relative = 1.0e-6};
  std::vector<std::array<std::optional<StepperErrorEstimate>, 2>> errors(
      number_of_elements);
  std::vector<std::array<std::optional<StepperErrorEstimate>, 2>>
      device_errors{};

  size_t dense_outputs = 0;
  TimeStepId time_step_id(true, 0, slab.start());
  while (time_step_id.step_time() < slab.end()) {
    CAPTURE(time_step_id);
    const TimeStepId next_time_step_id =
        stepper.next_time_id_for_error(time_step_id, time_step);
    std::vector<DtVars> dt_vars{};
    for (size_t e = 0; e < number_of_elements; ++e) {
      dt_vars.push_back(rhs(vars[e], e));
      RecordTimeStepperData<System>::apply(make_not_null(&histories[e]),
                                           time_step_id, vars[e], dt_vars[e]);
    }
    // The device variables are the host variables, so they are the same at
    // the start of each step
    to_device(make_not_null(&device_dt_vars), dt_vars);
    TimeSteppers::record_on_device(exec, make_not_null(&device_history),
                                   time_step_id, device_vars, device_dt_vars);

    // Dense output in the current and the previous step.
    for (const double fraction : {0.5, -0.5}) {
      const double dense_time =
          time_step_id.step_time().value() + fraction * time_step.value();
      const auto& history = histories.front();
      if (dense_time < history.front().time_step_id.step_time().value() or
          not history.step_start(dense_time).value.has_value()) {
        continue;
      }
      std::vector<Vars> dense_vars{};
      bool dense_output_possible = true;
      for (size_t e = 0; e < number_of_elements; ++e) {
        dense_vars.push_back(*histories[e].step_start(dense_time).value);
        dense_output_possible = stepper.dense_update_u(
            make_not_null(&dense_vars.back()), histories[e], dense_time);
      }
      BatchedVariables<tmpl::list<Var>> device_dense_vars{
          exec, number_of_points, number_of_elements};
      REQUIRE(TimeSteppers::dense_update_u_on_device(
                  exec, make_not_null(&device_dense_vars),
                  make_not_null(&device_history), stepper,
                  dense_time) == dense_output_possible);
      if (dense_output_possible) {
        const auto device_dense = to_host(device_dense_vars);
        for (size_t e = 0; e < number_of_elements; ++e) {
          CHECK_VARIABLES_APPROX(device_dense[e], dense_vars[e]);
        }
        ++dense_outputs;
      }
    }

    for (size_t e = 0; e < number_of_elements; ++e) {
      UpdateU<System>::apply(make_not_null(&vars[e]), make_not_null(&errors[e]),
                             stepper, time_step_id, next_time_step_id,
                             time_step, true, histories[e], tolerances);
    }
    TimeSteppers::update_u_on_device(
        exec, make_not_null(&device_vars), make_not_null(&device_errors),
        make_not_null(&device_history), stepper, time_step_id,
        next_time_step_id, time_step, true, tolerances);
    const auto updated = to_host(device_vars);
    REQUIRE(device_errors.size() == number_of_elements);
    for (size_t e = 0; e < number_of_elements; ++e) {
      CAPTURE(e);
      CHECK_VARIABLES_APPROX(updated[e], vars[e]);
      for (size_t i = 0; i < 2; ++i) {
        REQUIRE(gsl::at(device_errors[e], i).has_value() ==
                gsl::at(errors[e], i).has_value());
        if (gsl::at(errors[e], i).has_value()) {
          CHECK(gsl::at(device_errors[e], i)->step_time ==
                gsl::at(errors[e], i)->step_time);
          // The error estimate is a difference of nearly equal values,
          // so it amplifies the roundoff differences between the host
          // and device evolutions.
          CHECK(gsl::at(device_errors[e], i)->step_error() ==
                Approx::custom().epsilon(1.0e-10).margin(1.0e-6)(
                    gsl::at(errors[e], i)->step_error()));
        }
      }
    }
    // Continue from the host variables, so the evolutions don't drift apart
    to_device(make_not_null(&device_vars), vars);

    for (size_t e = 0; e < number_of_elements; ++e) {
      CleanHistory<System>::apply(make_not_null(&histories[e]), stepper);
    }
    device_history.clean(stepper);
    CHECK(device_history.times().steps ==
          TimeSteppers::HistoryTimes(histories.front()).steps);
    CHECK(device_history.times().substeps ==
          TimeSteppers::HistoryTimes(histories.front()).substeps);
    time_step_id = next_time_step_id;
  }
  CHECK(errors.front()[1].has_value());
  CHECK(dense_outputs > 0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeStepperOnDevice.Kokkos", "[Unit][Time]") {
  MAKE_GENERATOR(generator);
  {
    INFO("Rk3HesthavenSsp");
    test_stepper(TimeSteppers::Rk3HesthavenSsp{}, make_not_null(&generator));
  }
  {
    INFO("DormandPrince5");
    test_stepper(TimeSteppers::DormandPrince5{}, make_not_null(&generator));
  }
  {
    INFO("AdamsMoultonPc");
    test_stepper(TimeSteppers::AdamsMoultonPc<false>(3),
                 make_not_null(&generator));
  }
}
