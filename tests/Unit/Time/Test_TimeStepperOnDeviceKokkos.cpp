// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <random>

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

// Evolves y' = a y pointwise with the host actions and the device
// actions side by side and compares the variables, the error
// estimates, and dense output.
void test_stepper(const TimeStepper& stepper,
                  const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto rates = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), DataVector(number_of_points));
  Vars vars(number_of_points);
  get(get<Var>(vars)) = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), DataVector(number_of_points));
  Vars device_vars = vars;
  const auto rhs = [&rates](const Vars& local_vars) {
    DtVars result(number_of_points);
    get(get<::Tags::dt<Var>>(result)) = rates * get(get<Var>(local_vars));
    return result;
  };

  // The order requiring no past steps, as used to start an evolution.
  const size_t order = get<TimeSteppers::Tags::FixedOrder>(stepper.order()) -
                       stepper.number_of_past_steps();
  TimeSteppers::History<Vars> history{order};
  TimeSteppers::DeviceHistory<> device_history{};
  Initialization::DeviceTimeStepperHistory<System>::apply(
      make_not_null(&device_history), history);
  CHECK(device_history.integration_order() == order);
  // An empty host history, as the device actions expect.
  const TimeSteppers::History<Vars> empty_host_history{order};

  const Slab slab(0.0, 1.0);
  const TimeDelta time_step = slab.duration() / 16;
  const StepperErrorTolerances tolerances{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 1.0e-8,
      .relative = 1.0e-6};
  std::array<std::optional<StepperErrorEstimate>, 2> errors{};
  std::array<std::optional<StepperErrorEstimate>, 2> device_errors{};

  size_t dense_outputs = 0;
  TimeStepId time_step_id(true, 0, slab.start());
  while (time_step_id.step_time() < slab.end()) {
    CAPTURE(time_step_id);
    const TimeStepId next_time_step_id =
        stepper.next_time_id_for_error(time_step_id, time_step);
    RecordTimeStepperData<System>::apply(make_not_null(&history), time_step_id,
                                         vars, rhs(vars));
    RecordTimeStepperDataOnDevice<System>::apply(
        make_not_null(&device_history), time_step_id, device_vars,
        rhs(device_vars), empty_host_history);

    // Dense output in the current and the previous step.
    for (const double fraction : {0.5, -0.5}) {
      const double dense_time =
          time_step_id.step_time().value() + fraction * time_step.value();
      if (dense_time < history.front().time_step_id.step_time().value() or
          not history.step_start(dense_time).value.has_value()) {
        continue;
      }
      Vars dense_vars = *history.step_start(dense_time).value;
      if (stepper.dense_update_u(make_not_null(&dense_vars), history,
                                 dense_time)) {
        Vars device_dense_vars(number_of_points);
        REQUIRE(TimeSteppers::dense_update_u_on_device(
            Kokkos::DefaultExecutionSpace{}, make_not_null(&device_dense_vars),
            make_not_null(&device_history), stepper, dense_time));
        CHECK_VARIABLES_APPROX(device_dense_vars, dense_vars);
        ++dense_outputs;
      }
    }

    UpdateU<System>::apply(make_not_null(&vars), make_not_null(&errors),
                           stepper, time_step_id, next_time_step_id, time_step,
                           true, history, tolerances);
    UpdateUOnDevice<System>::apply(
        make_not_null(&device_vars), make_not_null(&device_errors),
        make_not_null(&device_history), stepper, time_step_id,
        next_time_step_id, time_step, true, tolerances);
    CHECK_VARIABLES_APPROX(device_vars, vars);
    for (size_t i = 0; i < 2; ++i) {
      REQUIRE(gsl::at(device_errors, i).has_value() ==
              gsl::at(errors, i).has_value());
      if (gsl::at(errors, i).has_value()) {
        CHECK(gsl::at(device_errors, i)->step_time ==
              gsl::at(errors, i)->step_time);
        // The error estimate is a difference of nearly equal values,
        // so it amplifies the roundoff differences between the host
        // and device evolutions.
        CHECK(gsl::at(device_errors, i)->step_error() ==
              Approx::custom().epsilon(1.0e-10).margin(1.0e-6)(
                  gsl::at(errors, i)->step_error()));
      }
    }

    CleanHistory<System>::apply(make_not_null(&history), stepper);
    CleanHistoryOnDevice<System>::apply(make_not_null(&device_history),
                                        stepper);
    CHECK(device_history.times().steps ==
          TimeSteppers::HistoryTimes(history).steps);
    CHECK(device_history.times().substeps ==
          TimeSteppers::HistoryTimes(history).substeps);
    time_step_id = next_time_step_id;
  }
  CHECK(errors[1].has_value());
  CHECK(dense_outputs > 0);

  // Recording on the device with data in the host history is an error.
  TimeSteppers::History<Vars> nonempty_host_history{order};
  nonempty_host_history.insert(time_step_id, vars, rhs(vars));
  CHECK_THROWS_WITH(RecordTimeStepperDataOnDevice<System>::apply(
                        make_not_null(&device_history), time_step_id,
                        device_vars, rhs(device_vars), nonempty_host_history),
                    Catch::Matchers::ContainsSubstring(
                        "but the history is being recorded on the device"));
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
