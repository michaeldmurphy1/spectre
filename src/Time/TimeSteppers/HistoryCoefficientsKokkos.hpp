// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <boost/container/static_vector.hpp>
#include <cmath>
#include <cstddef>
#include <optional>
#include <tuple>
#include <vector>

#include "Time/History.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeSteppers/HistoryCoefficients.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/DeviceParametersKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * \brief Device data of a batch of elements, indexed `(i, e)`.
 *
 * `i` runs over all components at all grid points of an element, e.g.,
 * the storage of a device `Variables`, and `e` over the elements of
 * the batch.  A single element is a batch of one.
 */
template <typename MemorySpace =
              typename Kokkos::DefaultExecutionSpace::memory_space>
using HistoryBatchView = Kokkos::View<double**, Kokkos::LayoutStride,
                                      MemorySpace, Kokkos::MemoryUnmanaged>;

/// \ingroup TimeSteppersGroup
/// The device data of one record of the time-stepper histories of a
/// batch of elements, see `HistoryBatchView`.  `value` is empty if the
/// value has been discarded (see `History::discard_value`).
template <typename MemorySpace =
              typename Kokkos::DefaultExecutionSpace::memory_space>
struct DeviceHistoryRecordView {
  Kokkos::View<const double**, Kokkos::LayoutStride, MemorySpace,
               Kokkos::MemoryUnmanaged>
      value{};
  Kokkos::View<const double**, Kokkos::LayoutStride, MemorySpace,
               Kokkos::MemoryUnmanaged>
      derivative{};
};

/*!
 * \ingroup TimeSteppersGroup
 * \brief Views of the device data of the time-stepper histories of a
 * batch of elements, indexed like the `HistoryTimes` the coefficients
 * were computed from.
 *
 * All elements of the batch must have histories at the same times, so
 * they share the same coefficients.  This is the case for global time
 * stepping.
 *
 * This is a non-owning view.  It doesn't prescribe how the history
 * data is stored; `DeviceHistory` stores it for a batch of elements.
 */
template <typename MemorySpace =
              typename Kokkos::DefaultExecutionSpace::memory_space>
struct DeviceHistoryView {
  boost::container::static_vector<DeviceHistoryRecordView<MemorySpace>,
                                  history_max_past_steps + 2>
      steps{};
  boost::container::static_vector<DeviceHistoryRecordView<MemorySpace>,
                                  history_max_substeps>
      substeps{};
};

namespace HistoryCoefficientsKokkos_detail {
// The number of orders of `ErrorCoefficients::errors`
constexpr size_t max_error_orders =
    std::tuple_size_v<decltype(ErrorCoefficients::errors)>;

// The device data of a history record as a pointer and strides, so that it
// can be passed to kernels in device memory (see `DeviceParameters`).
struct DeviceRecord {
  const double* data = nullptr;
  size_t value_stride = 0;
  size_t element_stride = 0;

  KOKKOS_FUNCTION double operator()(const size_t i, const size_t e) const {
    return data[i * value_stride + e * element_stride];
  }
};

// A `HistoryLinearCombination` with the history records resolved to device
// data.
struct ResolvedCombination {
  Kokkos::Array<DeviceRecord, max_history_terms> data{};
  Kokkos::Array<double, max_history_terms> coefficients{};
  size_t size = 0;
  bool accumulate = false;

  // Applies the terms in the same order as
  // `apply_history_linear_combination` on the host.  `current` is the
  // existing value of the result, used only if `accumulate` is true.
  KOKKOS_FUNCTION double evaluate(const size_t i, const size_t e,
                                  const double current) const {
    double result = accumulate ? current : coefficients[0] * data[0](i, e);
    for (size_t term = accumulate ? 0 : 1; term < size; ++term) {
      result += coefficients[term] * data[term](i, e);
    }
    return result;
  }
};

// The error measure of `largest_stepper_error` at one point.
struct ScaledError {
  DeviceRecord reference_value{};
  double absolute = 0.0;
  double relative = 0.0;

  KOKKOS_FUNCTION double operator()(const size_t i, const size_t e,
                                    const double error_value) const {
    const double value = reference_value(i, e);
    return std::abs(error_value) /
           (absolute + relative * std::max(std::abs(value),
                                           std::abs(value + error_value)));
  }
};

// Everything `apply_step_coefficients_on_device` needs on the device: the
// update and, with error estimates, the estimate at the stepper order and at
// the lower orders.
struct StepParameters {
  ResolvedCombination update{};
  ResolvedCombination error{};
  Kokkos::Array<ResolvedCombination, max_error_orders> lower_order_errors{};
  size_t number_of_lower_orders = 0;
  ScaledError scaled_error{};
};

template <typename MemorySpace>
DeviceRecord record_data(const DeviceHistoryView<MemorySpace>& history,
                         const HistoryTerm::Record record, const size_t index,
                         const HistoryTerm::Field field,
                         const size_t number_of_values,
                         const size_t number_of_elements) {
  const DeviceHistoryRecordView<MemorySpace>* const records =
      record == HistoryTerm::Record::Step ? history.steps.data()
                                          : history.substeps.data();
  const size_t number_of_records = record == HistoryTerm::Record::Step
                                       ? history.steps.size()
                                       : history.substeps.size();
  if (index >= number_of_records) {
    ERROR("The coefficients reference "
          << (record == HistoryTerm::Record::Step ? "step " : "substep ")
          << index << ", but the device history only has " << number_of_records
          << ".");
  }
  const auto& view = field == HistoryTerm::Field::Value
                         ? records[index].value
                         : records[index].derivative;
  if (view.data() == nullptr) {
    ERROR("The coefficients reference the "
          << (field == HistoryTerm::Field::Value ? "value" : "derivative")
          << " of "
          << (record == HistoryTerm::Record::Step ? "step " : "substep ")
          << index << ", which is not on the device (it may have been "
          << "discarded).");
  }
  if (view.extent(0) != number_of_values or
      view.extent(1) != number_of_elements) {
    ERROR("History data has extents ("
          << view.extent(0) << ", " << view.extent(1)
          << "), but the result has (" << number_of_values << ", "
          << number_of_elements << ").");
  }
  return DeviceRecord{.data = view.data(),
                      .value_stride = view.stride(0),
                      .element_stride = view.stride(1)};
}

template <typename MemorySpace>
ResolvedCombination resolve(const DeviceHistoryView<MemorySpace>& history,
                            const HistoryLinearCombination& combination,
                            const size_t number_of_values,
                            const size_t number_of_elements) {
  ASSERT(combination.accumulate or combination.size > 0,
         "Cannot overwrite the result with an empty linear combination.");
  ResolvedCombination result{};
  result.size = combination.size;
  result.accumulate = combination.accumulate;
  for (size_t i = 0; i < combination.size; ++i) {
    const HistoryTerm& term = gsl::at(combination.terms, i);
    result.data[i] = record_data(history, term.record, term.index, term.field,
                                 number_of_values, number_of_elements);
    result.coefficients[i] = term.coefficient;
  }
  return result;
}
}  // namespace HistoryCoefficientsKokkos_detail

/// \ingroup TimeSteppersGroup
/// The device buffer through which `apply_history_linear_combination_on_device`
/// passes the coefficients to its kernel (see `DeviceParameters`).
template <typename MemorySpace =
              typename Kokkos::DefaultExecutionSpace::memory_space>
using HistoryCombinationParameters =
    DeviceParameters<HistoryCoefficientsKokkos_detail::ResolvedCombination,
                     MemorySpace>;

/// \ingroup TimeSteppersGroup
/// The device buffer through which `apply_step_coefficients_on_device` passes
/// the coefficients to its kernel (see `DeviceParameters`).
template <typename MemorySpace =
              typename Kokkos::DefaultExecutionSpace::memory_space>
using StepCoefficientsParameters =
    DeviceParameters<HistoryCoefficientsKokkos_detail::StepParameters,
                     MemorySpace>;

/// \ingroup TimeSteppersGroup
/// Apply \p combination to the data in \p history on the device for a
/// batch of elements, storing the result in \p result.  This is the
/// device version of `apply_history_linear_combination`, e.g., for
/// dense output or the inhomogeneous terms of an implicit substep.
///
/// The coefficients and the locations of the history data are passed to
/// the kernel through \p parameters, so the kernel's arguments don't
/// change between calls.  See `DeviceParameters` for when \p parameters
/// can be reused.
template <typename ExecutionSpace>
void apply_history_linear_combination_on_device(
    const ExecutionSpace& exec,
    const HistoryBatchView<typename ExecutionSpace::memory_space>& result,
    const DeviceHistoryView<typename ExecutionSpace::memory_space>& history,
    const HistoryLinearCombination& combination,
    const gsl::not_null<
        HistoryCombinationParameters<typename ExecutionSpace::memory_space>*>
        parameters) {
  const size_t number_of_values = result.extent(0);
  const size_t number_of_elements = result.extent(1);
  parameters->set(
      exec, HistoryCoefficientsKokkos_detail::resolve(
                history, combination, number_of_values, number_of_elements));
  const auto resolved = parameters->device();
  Kokkos::parallel_for(
      "apply_history_linear_combination_on_device",
      Kokkos::RangePolicy<ExecutionSpace>(
          exec, 0, number_of_values * number_of_elements),
      KOKKOS_LAMBDA(const size_t k) {
        const size_t i = k % number_of_values;
        const size_t e = k / number_of_values;
        result(i, e) = resolved(0).evaluate(i, e, result(i, e));
      });
}

/*!
 * \ingroup TimeSteppersGroup
 * \brief Apply \p coefficients to the data in \p history on the device
 * for a batch of elements, storing the updated values in \p u and
 * returning the error estimate of each element if one is requested.
 *
 * This is the device version of `apply_step_coefficients`.  The update
 * and the error estimates are computed in a single kernel, reducing the
 * error measure (see `largest_stepper_error`) of each element with a
 * maximum.  The update and the estimate at the stepper order are
 * computed in a single pass over the data.  Lower-order estimates (for
 * `StepperErrorTolerances::Estimates::AllOrders`) take one more pass
 * each.
 *
 * The coefficients and the locations of the history data are passed to
 * the kernel through \p parameters, so the kernel's arguments don't
 * change between calls.  See `DeviceParameters` for when \p parameters
 * can be reused.  With error estimates, this waits for \p exec, since the
 * estimates are needed on the host.
 */
template <typename ExecutionSpace>
std::vector<std::optional<StepperErrorEstimate>>
apply_step_coefficients_on_device(
    const ExecutionSpace& exec,
    const HistoryBatchView<typename ExecutionSpace::memory_space>& u,
    const DeviceHistoryView<typename ExecutionSpace::memory_space>& history,
    const StepCoefficients& coefficients,
    const StepperErrorTolerances& tolerances,
    const gsl::not_null<
        StepCoefficientsParameters<typename ExecutionSpace::memory_space>*>
        parameters) {
  using MemorySpace = typename ExecutionSpace::memory_space;
  using HistoryCoefficientsKokkos_detail::max_error_orders;
  const size_t number_of_values = u.extent(0);
  const size_t number_of_elements = u.extent(1);
  std::vector<std::optional<StepperErrorEstimate>> result(number_of_elements);
  HistoryCoefficientsKokkos_detail::StepParameters step_parameters{};
  step_parameters.update = HistoryCoefficientsKokkos_detail::resolve(
      history, coefficients.update, number_of_values, number_of_elements);
  if (not coefficients.error.has_value()) {
    parameters->set(exec, step_parameters);
    const auto device_parameters = parameters->device();
    Kokkos::parallel_for(
        "apply_step_coefficients_on_device",
        Kokkos::RangePolicy<ExecutionSpace>(
            exec, 0, number_of_values * number_of_elements),
        KOKKOS_LAMBDA(const size_t k) {
          const size_t i = k % number_of_values;
          const size_t e = k / number_of_values;
          u(i, e) = device_parameters(0).update.evaluate(i, e, u(i, e));
        });
    return result;
  }

  const ErrorCoefficients& error_coefficients = *coefficients.error;
  if (not gsl::at(error_coefficients.errors, error_coefficients.order)
              .has_value()) {
    ERROR("No error estimate at the stepper order "
          << error_coefficients.order);
  }
  step_parameters.error = HistoryCoefficientsKokkos_detail::resolve(
      history, *gsl::at(error_coefficients.errors, error_coefficients.order),
      number_of_values, number_of_elements);
  ASSERT(not step_parameters.error.accumulate,
         "Error estimates cannot accumulate.");
  // The estimates at other orders than the stepper order.
  std::array<size_t, max_error_orders> lower_orders{};
  for (size_t order = 0; order < max_error_orders; ++order) {
    if (order != error_coefficients.order and
        gsl::at(error_coefficients.errors, order).has_value()) {
      auto& lower_order_error =
          step_parameters
              .lower_order_errors[step_parameters.number_of_lower_orders];
      lower_order_error = HistoryCoefficientsKokkos_detail::resolve(
          history, *gsl::at(error_coefficients.errors, order), number_of_values,
          number_of_elements);
      ASSERT(not lower_order_error.accumulate,
             "Error estimates cannot accumulate.");
      gsl::at(lower_orders, step_parameters.number_of_lower_orders) = order;
      ++step_parameters.number_of_lower_orders;
    }
  }
  // The error is measured relative to the value of the latest step.
  step_parameters.scaled_error = HistoryCoefficientsKokkos_detail::ScaledError{
      HistoryCoefficientsKokkos_detail::record_data(
          history, HistoryTerm::Record::Step, history.steps.size() - 1,
          HistoryTerm::Field::Value, number_of_values, number_of_elements),
      tolerances.absolute, tolerances.relative};
  parameters->set(exec, step_parameters);
  const auto device_parameters = parameters->device();
  const size_t number_of_lower_orders = step_parameters.number_of_lower_orders;

  // One team per element reduces the error measures of that element.
  // Column 0 is the estimate at the stepper order, followed by the
  // lower orders.
  const Kokkos::View<double**, Kokkos::LayoutRight, MemorySpace> largest_errors(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "largest_errors"),
      number_of_elements, 1 + number_of_lower_orders);
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  Kokkos::parallel_for(
      "apply_step_coefficients_on_device",
      TeamPolicy(exec, static_cast<int>(number_of_elements), Kokkos::AUTO),
      KOKKOS_LAMBDA(const typename TeamPolicy::member_type& team) {
        const auto& params = device_parameters(0);
        const size_t e = static_cast<size_t>(team.league_rank());
        double largest_error = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, number_of_values),
            [&](const size_t i, double& local_largest_error) {
              local_largest_error = std::max(
                  local_largest_error,
                  params.scaled_error(i, e, params.error.evaluate(i, e, 0.0)));
              u(i, e) = params.update.evaluate(i, e, u(i, e));
            },
            Kokkos::Max<double>(largest_error));
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { largest_errors(e, 0) = largest_error; });
        for (size_t k = 0; k < number_of_lower_orders; ++k) {
          double largest_lower_order_error = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, number_of_values),
              [&](const size_t i, double& local_largest_error) {
                local_largest_error = std::max(
                    local_largest_error,
                    params.scaled_error(
                        i, e,
                        params.lower_order_errors[k].evaluate(i, e, 0.0)));
              },
              Kokkos::Max<double>(largest_lower_order_error));
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            largest_errors(e, 1 + k) = largest_lower_order_error;
          });
        }
      });

  const auto host_largest_errors =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, largest_errors);
  Kokkos::deep_copy(exec, host_largest_errors, largest_errors);
  exec.fence();
  for (size_t e = 0; e < number_of_elements; ++e) {
    auto& estimate = result[e].emplace(
        error_coefficients.step_time, error_coefficients.step_size,
        error_coefficients.order, host_largest_errors(e, 0));
    for (size_t k = 0; k < number_of_lower_orders; ++k) {
      gsl::at(estimate.errors, gsl::at(lower_orders, k))
          .emplace(host_largest_errors(e, 1 + k));
    }
  }
  return result;
}
}  // namespace TimeSteppers
#endif  // SPECTRE_KOKKOS
