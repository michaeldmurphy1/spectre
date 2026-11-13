// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <vector>

#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/HistoryCoefficients.hpp"
#include "Time/TimeSteppers/HistoryCoefficientsKokkos.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * \brief The time-stepper histories of a batch of elements, with the
 * data in device memory.
 *
 * This is the device counterpart of `History`, for
 * `number_of_elements()` elements that share their record times, as
 * with global time stepping (e.g., the FD elements of a patch).  A
 * single element is a batch of one.  Each record holds a derivative
 * and possibly a value, each a `HistoryBatchView` of
 * `number_of_values()` doubles per element (e.g., the storage of a
 * `BatchedVariables`), with the data of each element contiguous.  The
 * data is stored in a single allocation of "slots", each holding the
 * data of one record for all elements, that is reused as records are
 * added and removed, so that a steady-state evolution performs no
 * device allocations.  The allocation grows when more slots are needed
 * (or by calling `reserve`), which invalidates any views obtained
 * before.  Allocations and the copies of a growing allocation are
 * done in the execution space instance passed to the function that
 * allocates.
 *
 * The bookkeeping (the record times and which slots hold their data)
 * is done on the host, so the `TimeStepper` methods that only depend
 * on the times can be used through `times()`, `clean`, and
 * `can_change_step_size`.  The device data is used through
 * `device_view()`, e.g., with `apply_step_coefficients_on_device`.
 *
 * \note The bookkeeping is a host `History<double>` whose entries are
 * slot indices, so that `TimeStepper::clean_history` manages it
 * unchanged.  A slot is in use as long as an entry of that history
 * refers to it.  `History::undo_latest` is not supported, because a
 * discarded value is not kept.
 */
template <typename MemorySpace =
              typename Kokkos::DefaultExecutionSpace::memory_space>
class DeviceHistory {
 public:
  /// The device storage of a record.  `value` is empty if the record
  /// doesn't hold a value.
  struct RecordView {
    HistoryBatchView<MemorySpace> value{};
    HistoryBatchView<MemorySpace> derivative{};
  };

  DeviceHistory() = default;
  DeviceHistory(const size_t integration_order, const size_t number_of_values,
                const size_t number_of_elements)
      : bookkeeping_(integration_order),
        number_of_values_(number_of_values),
        number_of_elements_(number_of_elements) {}
  // Copying a `Kokkos::View` doesn't copy the data, so a copy would
  // alias the original.
  DeviceHistory(const DeviceHistory&) = delete;
  DeviceHistory& operator=(const DeviceHistory&) = delete;
  DeviceHistory(DeviceHistory&&) = default;
  DeviceHistory& operator=(DeviceHistory&&) = default;
  ~DeviceHistory() = default;

  /// The order the time stepper is running at.  See `History`.
  /// @{
  size_t integration_order() const { return bookkeeping_.integration_order(); }
  void integration_order(const size_t new_integration_order) {
    bookkeeping_.integration_order(new_integration_order);
  }
  /// @}

  /// The number of doubles in each value and derivative of each
  /// element.  This can only be changed while the history is empty,
  /// and frees the allocation if it changes.
  /// @{
  size_t number_of_values() const { return number_of_values_; }
  void set_number_of_values(const size_t number_of_values) {
    if (number_of_values == number_of_values_) {
      return;
    }
    if (not empty()) {
      ERROR(
          "Cannot change the number of values of a non-empty "
          "DeviceHistory from "
          << number_of_values_ << " to " << number_of_values);
    }
    number_of_values_ = number_of_values;
    storage_ = StorageView{};
  }
  /// @}

  /// The number of elements in the batch.
  size_t number_of_elements() const { return number_of_elements_; }

  /// The number of steps (not substeps) in the history.
  size_t size() const { return bookkeeping_.size(); }
  bool empty() const { return bookkeeping_.empty(); }
  size_t number_of_substeps() const { return bookkeeping_.substeps().size(); }
  bool at_step_start() const { return bookkeeping_.at_step_start(); }

  /// The `TimeStepId` of step \p index, earliest first.
  const TimeStepId& step_id(const size_t index) const {
    return bookkeeping_[index].time_step_id;
  }

  /// The times in the history, for the `TimeStepper` coefficient
  /// methods.
  HistoryTimes times() const { return HistoryTimes(bookkeeping_); }

  /*!
   * \brief The host bookkeeping of the history, whose entries hold the
   * slot indices of the data.
   *
   * This is a `History` with the same times as this history, so code
   * that only uses the times of a history (e.g.,
   * `TimeStepper::can_change_step_size`) can use it.  It may be
   * modified in any way that doesn't add entries or restore removed
   * ones (`insert`, `insert_initial`, `insert_in_place`, and
   * `undo_latest`), e.g., to remove entries, change the integration
   * order, or change a `TimeStepId` to that of the same time in a new
   * slab.
   */
  /// @{
  const History<double>& bookkeeping() const { return bookkeeping_; }
  History<double>& bookkeeping() { return bookkeeping_; }
  /// @}

  /*!
   * \brief Add a record and return its device storage, to be filled by
   * the caller (e.g., by the kernel computing the time derivative).
   *
   * The value is stored only if \p store_value is true.  This
   * corresponds to `History::insert_in_place`, and to
   * `History::insert_initial` if \p initial is true.  If the
   * allocation grows, it does so in \p exec.
   */
  template <typename ExecutionSpace>
  RecordView insert(const ExecutionSpace& exec, const TimeStepId& time_step_id,
                    const bool store_value, const bool initial = false) {
    const size_t derivative_slot = free_slot(exec);
    std::optional<size_t> value_slot{};
    if (store_value) {
      value_slot = free_slot(exec, derivative_slot);
    }
    const auto as_double = [](const size_t slot) {
      return static_cast<double>(slot);
    };
    if (initial) {
      if (value_slot.has_value()) {
        bookkeeping_.insert_initial(time_step_id, as_double(*value_slot),
                                    as_double(derivative_slot));
      } else {
        bookkeeping_.insert_initial(time_step_id, History<double>::no_value,
                                    as_double(derivative_slot));
      }
    } else {
      if (value_slot.has_value()) {
        bookkeeping_.insert(time_step_id, as_double(*value_slot),
                            as_double(derivative_slot));
      } else {
        bookkeeping_.insert(time_step_id, History<double>::no_value,
                            as_double(derivative_slot));
      }
    }
    return RecordView{.value = value_slot.has_value()
                                   ? slot_view(*value_slot)
                                   : HistoryBatchView<MemorySpace>{},
                      .derivative = slot_view(derivative_slot)};
  }

  /// Remove old entries using `TimeStepper::clean_history`.
  void clean(const TimeStepper& stepper) {
    stepper.clean_history(make_not_null(&bookkeeping_));
  }

  /// See `TimeStepper::can_change_step_size`.
  bool can_change_step_size(const TimeStepper& stepper,
                            const TimeStepId& time_id) const {
    return stepper.can_change_step_size(time_id, bookkeeping_);
  }

  /// See `History`.
  /// @{
  void discard_value(const TimeStepId& id_to_discard) {
    bookkeeping_.discard_value(id_to_discard);
  }
  void pop_front() { bookkeeping_.pop_front(); }
  void clear_substeps() { bookkeeping_.clear_substeps(); }
  void clear() { bookkeeping_.clear(); }
  /// @}

  /// The value at the start of the step containing \p time, which is
  /// the initial value for dense output.  See `History::step_start`.
  HistoryBatchView<MemorySpace> step_start_value(const double time) const {
    const auto& record = bookkeeping_.step_start(time);
    if (not record.value.has_value()) {
      ERROR("The value at " << record.time_step_id << " has been discarded.");
    }
    return slot_view(slot_index(*record.value));
  }

  /// Device views of the data, indexed like `times()`, for the
  /// functions applying `TimeStepper` coefficients on the device.
  DeviceHistoryView<MemorySpace> device_view() const {
    DeviceHistoryView<MemorySpace> result{};
    const auto record_view = [this](const StepRecord<double>& record) {
      DeviceHistoryRecordView<MemorySpace> view{};
      if (record.value.has_value()) {
        view.value = slot_view(slot_index(*record.value));
      }
      view.derivative = slot_view(slot_index(record.derivative));
      return view;
    };
    for (const auto& record : bookkeeping_) {
      result.steps.push_back(record_view(record));
    }
    for (const auto& record : bookkeeping_.substeps()) {
      result.substeps.push_back(record_view(record));
    }
    return result;
  }

  /// The device buffers through which the coefficients applied to this
  /// history are passed to the kernels (see `DeviceParameters`): one for the
  /// update of a step (`apply_step_coefficients_on_device`) and one for dense
  /// output (`apply_history_linear_combination_on_device`).
  /// @{
  gsl::not_null<StepCoefficientsParameters<MemorySpace>*>
  step_coefficients_parameters() {
    return &step_coefficients_parameters_;
  }
  gsl::not_null<HistoryCombinationParameters<MemorySpace>*>
  dense_output_parameters() {
    return &dense_output_parameters_;
  }
  /// @}

  /// The number of slots allocated.
  size_t capacity() const { return storage_.extent(1); }

  /// Allocate at least \p number_of_slots slots, in \p exec.
  template <typename ExecutionSpace>
  void reserve(const ExecutionSpace& exec, const size_t number_of_slots) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecutionSpace, MemorySpace>::accessible);
    if (number_of_slots <= capacity()) {
      return;
    }
    StorageView new_storage(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "DeviceHistory"),
        slot_size(), number_of_slots);
    if (capacity() > 0) {
      Kokkos::deep_copy(
          exec,
          Kokkos::subview(new_storage, Kokkos::ALL(),
                          Kokkos::make_pair(size_t{0}, capacity())),
          storage_);
    }
    storage_ = std::move(new_storage);
  }

  /// Serialization copies the device data through the host.
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | bookkeeping_;
    p | number_of_values_;
    p | number_of_elements_;
    size_t number_of_slots = capacity();
    p | number_of_slots;
    if (p.isUnpacking()) {
      storage_ = StorageView{};
      const typename MemorySpace::execution_space exec{};
      reserve(exec, number_of_slots);
      exec.fence();
    }
    if (number_of_slots > 0) {
      if (p.isUnpacking()) {
        const auto host_storage = Kokkos::create_mirror_view(storage_);
        PUParray(p, host_storage.data(), host_storage.size());
        Kokkos::deep_copy(storage_, host_storage);
      } else {
        const auto host_storage =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, storage_);
        PUParray(p, host_storage.data(), host_storage.size());
      }
    }
  }

 private:
  using StorageView = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>;

  static size_t slot_index(const double slot) {
    return static_cast<size_t>(slot);
  }

  // The number of doubles in a slot: the data of all elements.
  size_t slot_size() const { return number_of_values_ * number_of_elements_; }

  // A view of the batch data starting at \p data, with the data of each
  // element contiguous.  The pointer isn't dereferenced, so it may point
  // to device memory.
  HistoryBatchView<MemorySpace> batch_view(double* const data) const {
    return HistoryBatchView<MemorySpace>(
        data, Kokkos::LayoutStride(number_of_values_, 1, number_of_elements_,
                                   number_of_values_));
  }

  HistoryBatchView<MemorySpace> slot_view(const size_t slot) const {
    ASSERT(slot < capacity(),
           "Slot " << slot << " out of range " << capacity());
    // The data of a slot is contiguous, and the pointer arithmetic
    // doesn't access device memory from the host.
    return batch_view(storage_.data() + slot * storage_.stride(1));
  }

  // The lowest slot that no record uses, excluding \p reserved_slot.
  // Grows the allocation in \p exec if all slots are in use.
  template <typename ExecutionSpace>
  size_t free_slot(const ExecutionSpace& exec,
                   const std::optional<size_t> reserved_slot = std::nullopt) {
    std::vector<bool> in_use(capacity(), false);
    const auto mark = [&in_use](const StepRecord<double>& record) {
      if (record.value.has_value()) {
        in_use[slot_index(*record.value)] = true;
      }
      in_use[slot_index(record.derivative)] = true;
    };
    for (const auto& record : bookkeeping_) {
      mark(record);
    }
    for (const auto& record : bookkeeping_.substeps()) {
      mark(record);
    }
    if (reserved_slot.has_value()) {
      in_use[*reserved_slot] = true;
    }
    const auto unused = std::find(in_use.begin(), in_use.end(), false);
    if (unused != in_use.end()) {
      return static_cast<size_t>(unused - in_use.begin());
    }
    const size_t slot = capacity();
    reserve(exec, std::max(2 * capacity(), size_t{2}));
    return slot;
  }

  History<double> bookkeeping_{};
  size_t number_of_values_ = 0;
  size_t number_of_elements_ = 1;
  // The data of slot `s` is `storage_(:, s)`, holding the data of
  // element `e` at `e * number_of_values_`.
  StorageView storage_{};
  // Not serialized: they are set before each use
  StepCoefficientsParameters<MemorySpace> step_coefficients_parameters_{
      "DeviceHistory step coefficients"};
  HistoryCombinationParameters<MemorySpace> dense_output_parameters_{
      "DeviceHistory dense output coefficients"};
};

}  // namespace TimeSteppers
#endif  // SPECTRE_KOKKOS
