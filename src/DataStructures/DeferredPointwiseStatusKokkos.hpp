// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace DeferredPointwiseStatus_detail {
// The failure record on the device. Kernels count all failures, and the first
// thread that claims the record saves the grid point, element and snapshot of
// its failure.
struct Record {
  size_t number_of_failures;
  size_t grid_point;
  size_t element;
  int claimed;
  int any_flag;
};
}  // namespace DeferredPointwiseStatus_detail

/*!
 * \brief The `PointwiseStatus` of pointwise functions on the device, recorded
 * without waiting for the device.
 *
 * \details Returning a `PointwiseStatusSummary` from
 * `apply_pointwise_on_device` reduces the status into host memory, so the host
 * waits for the kernel. Passing a `DeferredPointwiseStatus` instead records
 * the failures in device memory, so the kernel is only enqueued. The status
 * is then:
 * 1. cleared with `reset()`,
 * 2. accumulated by any number of kernels (e.g. on the lower and the upper
 *    faces of a batch),
 * 3. copied to the host with `copy_to_host()`, which enqueues the copy,
 * 4. read with `summary()` and `snapshot()` once the caller has fenced the
 *    execution space instance anyway, e.g. before sending data.
 *
 * Errors are then reported later than with a `PointwiseStatusSummary`, when
 * the data at the failed grid point may have changed. So the status saves a
 * *snapshot* of the arguments `SnapshotTags` (tensors that are arguments of
 * the function) at one failed grid point, which `snapshot()` returns for the
 * error message.
 *
 * Which failed grid point is recorded is whichever thread claims the record
 * first. On a GPU this can differ between runs, unlike the
 * `PointwiseStatusSummary::first_failed_grid_point`, which is the failure
 * with the smallest index. The number of failures is exact.
 */
template <typename SnapshotTags = tmpl::list<>>
class DeferredPointwiseStatus;

template <typename... SnapshotTags>
class DeferredPointwiseStatus<tmpl::list<SnapshotTags...>> {
 public:
  using Record = DeferredPointwiseStatus_detail::Record;
  using snapshot_tags = tmpl::list<SnapshotTags...>;
  /// The number of values in the snapshot
  static constexpr size_t snapshot_size =
      (0 + ... + ::Tags::AtIndex<SnapshotTags>::type::size());

  DeferredPointwiseStatus() = default;
  /// Allocate the status on the device and its copy on the host
  explicit DeferredPointwiseStatus(const Kokkos::DefaultExecutionSpace& exec)
      : record_(Kokkos::view_alloc(exec, "DeferredPointwiseStatus")),
        snapshot_(Kokkos::view_alloc(exec, "DeferredPointwiseStatusSnapshot"),
                  snapshot_size),
        host_record_(Kokkos::view_alloc(exec, "DeferredPointwiseStatusHost")),
        host_snapshot_(
            Kokkos::view_alloc(exec, "DeferredPointwiseStatusSnapshotHost"),
            snapshot_size) {}

  /// Whether the status is allocated
  bool is_allocated() const { return record_.is_allocated(); }

  /// Enqueue clearing the status on `exec`
  void reset(const Kokkos::DefaultExecutionSpace& exec) const {
    Kokkos::deep_copy(exec, record_, Record{0, 0, 0, 0, 0});
  }

  /// Enqueue copying the status to the host on `exec`. Its values are on the
  /// host after `exec` is fenced.
  void copy_to_host(const Kokkos::DefaultExecutionSpace& exec) const {
    Kokkos::deep_copy(exec, host_record_, record_);
    if constexpr (snapshot_size > 0) {
      Kokkos::deep_copy(exec, host_snapshot_, snapshot_);
    }
  }

  /// @{
  /// The status at the last `copy_to_host()`. Only valid after the execution
  /// space instance of the copy is fenced.
  bool failed() const { return host_record_().number_of_failures > 0; }
  PointwiseStatusSummary summary() const {
    const Record& record = host_record_();
    return {record.any_flag != 0, record.number_of_failures, record.grid_point,
            record.element};
  }
  /// The `SnapshotTags` at the recorded failure. Only meaningful if
  /// `failed()`.
  tuples::TaggedTuple<::Tags::AtIndex<SnapshotTags>...> snapshot() const {
    tuples::TaggedTuple<::Tags::AtIndex<SnapshotTags>...> result{};
    size_t offset = 0;
    const auto unpack = [this, &offset](auto& tensor) {
      for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = host_snapshot_(offset + i);
      }
      offset += tensor.size();
    };
    (unpack(get<::Tags::AtIndex<SnapshotTags>>(result)), ...);
    return result;
  }
  /// @}

  /// @{
  /// The device data, written by the kernels
  const Kokkos::View<Record>& record() const { return record_; }
  const Kokkos::View<double*>& snapshot_values() const { return snapshot_; }
  /// @}

 private:
  Kokkos::View<Record> record_{};
  Kokkos::View<double*> snapshot_{};
  Kokkos::View<Record, Kokkos::SharedHostPinnedSpace> host_record_{};
  Kokkos::View<double*, Kokkos::SharedHostPinnedSpace> host_snapshot_{};
};
#endif  // SPECTRE_KOKKOS
