// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace evolution::patches {
/*!
 * \brief A copy in pinned host memory of the variables `TagsList` of the
 * elements of a patch, for host code that reads them, e.g. the element boxes
 * of events (see the patch section of the GPU support guide).
 *
 * \details The host memory is allocated at the first copy and kept for later
 * copies, so a patch whose events never read its variables doesn't hold it,
 * and the copies of later observations don't allocate. It is pinned, so that
 * the copy from the device is asynchronous. The source batch may hold more
 * variables than `TagsList`, e.g. all spacetime variables at the cell centers
 * of which the element boxes only hold some. These are gathered on the device
 * first, into a device buffer that is also kept, so that each copy to the
 * host is a single transfer.
 *
 * \note The copy holds all `TagsList` of all elements, as the DataBox of an
 * element on the host does. Events often read only a few of them, e.g. one
 * tensor for an `ObserveNorms`, so copying only the tensors the events of an
 * observation read would cost less time and memory. That needs each event to
 * report the tensors it reads, including those that its compute tags read,
 * and DataBoxes whose other variables hold no data, so it isn't done.
 */
template <typename TagsList>
class HostCopy {
 public:
  using HostBatch = BatchedVariables<TagsList, Kokkos::SharedHostPinnedSpace>;

  /*!
   * \brief Enqueue the copy of the `TagsList` of `source` to the host on
   * `exec`, allocating the memory at the first call, and return the host
   * batch.
   *
   * \details The host batch holds the copy once `exec` is fenced. It is valid
   * until the next call, which overwrites it. It isn't const, so that host
   * `Variables` can refer to its data (see `set_data_ref_to_element`). Don't
   * call this again before `exec` is fenced, so that two copies into the same
   * memory don't overlap.
   */
  template <typename SourceTagsList>
  HostBatch& enqueue_copy(const Kokkos::DefaultExecutionSpace& exec,
                          const BatchedVariables<SourceTagsList>& source) {
    static_assert(
        tmpl::size<tmpl::list_difference<TagsList, SourceTagsList>>::value == 0,
        "The source batch must hold all variables of the host copy.");
    const size_t number_of_grid_points = source.number_of_grid_points();
    const size_t number_of_elements = source.number_of_elements();
    if (host_.number_of_grid_points() != number_of_grid_points or
        host_.number_of_elements() != number_of_elements) {
      host_ = HostBatch{exec, number_of_grid_points, number_of_elements};
    }
    if constexpr (std::is_same_v<TagsList, SourceTagsList>) {
      deep_copy(exec, make_not_null(&host_), source);
    } else {
      if (gathered_.number_of_grid_points() != number_of_grid_points or
          gathered_.number_of_elements() != number_of_elements) {
        gathered_ = BatchedVariables<TagsList>{exec, number_of_grid_points,
                                               number_of_elements};
      }
      tmpl::for_each<TagsList>([&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const auto& from = get<::Tags::BatchView<Tag>>(source);
        const auto& to = get<::Tags::BatchView<Tag>>(gathered_);
        for (size_t component = 0; component < from.size(); ++component) {
          Kokkos::deep_copy(exec, to[component], from[component]);
        }
      });
      deep_copy(exec, make_not_null(&host_), gathered_);
    }
    return host_;
  }

  /// Whether the host memory is allocated, i.e. `enqueue_copy()` was called
  bool is_allocated() const { return host_.number_of_elements() > 0; }

 private:
  HostBatch host_{};
  BatchedVariables<TagsList> gathered_{};
};
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
