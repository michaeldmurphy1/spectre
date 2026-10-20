// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesKokkos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace Tags {
/*!
 * \brief Prefix tag for the tensor `Tag` of a batch of elements in the Kokkos
 * memory space `Space`.
 *
 * Each component is a `Kokkos::View<double**, Kokkos::LayoutStride>` indexed
 * by grid point and element, `(i, e)`. See `BatchedVariables`.
 */
template <typename Tag,
          typename Space = typename Kokkos::DefaultExecutionSpace::memory_space>
struct BatchView : db::PrefixTag {
  using tag = Tag;
  using type = TensorMetafunctions::swap_type<
      Kokkos::View<double**, Kokkos::LayoutStride, Space>, typename Tag::type>;
};
}  // namespace Tags

/// \cond
template <typename TagsList,
          typename Space = typename Kokkos::DefaultExecutionSpace::memory_space>
class BatchedVariables;
/// \endcond

/*!
 * \brief The variables `Tags` of a batch of elements in one allocation in the
 * Kokkos memory space `Space`.
 *
 * The data of each element has the same layout as a `Variables<Tags>`: all
 * grid points of the first tensor component, then all grid points of the
 * second, and so on. The elements follow each other in memory. All elements
 * have the same number of grid points.
 *
 * The data can be accessed in two ways, both without copying:
 * - `get<Tags::BatchView<Tag>>(batch)` is the tensor `Tag` of all elements.
 *   Each component is a view indexed by grid point and element, `(i, e)`.
 *   This is what pointwise kernels use (see `apply_pointwise_on_device`).
 * - `batch.element(e)` is a `Variables` of the `Tags::MirrorView<Tags>` of the
 *   element `e`, e.g. for code that works on one element at a time.
 *
 * `Tags` hold `Tensor<DataVector>`s, like the tags of a host `Variables`. With
 * `SPECTRE_NAN_INIT`, newly allocated data is set to signaling NaNs, like for
 * `Variables`.
 */
template <typename... Tags, typename Space>
class BatchedVariables<tmpl::list<Tags...>, Space> {
 public:
  using tags_list = tmpl::list<Tags...>;
  using memory_space = Space;
  static constexpr size_t number_of_independent_components =
      (... + Tags::type::size());
  /// The storage, indexed by grid point, tensor component and element
  using storage_type = Kokkos::View<double***, Kokkos::LayoutLeft, Space>;
  /// The `Variables` of one element
  using element_type =
      Variables<tmpl::list<::Tags::MirrorView<Tags, Space>...>>;

  BatchedVariables() = default;

  /// Allocate the data of `number_of_elements` elements with
  /// `number_of_grid_points` grid points each. The data is only initialized
  /// with `SPECTRE_NAN_INIT`.
  template <typename ExecSpace>
    requires Kokkos::is_execution_space_v<ExecSpace>
  BatchedVariables(const ExecSpace& exec, const size_t number_of_grid_points,
                   const size_t number_of_elements)
      : BatchedVariables(storage_type{
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                               "BatchedVariables"),
            number_of_grid_points, number_of_independent_components,
            number_of_elements}) {
#ifdef SPECTRE_NAN_INIT
    // Like `Variables`, so that reading uninitialized data is caught
    Kokkos::parallel_for(
        "BatchedVariables::nan_init",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<size_t>>(
            exec, 0, storage_.size()),
        VariablesKokkos_detail::FillWithSignalingNaN<double>{storage_.data()});
#endif  // SPECTRE_NAN_INIT
  }

  BatchedVariables(const size_t number_of_grid_points,
                   const size_t number_of_elements)
      : BatchedVariables(typename Space::execution_space{},
                         number_of_grid_points, number_of_elements) {}

  explicit BatchedVariables(storage_type storage)
      : storage_(std::move(storage)) {
    ASSERT(storage_.extent(1) == number_of_independent_components,
           "The storage has " << storage_.extent(1)
                              << " tensor components, but the tags have "
                              << number_of_independent_components);
    size_t component_offset = 0;
    tmpl::for_each<tags_list>(
        [this, &component_offset]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          auto& tensor = get<::Tags::BatchView<Tag, Space>>(batch_views_);
          for (size_t i = 0; i < tensor.size(); ++i) {
            tensor[i] = Kokkos::subview(storage_, Kokkos::ALL(),
                                        component_offset, Kokkos::ALL());
            ++component_offset;
          }
        });
  }

  size_t number_of_grid_points() const { return storage_.extent(0); }
  size_t number_of_elements() const { return storage_.extent(2); }

  storage_type& view() { return storage_; }
  const storage_type& view() const { return storage_; }

  /// The data of the element `e`, which shares the memory of the batch
  element_type element(const size_t e) const {
    ASSERT(e < number_of_elements(),
           "Element " << e << " is not in the batch of " << number_of_elements()
                      << " elements");
    return element_type{typename element_type::storage_type{
        Kokkos::subview(storage_, Kokkos::ALL(), Kokkos::ALL(), e)}};
  }

  template <typename Tag>
  friend const typename Tag::type& get(const BatchedVariables& batch) {
    return tuples::get<Tag>(batch.batch_views_);
  }

 private:
  storage_type storage_{};
  tuples::TaggedTuple<::Tags::BatchView<Tags, Space>...> batch_views_{};
};

/*!
 * \brief Make `host_element` refer to the data of the element `e` of the
 * host batch `host_batch`, without copying.
 *
 * This lets host code that works on `Variables<Tags>` (with `DataVector`s)
 * write directly into a batch that is then copied to the device in one go.
 */
template <typename... Tags>
void set_data_ref_to_element(
    const gsl::not_null<Variables<tmpl::list<Tags...>>*> host_element,
    BatchedVariables<tmpl::list<Tags...>, Kokkos::HostSpace>& host_batch,
    const size_t e) {
  ASSERT(e < host_batch.number_of_elements(),
         "Element " << e << " is not in the batch of "
                    << host_batch.number_of_elements() << " elements");
  constexpr size_t number_of_components =
      Variables<tmpl::list<Tags...>>::number_of_independent_components;
  host_element->set_data_ref(
      Kokkos::subview(host_batch.view(), Kokkos::ALL(), Kokkos::ALL(), e)
          .data(),
      host_batch.number_of_grid_points() * number_of_components);
}

/// Enqueue the copy of the batch `source` into `destination` on `exec`. Both
/// must have the same number of grid points and elements.
template <typename ExecSpace, typename TagsList, typename DestinationSpace,
          typename SourceSpace>
  requires Kokkos::is_execution_space_v<ExecSpace>
void deep_copy(
    const ExecSpace& exec,
    const gsl::not_null<BatchedVariables<TagsList, DestinationSpace>*>
        destination,
    const BatchedVariables<TagsList, SourceSpace>& source) {
  Kokkos::deep_copy(exec, destination->view(), source.view());
}
#endif  // SPECTRE_KOKKOS
