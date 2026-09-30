// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariables.hpp"

namespace Tags {
/*!
 * \brief Prefix tag for a non-tensor argument of a pointwise function that has
 * a different value for each element of a batch.
 *
 * `make_device_arguments` takes a `std::vector` with one value per element for
 * it, and `device_argument_at_index` returns the value of the element.
 */
template <typename Tag>
struct PerElement : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = typename Tag::type;
};
}  // namespace Tags

namespace DeviceArguments_detail {
template <typename Tag>
constexpr bool is_tensor_v = tt::is_a_v<Tensor, typename Tag::type>;

// The values of a `Tags::PerElement` argument on the device
template <typename Tag>
struct PerElementView : db::PrefixTag {
  using tag = Tag;
  using type = Kokkos::View<const typename Tag::type*>;
};

template <typename Tag, bool IsTensor = is_tensor_v<Tag>>
struct DeviceTag {
  static_assert(std::is_trivially_copyable_v<typename Tag::type> and
                    not std::is_pointer_v<typename Tag::type>,
                "Only tensors and trivially copyable values (without host "
                "pointers) can be passed to a Kokkos kernel as arguments.");
  using type = Tag;
};

template <typename Tag>
struct DeviceTag<Tag, true> {
  using type = ::Tags::BatchView<Tag>;
};

template <typename Tag>
struct DeviceTag<::Tags::PerElement<Tag>, false> {
  static_assert(not is_tensor_v<Tag>,
                "Tensors already have a value for each element of a batch, "
                "so don't wrap them in `Tags::PerElement`.");
  using type = PerElementView<Tag>;
};

// The device tensor of one element as a batch of one element, which refers to
// the same memory (without reference counting)
template <typename Tag, typename DeviceTensor>
typename ::Tags::BatchView<Tag>::type as_batch_of_one(
    const DeviceTensor& device_tensor) {
  typename ::Tags::BatchView<Tag>::type batch{};
  for (size_t i = 0; i < device_tensor.size(); ++i) {
    const size_t number_of_grid_points = device_tensor[i].extent(0);
    batch[i] = typename ::Tags::BatchView<Tag>::type::type{
        device_tensor[i].data(),
        Kokkos::LayoutStride{number_of_grid_points, device_tensor[i].stride(0),
                             1, number_of_grid_points}};
  }
  return batch;
}

template <typename Tag>
struct is_per_element : std::false_type {};

template <typename Tag>
struct is_per_element<::Tags::PerElement<Tag>> : std::true_type {};
}  // namespace DeviceArguments_detail

/*!
 * \brief The tag under which the argument `Tag` is stored in
 * `DeviceArguments`.
 *
 * Tensors are stored as `Tags::BatchView<Tag>`, i.e. in device memory for a
 * batch of elements (see `BatchedVariables`). Arguments that have a value for
 * each element of the batch (`Tags::PerElement`) are stored as a
 * `Kokkos::View` of the values. Other trivially copyable values (e.g.
 * `double`, enums, `std::array`) are stored as-is under `Tag`, since they are
 * copied to the device with the kernel and are the same for all elements.
 *
 * \warning Trivially copyable types that hold pointers to host memory (e.g.
 * `gsl::span`) can't be detected and must not be passed to a kernel.
 */
template <typename Tag>
using device_argument_tag =
    typename DeviceArguments_detail::DeviceTag<Tag>::type;

/*!
 * \brief A `tuples::TaggedTuple` holding the arguments `Tags` of a pointwise
 * function so they can be copied to a Kokkos kernel by value.
 *
 * Create it with `make_device_arguments` and read the argument at a grid point
 * of an element inside the kernel with `device_argument_at_index`.
 */
template <typename... Tags>
using DeviceArguments = tuples::TaggedTuple<device_argument_tag<Tags>...>;

/*!
 * \brief The argument `Tag` in device memory, from either its host value or
 * (for tensors) a batch that is already in device memory.
 *
 * A host tensor is copied to a batch of one element. A tensor in device
 * memory (e.g. from `get<Tags::MirrorView<Tag>>(device_variables)`) is used
 * as a batch of one element, and a batch of tensors (e.g. from
 * `get<Tags::BatchView<Tag>>(batched_variables)`) is used as-is, both without
 * copying. A `Tags::PerElement` argument takes a `std::vector` with one value
 * per element.
 *
 * \warning A tensor in device memory is referred to without reference
 * counting, so it must stay alive until the kernel that uses it is complete.
 */
template <typename Tag, typename ExecSpace, typename Value>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto to_device_argument(const ExecSpace& exec, const Value& value) {
  if constexpr (DeviceArguments_detail::is_tensor_v<Tag>) {
    using BatchTensor = typename ::Tags::BatchView<Tag>::type;
    if constexpr (std::is_same_v<Value, BatchTensor>) {
      return value;
    } else if constexpr (std::is_same_v<
                             Value, typename ::Tags::MirrorView<Tag>::type>) {
      return DeviceArguments_detail::as_batch_of_one<Tag>(value);
    } else {
      static_assert(std::is_same_v<Value, typename Tag::type>,
                    "A tensor argument must be the host tensor, the tensor "
                    "in device memory, or a batch of the tensor in device "
                    "memory.");
      using DeviceView =
          Kokkos::View<double**, Kokkos::LayoutLeft,
                       typename Kokkos::DefaultExecutionSpace::memory_space>;
      using HostUnmanaged =
          Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace,
                       Kokkos::MemoryUnmanaged>;
      BatchTensor device_tensor{};
      for (size_t i = 0; i < value.size(); ++i) {
        // Every component is overwritten, so don't initialize the memory
        const DeviceView component{
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "Tensor"),
            value[i].size(), 1};
        Kokkos::deep_copy(exec, component,
                          HostUnmanaged(value[i].data(), value[i].size(), 1));
        device_tensor[i] = component;
      }
      return device_tensor;
    }
  } else if constexpr (DeviceArguments_detail::is_per_element<Tag>::value) {
    using ValueType = typename Tag::type;
    static_assert(std::is_same_v<Value, std::vector<ValueType>>,
                  "A `Tags::PerElement` argument must be a `std::vector` with "
                  "one value per element.");
    const Kokkos::View<ValueType*> device_values{
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "PerElement"),
        value.size()};
    // Copied element by element, since `std::vector<bool>` has no `data()`
    const auto host_values =
        Kokkos::create_mirror_view(Kokkos::HostSpace{}, device_values);
    for (size_t e = 0; e < value.size(); ++e) {
      host_values(e) = value[e];
    }
    Kokkos::deep_copy(exec, device_values, host_values);
    // `host_values` is deallocated when this function returns
    exec.fence("to_device_argument");
    return Kokkos::View<const ValueType*>{device_values};
  } else {
    static_assert(std::is_same_v<Value, typename Tag::type>,
                  "A non-tensor argument must have the type of its tag.");
    return value;
  }
}

/*!
 * \brief Create the `DeviceArguments` for `Tags` from their values, e.g.
 * `make_device_arguments<ArgsTags...>(db::get<ArgsTags>(box)...)`.
 *
 * See `to_device_argument` for the values that each argument can have. The
 * copies are enqueued on the execution space instance `exec`, so the kernel
 * that uses the arguments must run on the same instance (or fence it first),
 * and the host values must stay alive until `exec` is fenced. Without `exec`
 * the default instance is used and this function blocks until the copies are
 * complete.
 */
template <typename... Tags, typename ExecSpace, typename... Values>
  requires(Kokkos::is_execution_space_v<ExecSpace> and
           sizeof...(Tags) == sizeof...(Values))
DeviceArguments<Tags...> make_device_arguments(const ExecSpace& exec,
                                               const Values&... values) {
  return DeviceArguments<Tags...>{to_device_argument<Tags>(exec, values)...};
}

template <typename... Tags, typename... Values>
  requires(sizeof...(Tags) == sizeof...(Values))
DeviceArguments<Tags...> make_device_arguments(const Values&... values) {
  const Kokkos::DefaultExecutionSpace exec{};
  auto device_arguments = make_device_arguments<Tags...>(exec, values...);
  exec.fence("make_device_arguments");
  return device_arguments;
}

/*!
 * \brief The argument `Tag` at grid point `i` of the element `e` of the batch.
 *
 * Returns a `Tensor<double>` (by value) for tensors, the value of the element
 * for `Tags::PerElement` arguments, and a reference to the stored value
 * otherwise.
 */
template <typename Tag, typename... DeviceTags>
KOKKOS_FUNCTION decltype(auto) device_argument_at_index(
    const tuples::TaggedTuple<DeviceTags...>& device_arguments, const size_t i,
    const size_t e = 0) {
  if constexpr (DeviceArguments_detail::is_tensor_v<Tag>) {
    return make_at_index(get<::Tags::BatchView<Tag>>(device_arguments), i, e);
  } else if constexpr (DeviceArguments_detail::is_per_element<Tag>::value) {
    return get<DeviceArguments_detail::PerElementView<typename Tag::tag>>(
        device_arguments)(e);
  } else {
    (void)i;
    (void)e;
    return get<Tag>(device_arguments);
  }
}
#endif  // SPECTRE_KOKKOS
