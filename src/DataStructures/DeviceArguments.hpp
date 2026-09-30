// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/Tags/MirrorView.hpp"

namespace DeviceArguments_detail {
template <typename Tag>
constexpr bool is_tensor_v = tt::is_a_v<Tensor, typename Tag::type>;

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
  using type = ::Tags::MirrorView<Tag>;
};
}  // namespace DeviceArguments_detail

/*!
 * \brief The tag under which the argument `Tag` is stored in
 * `DeviceArguments`.
 *
 * Tensors are stored as `Tags::MirrorView<Tag>`, i.e. in device memory.
 * Trivially copyable values (e.g. `double`, enums, `std::array`) are stored
 * as-is under `Tag`, since they are copied to the device with the kernel.
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
 * inside the kernel with `device_argument_at_index`.
 */
template <typename... Tags>
using DeviceArguments = tuples::TaggedTuple<device_argument_tag<Tags>...>;

/// Copy the argument `Tag` to its representation in `DeviceArguments`,
/// enqueuing any copies on the execution space instance `exec` (see
/// `copy_to_device`).
template <typename Tag, typename ExecSpace>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto to_device_argument(const ExecSpace& exec,
                        const typename Tag::type& value) {
  if constexpr (DeviceArguments_detail::is_tensor_v<Tag>) {
    return copy_to_device(exec, value);
  } else {
    return value;
  }
}

/// @{
/*!
 * \brief Create the `DeviceArguments` for `Tags` from their host values, e.g.
 * `make_device_arguments<ArgsTags...>(db::get<ArgsTags>(box)...)`.
 *
 * The copies are enqueued on the execution space instance `exec`, so the kernel
 * that uses the arguments must run on the same instance (or fence it first),
 * and the host values must stay alive until `exec` is fenced. Without `exec`
 * the default instance is used and this function blocks until the copies are
 * complete.
 */
template <typename... Tags, typename ExecSpace>
  requires Kokkos::is_execution_space_v<ExecSpace>
DeviceArguments<Tags...> make_device_arguments(
    const ExecSpace& exec, const typename Tags::type&... values) {
  return DeviceArguments<Tags...>{to_device_argument<Tags>(exec, values)...};
}

template <typename... Tags>
DeviceArguments<Tags...> make_device_arguments(
    const typename Tags::type&... values) {
  const Kokkos::DefaultExecutionSpace exec{};
  auto device_arguments = make_device_arguments<Tags...>(exec, values...);
  exec.fence("make_device_arguments");
  return device_arguments;
}
/// @}

/*!
 * \brief The argument `Tag` at grid point `i`.
 *
 * Returns a `Tensor<double>` (by value) for tensors and a reference to the
 * stored value otherwise.
 */
template <typename Tag, typename... DeviceTags>
KOKKOS_FUNCTION decltype(auto) device_argument_at_index(
    const tuples::TaggedTuple<DeviceTags...>& device_arguments,
    const size_t i) {
  if constexpr (DeviceArguments_detail::is_tensor_v<Tag>) {
    return make_at_index(get<::Tags::MirrorView<Tag>>(device_arguments), i);
  } else {
    return get<Tag>(device_arguments);
  }
}
#endif  // SPECTRE_KOKKOS
