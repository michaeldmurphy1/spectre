// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS

namespace VariablesKokkos_detail {
// Sets the data to signaling NaNs in a kernel. `Kokkos::deep_copy` with a value
// can't be used for this, because it compares the value to zero on the host,
// which raises a floating-point exception for a signaling NaN.
template <typename ValueType>
struct FillWithSignalingNaN {
  KOKKOS_FUNCTION void operator()(const size_t i) const {
    data[i] = std::numeric_limits<ValueType>::signaling_NaN();
  }
  ValueType* data;
};
}  // namespace VariablesKokkos_detail

/*!
 * \brief Variables specialization that is backed by a Kokkos::View, meaning
 * it can be allocated on device (GPU) memory.
 *
 * Given a list of tags that hold `Tensor<Kokkos::View<double*>>`, this class
 * allocates a larger `Kokkos::View<double*[num_components]>` to hold all the
 * tensor components in contiguous memory. The memory is either allocated on the
 * host or on a device, depending on the `Kokkos::View` (see Kokkos
 * documentation for details). Individual tensors can be accessed with reference
 * semantics, meaning the `Kokkos::View` returned by `get` are subviews that
 * point into the large memory allocation.
 *
 * This class doesn't support all the arithmetic operations that the `Variables`
 * class backed by a `DataVector` supports. This is because arithmetic
 * operations should be done on device using Kokkos parallel algorithms with
 * precise control over launching kernels on device.
 */
template <typename... Tags>
  requires(sizeof...(Tags) > 0 and
           (Kokkos::is_view<typename Tags::type::type>::value and ...))
class Variables<tmpl::list<Tags...>> {
 public:
  using tags_list = tmpl::list<Tags...>;
  static constexpr auto number_of_variables = sizeof...(Tags);
  static constexpr size_t number_of_independent_components =
      (... + Tags::type::size());

  // The 1D Kokkos::View that backs the tensors, e.g. Kokkos::View<double*>.
  using vector_type = typename tmpl::front<tags_list>::type::type;
  static_assert(vector_type::rank() == 1,
                "Variables can only be used with a 1D Kokkos::View. "
                "A second dimension over tensor components is added "
                "automatically.");
  // The value type of the Kokkos::View, e.g. double.
  using value_type = typename vector_type::value_type;
  // Store the data on device in a 2D Kokkos::View over grid points and tensor
  // components. The two common layouts are:
  // - SoA: tensor components x grid points
  //   (typically better for derivatives because neighboring grid points
  //   are contiguous for each tensor component)
  // - AoS: grid points x tensor components
  //   (typically better for pointwise operations because all tensor components
  //   for a grid point are contiguous)
  // We choose SoA ("LayoutLeft") for now, but this can be changed if needed and
  // should be measured for performance.
  using storage_type =
      Kokkos::View<value_type* [number_of_independent_components],
                   Kokkos::LayoutLeft, typename vector_type::memory_space>;
  // The Kokkos memory space the data lives in, e.g. Kokkos::CudaSpace.
  using memory_space = typename vector_type::memory_space;

  Variables() = default;
  explicit Variables(const size_t number_of_grid_points) {
    initialize(number_of_grid_points);
  }
  explicit Variables(storage_type&& rhs) : storage_(std::move(rhs)) {
    set_reference_variable_data();
  }
  ~Variables() = default;

  /// @{
  /*!
   * \brief Initialize the memory for the Variables. Does nothing if the number
   * of grid points is unchanged. Fills the memory with NaN only if
   * SPECTRE_NAN_INIT is defined.
   *
   * The allocation and the NaN fill are enqueued on the execution space
   * instance `exec`, so work that uses the data must run on the same instance
   * (or fence it first). Without `exec` the default instance of the memory
   * space's execution space is used and this function blocks until it is
   * done. You can specify a `memory_space` to allocate the memory on a specific
   * GPU.
   */
  template <typename ExecSpace>
    requires Kokkos::is_execution_space_v<ExecSpace>
  void initialize(const ExecSpace& exec, const size_t number_of_grid_points) {
    initialize_impl(exec, number_of_grid_points, memory_space{});
  }

  void initialize(const size_t number_of_grid_points,
                  const memory_space& space = {}) {
    const typename memory_space::execution_space exec{};
    if (initialize_impl(exec, number_of_grid_points, space)) {
      exec.fence("Variables::initialize");
    }
  }
  /// @}

  constexpr SPECTRE_ALWAYS_INLINE size_t size() const {
    return storage_.size();
  }
  constexpr SPECTRE_ALWAYS_INLINE size_t number_of_grid_points() const {
    return storage_.extent(0);
  }

  storage_type& view() { return storage_; }
  const storage_type& view() const { return storage_; }

  template <typename Tag, typename TagList>
  friend constexpr typename Tag::type& get(  // NOLINT
      Variables<TagList>& v);
  template <typename Tag, typename TagList>
  friend constexpr const typename Tag::type& get(  // NOLINT
      const Variables<TagList>& v);

  template <typename Space>
  auto create_mirror_view(const Space& space) const {
    return Variables<tmpl::list<::Tags::MirrorView<Tags, Space>...>>(
        Kokkos::create_mirror_view(space, storage_));
  }

  template <typename Space>
  auto create_mirror_view_and_copy(const Space& space) const {
    return Variables<tmpl::list<::Tags::MirrorView<Tags, Space>...>>(
        Kokkos::create_mirror_view_and_copy(space, storage_));
  }

 private:
  // Returns whether the memory was reallocated
  template <typename ExecSpace>
  bool initialize_impl(const ExecSpace& exec,
                       const size_t number_of_grid_points,
                       const memory_space& space) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace, memory_space>::accessible,
        "The execution space must be able to access the memory space of the "
        "Variables.");
    if (number_of_grid_points == this->number_of_grid_points()) {
      return false;
    }
    storage_ =
        storage_type(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                                        "Variables", space),
                     number_of_grid_points);
#if defined(SPECTRE_NAN_INIT)
    Kokkos::parallel_for(
        "Variables::nan_init",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<size_t>>(
            exec, 0, storage_.size()),
        VariablesKokkos_detail::FillWithSignalingNaN<value_type>{
            storage_.data()});
#endif  // SPECTRE_NAN_INIT
    set_reference_variable_data();
    return true;
  }

  void set_reference_variable_data() {
    size_t variable_offset = 0;
    tmpl::for_each<tags_list>([this, &variable_offset](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      auto& var = tuples::get<Tag>(reference_variable_data_);
      for (size_t i = 0; i < Tag::type::size(); ++i) {
        // The subview is contiguous if each tensor component is contiguous in
        // memory, which is the case for Kokkos::LayoutLeft (SoA). Otherwise,
        // the subview has a stride.
        var[i] = Kokkos::subview(storage_, Kokkos::ALL(), variable_offset);
        ++variable_offset;
      }
    });
  }

  storage_type storage_{};
  tuples::TaggedTuple<Tags...> reference_variable_data_;
};

/// Get each tensor at the given grid-point index.
template <typename... VarTags, typename... Is>
KOKKOS_FUNCTION auto make_at_index(
    const Variables<tmpl::list<VarTags...>>& vars, const Is&... i) {
  tuples::TaggedTuple<::Tags::AtIndex<VarTags>...> result{};
  (..., (get<::Tags::AtIndex<VarTags>>(result) =
             make_at_index(get<VarTags>(vars), i...)));
  return result;
}

/// @{
/// Set each tensor at the given grid-point index.
template <typename... VarTags, typename... Is>
KOKKOS_FUNCTION void set_at_index(
    const gsl::not_null<Variables<tmpl::list<VarTags...>>*> vars,
    const tuples::TaggedTuple<::Tags::AtIndex<VarTags>...>& value,
    const Is&... i) {
  (..., (set_at_index(make_not_null(&get<VarTags>(*vars)),
                      get<::Tags::AtIndex<VarTags>>(value), i...)));
}
template <typename... VarTags, typename... Is>
KOKKOS_FUNCTION void set_at_index(
    const gsl::not_null<const Variables<tmpl::list<VarTags...>>*> vars,
    const tuples::TaggedTuple<::Tags::AtIndex<VarTags>...>& value,
    const Is&... i) {
  (..., (set_at_index(make_not_null(&get<VarTags>(*vars)),
                      get<::Tags::AtIndex<VarTags>>(value), i...)));
}
/// @}

namespace VariablesKokkos_detail {
// Views that wrap the data of a DataVector-backed Variables without owning it,
// so it can be deep-copied. Relies on the memory layout of the
// DataVector-backed Variables being the same as the Kokkos view (LayoutLeft).
template <typename HostVars>
using HostUnmanaged =
    Kokkos::View<typename HostVars::value_type *
                     [HostVars::number_of_independent_components],
                 Kokkos::LayoutLeft, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>;
template <typename HostVars>
using ConstHostUnmanaged =
    Kokkos::View<const typename HostVars::value_type *
                     [HostVars::number_of_independent_components],
                 Kokkos::LayoutLeft, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>;
}  // namespace VariablesKokkos_detail

/// @{
/*!
 * \brief Allocate a `Variables` of the same size as `vars_host` in device
 * memory, without copying the data yet.
 *
 * The allocation is enqueued on the execution space instance `exec` (see
 * `Variables::initialize`). Without `exec` this blocks until the memory is
 * ready. You can specify a `memory_space` to allocate the memory on a specific
 * GPU.
 */
template <typename ExecSpace, typename... Tags>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto allocate_on_device(const ExecSpace& exec,
                        const Variables<tmpl::list<Tags...>>& vars_host) {
  static_assert((std::is_same_v<typename Tags::type::type, DataVector> and ...),
                "Only a DataVector-backed Variables can be mirrored in device "
                "memory.");
  Variables<tmpl::list<::Tags::MirrorView<Tags>...>> vars_device{};
  if constexpr (sizeof...(Tags) > 0) {
    vars_device.initialize(exec, vars_host.number_of_grid_points());
  }
  return vars_device;
}

template <typename... Tags>
auto allocate_on_device(
    const Variables<tmpl::list<Tags...>>& vars_host,
    const typename Kokkos::DefaultExecutionSpace::memory_space& space = {}) {
  static_assert((std::is_same_v<typename Tags::type::type, DataVector> and ...),
                "Only a DataVector-backed Variables can be mirrored in device "
                "memory.");
  Variables<tmpl::list<::Tags::MirrorView<Tags>...>> vars_device{};
  if constexpr (sizeof...(Tags) > 0) {
    vars_device.initialize(vars_host.number_of_grid_points(), space);
  }
  return vars_device;
}
/// @}

/// @{
/*!
 * \brief Copy the `vars_host` to an existing `Variables` in device memory. The
 * device memory is only reallocated if the number of grid points changed.
 *
 * The allocation and copy are enqueued on the execution space instance `exec`,
 * so work that uses the result must run on the same instance (or fence it
 * first), and `vars_host` must stay alive until `exec` is fenced. Without
 * `exec` the default instance is used and this function blocks until the copy
 * is complete, like `Kokkos::deep_copy`.
 */
template <typename ExecSpace, typename DeviceTags, typename... Tags>
  requires Kokkos::is_execution_space_v<ExecSpace>
void copy_to_device(const ExecSpace& exec,
                    const gsl::not_null<Variables<DeviceTags>*> vars_device,
                    const Variables<tmpl::list<Tags...>>& vars_host) {
  static_assert((std::is_same_v<typename Tags::type::type, DataVector> and ...),
                "Only a DataVector-backed Variables can be copied to device "
                "memory.");
  if constexpr (sizeof...(Tags) > 0) {
    const size_t num_points = vars_host.number_of_grid_points();
    vars_device->initialize(exec, num_points);
    const VariablesKokkos_detail::ConstHostUnmanaged<
        Variables<tmpl::list<Tags...>>>
        unmanaged_host_view(vars_host.data(), num_points);
    Kokkos::deep_copy(exec, vars_device->view(), unmanaged_host_view);
  }
}

template <typename DeviceTags, typename... Tags>
void copy_to_device(const gsl::not_null<Variables<DeviceTags>*> vars_device,
                    const Variables<tmpl::list<Tags...>>& vars_host) {
  const Kokkos::DefaultExecutionSpace exec{};
  copy_to_device(exec, vars_device, vars_host);
  exec.fence("copy_to_device");
}
/// @}

/// @{
/*!
 * \brief Copy the `vars_host` to device memory.
 *
 * See the other overloads for the meaning of `exec`. You can specify a
 * `memory_space` to allocate the memory on a specific GPU.
 */
template <typename ExecSpace, typename... Tags>
  requires Kokkos::is_execution_space_v<ExecSpace>
auto copy_to_device(const ExecSpace& exec,
                    const Variables<tmpl::list<Tags...>>& vars_host) {
  auto vars_device = allocate_on_device(exec, vars_host);
  copy_to_device(exec, make_not_null(&vars_device), vars_host);
  return vars_device;
}

template <typename... Tags>
auto copy_to_device(
    const Variables<tmpl::list<Tags...>>& vars_host,
    const typename Kokkos::DefaultExecutionSpace::memory_space& space = {}) {
  auto vars_device = allocate_on_device(vars_host, space);
  copy_to_device(make_not_null(&vars_device), vars_host);
  return vars_device;
}
/// @}

/// @{
/*!
 * \brief Copy a `Variables` from device to host memory.
 *
 * The copy is enqueued on the execution space instance `exec` (the default
 * instance if not given), which is fenced before returning so the host data is
 * ready to use.
 */
template <typename ExecSpace, typename HostTags, typename DeviceTags>
  requires Kokkos::is_execution_space_v<ExecSpace>
void copy_to_host(const ExecSpace& exec,
                  const gsl::not_null<Variables<HostTags>*> vars_host,
                  const Variables<DeviceTags>& vars_device) {
  if constexpr (tmpl::size<DeviceTags>::value > 0) {
    using HostVars = Variables<HostTags>;
    static_assert(std::is_same_v<typename HostVars::vector_type, DataVector>,
                  "Only a DataVector-backed Variables can be copied from "
                  "device memory.");
    const size_t num_points = vars_device.number_of_grid_points();
    if (vars_host->number_of_grid_points() != num_points) {
      vars_host->initialize(num_points);
    }
    const VariablesKokkos_detail::HostUnmanaged<HostVars> unmanaged_host_view(
        vars_host->data(), num_points);
    Kokkos::deep_copy(exec, unmanaged_host_view, vars_device.view());
    exec.fence("copy_to_host");
  }
}

template <typename HostTags, typename DeviceTags>
void copy_to_host(const gsl::not_null<Variables<HostTags>*> vars_host,
                  const Variables<DeviceTags>& vars_device) {
  copy_to_host(Kokkos::DefaultExecutionSpace{}, vars_host, vars_device);
}
/// @}

#endif  // SPECTRE_KOKKOS
