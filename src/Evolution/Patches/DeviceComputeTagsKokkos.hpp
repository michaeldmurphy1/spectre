// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyMatricesKokkos.hpp"

/// \cond
class Event;
namespace Events {
template <typename ObservableTensorTagsList, typename NonTensorComputeTagsList,
          typename ArraySectionIdTag, typename OptionName>
class ObserveNorms;
}  // namespace Events
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

/*!
 * \brief Evaluate observable tensors on the device from the compute tags that
 * define them on the host
 *
 * \details A patch observes a tensor on the device the way the DataBox of an
 * element computes it on the host: from the compute tags of the DataBoxes of
 * the elements and of the events. For a tensor (the `base` of a compute tag
 * or a simple tag) the evaluator takes the first of these providers:
 * 1. A batch that the patch holds (`Sources::base_tags`), e.g. the evolved
 *    and primitive variables, the metric it computes for the fixing, the
 *    geometry, or values it knows, e.g. a vanishing gauge source function.
 * 2. A spatial derivative `::Tags::deriv<U, 3, Frame::Inertial>` of a tensor
 *    `U` that can be evaluated, computed with the differentiation matrices of
 *    the mesh of the patch and the inverse Jacobian, as `::Tags::DerivCompute`
 *    does on the observer mesh.
 * 3. A compute tag of the universe whose `base` is the tensor and whose
 *    `function` can be called with tensors of `double`s (a function template
 *    over the data type, marked `KOKKOS_FUNCTION`), after evaluating its
 *    argument tags. One kernel computes it at all grid points.
 *
 * Whether a tensor can be evaluated is known at compile time
 * (`is_provided`), so events whose tensors can't be evaluated keep running on
 * the host.
 *
 * The `Sources` of a system provide the batches, see
 * `grmhd::GhValenciaDivClean::fd_patch::DeviceObservableSources` for an
 * example.
 */
namespace evolution::patches::device_compute {
/// The batch of a tensor: its components at the grid points of a range of
/// elements, indexed by grid point, component and element
using View = Kokkos::View<double***, Kokkos::LayoutStride>;

/// How the functions of the compute tags are called
enum class Mode {
  /// Inlined into the kernels
  Generic,
  /// As `Generic`, but call the functions of the compute tags out of line, as
  /// for functions defined in other files with separate device compilation
  GenericOutOfLine
};

/*!
 * \brief The `Mode` of the evaluation of observables on the device
 *
 * \details For experiments: the environment variable
 * `SPECTRE_DEVICE_OBSERVABLES` selects it (`generic`, the default, or
 * `generic_out_of_line`), read once.
 */
Mode mode();

/// The number of elements evaluated at a time, which bounds the memory of the
/// intermediate tensors. The environment variable
/// `SPECTRE_DEVICE_OBSERVABLES_CHUNK` sets it (default 128), read once.
size_t chunk_size();

/// The elements `first` to `first + size - 1` of a patch
struct ElementRange {
  size_t first;
  size_t size;
};

/*!
 * \brief Buffers of the evaluation that are kept between observations
 *
 * \details The batches of the intermediate tensors (allocated for the largest
 * number of elements they were used with, and reused), the differentiation
 * matrices of the mesh, and the norm statistics.
 */
class Scratch {
 public:
  /// A batch of `number_of_components` components for `number_of_elements`
  /// elements, named `name` (the same name always gives the same memory)
  View get(const std::string& name, size_t number_of_grid_points,
           size_t number_of_components, size_t number_of_elements);

  /// The operators of the logical derivatives in each dimension on `mesh`
  std::array<DeviceApplyMatrices, 3>& logical_derivative_operators(
      const std::string& name, const Mesh<3>& mesh);

  /// The norm statistics of all elements, indexed by element and column
  Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight>& statistics(
      size_t number_of_elements, size_t number_of_columns);

 private:
  std::unordered_map<std::string, Kokkos::View<double***, Kokkos::LayoutLeft>>
      views_{};
  Mesh<3> mesh_{};
  std::unordered_map<std::string, std::array<DeviceApplyMatrices, 3>>
      derivative_operators_{};
  Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight> statistics_{};
};

/// \cond
template <typename Tag>
struct NoProvider {};
template <typename Tag>
struct BaseProvider {};
template <typename Tag, typename DerivativeOf>
struct DerivativeProvider {};
template <typename Tag, typename ComputeTag>
struct ComputeProvider {};
/// \endcond

namespace detail {
template <typename T>
using double_type = TensorMetafunctions::swap_type<double, T>;

// Whether the `function` of `ComputeTag` can be called with tensors of
// doubles
template <typename ComputeTag, typename... ArgumentTags>
concept CallableWithDoubles = requires(
    gsl::not_null<double_type<typename ComputeTag::base::type>*> result,
    const double_type<typename ArgumentTags::type>&... arguments) {
  ComputeTag::function(result, arguments...);
};

template <typename ComputeTag, typename ArgumentTags>
struct callable_with_doubles;

template <typename ComputeTag, typename... ArgumentTags>
struct callable_with_doubles<ComputeTag, tmpl::list<ArgumentTags...>>
    : std::bool_constant<CallableWithDoubles<ComputeTag, ArgumentTags...>> {};

template <typename Tag>
struct derivative_of {
  using type = void;
};
template <typename Tag>
struct derivative_of<::Tags::deriv<Tag, tmpl::size_t<3>, Frame::Inertial>> {
  using type = Tag;
};

template <typename Tag>
struct base_of {
  using type = Tag;
};
template <typename Tag>
  requires db::is_compute_tag_v<Tag>
struct base_of<Tag> {
  using type = typename Tag::base;
};
}  // namespace detail

/// The tensor of `Tag`: the `base` of a compute tag, else `Tag` itself
template <typename Tag>
using base_of = typename detail::base_of<Tag>::type;

/// The provider of `Tag` (see `device_compute`): `BaseProvider`,
/// `DerivativeProvider`, `ComputeProvider` or `NoProvider`
template <typename Tag, typename Sources, typename Universe>
struct provider;

/// Whether `Tag` can be evaluated on the device
template <typename Tag, typename Sources, typename Universe>
constexpr bool is_provided =
    not tt::is_a_v<NoProvider, typename provider<Tag, Sources, Universe>::type>;

namespace detail {
template <typename ComputeTag, typename Sources, typename Universe>
struct compute_tag_is_evaluable;

template <typename ComputeTag, typename Sources, typename Universe>
  requires(tt::is_a_v<Tensor, typename ComputeTag::base::type>)
struct compute_tag_is_evaluable<ComputeTag, Sources, Universe> {
  template <typename... ArgumentTags>
  static constexpr bool arguments_provided(
      tmpl::list<ArgumentTags...> /*meta*/) {
    return (... and is_provided<ArgumentTags, Sources, Universe>);
  }
  static constexpr bool value =
      callable_with_doubles<ComputeTag,
                            typename ComputeTag::argument_tags>::value and
      arguments_provided(typename ComputeTag::argument_tags{});
};

template <typename ComputeTag, typename Sources, typename Universe>
  requires(not tt::is_a_v<Tensor, typename ComputeTag::base::type>)
struct compute_tag_is_evaluable<ComputeTag, Sources, Universe>
    : std::false_type {};

template <typename Tag, typename Sources, typename Universe,
          typename Candidates>
struct find_compute_tag {
  using type = NoProvider<Tag>;
};

template <typename Tag, typename Sources, typename Universe,
          typename ComputeTag, typename... ComputeTags>
struct find_compute_tag<Tag, Sources, Universe,
                        tmpl::list<ComputeTag, ComputeTags...>> {
  using type = typename std::conditional_t<
      std::conjunction_v<
          std::is_same<typename base_of<ComputeTag>::type, Tag>,
          compute_tag_is_evaluable<ComputeTag, Sources, Universe>>,
      std::type_identity<ComputeProvider<Tag, ComputeTag>>,
      find_compute_tag<Tag, Sources, Universe,
                       tmpl::list<ComputeTags...>>>::type;
};

// The provider of `Tag`, as a `std::type_identity`. The branches that aren't
// taken aren't instantiated, which ends the recursion.
template <typename Tag, typename Sources, typename Universe>
constexpr auto resolve() {
  using derivative = typename derivative_of<Tag>::type;
  if constexpr (tmpl::list_contains_v<typename Sources::base_tags, Tag>) {
    return std::type_identity<BaseProvider<Tag>>{};
  } else if constexpr (not std::is_void_v<derivative>) {
    if constexpr (is_provided<derivative, Sources, Universe>) {
      return std::type_identity<DerivativeProvider<Tag, derivative>>{};
    } else {
      return std::type_identity<NoProvider<Tag>>{};
    }
  } else {
    return std::type_identity<typename find_compute_tag<
        Tag, Sources, Universe,
        tmpl::filter<Universe, db::is_compute_tag<tmpl::_1>>>::type>{};
  }
}
}  // namespace detail

template <typename Tag, typename Sources, typename Universe>
struct provider {
  using type =
      typename decltype(detail::resolve<Tag, Sources, Universe>())::type;
};

/// Whether `T` is an `Events::ObserveNorms`
template <typename T>
struct IsObserveNorms : std::false_type {};
/// \cond
template <typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct IsObserveNorms<::Events::ObserveNorms<Fields, NonTensorComputeTags,
                                             ArraySectionIdTag, OptionName>>
    : std::true_type {};
/// \endcond

/// The `ObserveNorms` events of the executable
template <typename Metavariables>
using observe_norms_events = tmpl::filter<
    tmpl::at<typename Metavariables::factory_creation::factory_classes, Event>,
    IsObserveNorms<tmpl::_1>>;

namespace detail {
template <typename Event>
struct observable_tags;
template <typename... Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct observable_tags<
    ::Events::ObserveNorms<tmpl::list<Fields...>, NonTensorComputeTags,
                           ArraySectionIdTag, OptionName>> {
  using type = tmpl::list<Fields...>;
  using compute_tags = tmpl::append<
      tmpl::filter<tmpl::list<Fields...>, db::is_compute_tag<tmpl::_1>>,
      tmpl::filter<NonTensorComputeTags, db::is_compute_tag<tmpl::_1>>>;
};
template <typename Event>
using observable_tags_t = typename observable_tags<Event>::type;
template <typename Event>
using event_compute_tags_t = typename observable_tags<Event>::compute_tags;
}  // namespace detail

/// The tags that the `ObserveNorms` events of the executable can observe
template <typename Metavariables>
using observable_tags = tmpl::remove_duplicates<tmpl::flatten<
    tmpl::transform<observe_norms_events<Metavariables>,
                    tmpl::bind<detail::observable_tags_t, tmpl::_1>>>>;

/// The compute tags from which observable tensors are evaluated: those of
/// the DataBoxes of the elements (`Sources::compute_tags`) and of the
/// `ObserveNorms` events
template <typename Metavariables, typename Sources>
using universe = tmpl::remove_duplicates<
    tmpl::append<typename Sources::compute_tags,
                 tmpl::flatten<tmpl::transform<
                     observe_norms_events<Metavariables>,
                     tmpl::bind<detail::event_compute_tags_t, tmpl::_1>>>>>;

/// Whether the tensor named `name` (its `db::tag_name`) that an
/// `ObserveNorms` of the executable observes can be evaluated on the device
template <typename Metavariables, typename Sources>
bool is_device_observable(const std::string& name) {
  bool result = false;
  tmpl::for_each<observable_tags<Metavariables>>(
      [&name, &result]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        if (db::tag_name<Tag>() == name) {
          result = is_provided<base_of<Tag>, Sources,
                               universe<Metavariables, Sources>>;
        }
      });
  return result;
}

/// The norm statistics of the observed tensors of all elements of a patch,
/// on the host (see `device_norm_statistics`)
struct DeviceNormStatistics {
  /// An observed tensor and where its components are in `statistics`
  struct ObservedTensor {
    /// The name of the tensor
    std::string name;
    /// The column of the first component in `statistics`
    size_t first_component;
    /// The names of the components, as `Tensor::get_vector_of_data()`
    std::vector<std::string> component_names;
  };
  /// The tensors, in the order of their components in `statistics`
  std::vector<ObservedTensor> tensors{};
  /// The statistics, indexed by element and column. The extra last column is
  /// the constant 1, whose `integral` is the volume of the element.
  Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight, Kokkos::HostSpace>
      statistics{};
};

/*!
 * \brief The `Events::NormStatistics` of the observable tensors named
 * `tensor_names` of all elements of a patch, evaluated on the device (see
 * `device_compute`) and copied to the host
 *
 * \details The elements are evaluated `chunk_size()` at a time. The norm
 * statistics are reduced from the batches of the tensors without copying
 * them. This waits for `exec` to complete.
 *
 * Executables instantiate this in a separate file (see
 * `DeviceComputeTagsKokkos.tpp`), which keeps the kernels of all observable
 * tensors out of the file of the executable.
 */
template <typename Metavariables, typename Sources>
DeviceNormStatistics device_norm_statistics(
    const Kokkos::DefaultExecutionSpace& exec, const Sources& sources,
    const std::vector<std::string>& tensor_names);
}  // namespace evolution::patches::device_compute
#endif  // SPECTRE_KOKKOS
