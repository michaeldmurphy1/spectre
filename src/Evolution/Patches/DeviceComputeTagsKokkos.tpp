// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Patches/DeviceComputeTagsKokkos.hpp"

#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/ApplyMatricesKokkos.hpp"
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace evolution::patches::device_compute {
namespace detail {
#if defined(__CUDACC__)
#define SPECTRE_DEVICE_COMPUTE_NOINLINE __noinline__
#else
#define SPECTRE_DEVICE_COMPUTE_NOINLINE __attribute__((noinline))
#endif

// Calls the function of `ComputeTag` without inlining it, as a call to a
// function defined in another file would be with separate device compilation
template <typename ComputeTag, typename Result, typename... Arguments>
SPECTRE_DEVICE_COMPUTE_NOINLINE KOKKOS_FUNCTION void call_out_of_line(
    const gsl::not_null<Result*> result, const Arguments&... arguments) {
  ComputeTag::function(result, arguments...);
}
#undef SPECTRE_DEVICE_COMPUTE_NOINLINE

// The function of `ComputeTag` at a grid point, for `apply_pointwise_on_device`
template <typename ComputeTag, bool OutOfLine>
struct ComputeTagAtGridPoint {
  template <typename Result, typename... Arguments>
  KOKKOS_FUNCTION static void apply(const gsl::not_null<Result*> result,
                                    const Arguments&... arguments) {
    if constexpr (OutOfLine) {
      call_out_of_line<ComputeTag>(result, arguments...);
    } else {
      ComputeTag::function(result, arguments...);
    }
  }
};

// The batch of `Tag` in the components of `view`
template <typename Tag>
typename ::Tags::BatchView<Tag>::type as_batch(const View& view) {
  typename ::Tags::BatchView<Tag>::type result{};
  for (size_t c = 0; c < result.size(); ++c) {
    result[c] = Kokkos::subview(view, Kokkos::ALL(), c, Kokkos::ALL());
  }
  return result;
}

// The spatial derivatives from the logical ones, as `partial_derivatives`
struct DerivativeKernel {
  View derivatives;
  View logical_derivatives;
  View inverse_jacobian;
  // For each component of the derivative: its spatial index and the
  // component of the differentiated tensor
  Kokkos::View<const size_t* [2]> components;
  // The components of the inverse Jacobian (logical index, spatial index)
  Kokkos::Array<size_t, 9> inverse_jacobian_components;
  size_t number_of_components;
  size_t number_of_grid_points;

  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const size_t i = k % number_of_grid_points;
    const size_t e = k / number_of_grid_points;
    for (size_t c = 0; c < components.extent(0); ++c) {
      const size_t d = components(c, 0);
      const size_t u = components(c, 1);
      double result = inverse_jacobian(i, inverse_jacobian_components[d], e) *
                      logical_derivatives(i, u, e);
      for (size_t logical = 1; logical < 3; ++logical) {
        result += inverse_jacobian(
                      i, inverse_jacobian_components[logical * 3 + d], e) *
                  logical_derivatives(i, logical * number_of_components + u, e);
      }
      derivatives(i, c, e) = result;
    }
  }
};

// One column of the norm statistics: a component of a batch (`nullptr` for
// the constant 1 of the volume)
struct Column {
  const double* data;
  size_t point_stride;
  size_t element_stride;
};

// Kokkos reducer for `Events::NormStatistics`
struct NormStatisticsReducer {
  using reducer = NormStatisticsReducer;
  using value_type = Events::NormStatistics;
  using result_view_type =
      Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  KOKKOS_FUNCTION explicit NormStatisticsReducer(value_type& value)
      : value_(&value) {}

  KOKKOS_FUNCTION static void join(value_type& dest, const value_type& src) {
    dest.max = Kokkos::max(dest.max, src.max);
    dest.min = Kokkos::min(dest.min, src.min);
    dest.sum_abs += src.sum_abs;
    dest.sum_square += src.sum_square;
    dest.integral_abs += src.integral_abs;
    dest.integral_square += src.integral_square;
    dest.integral += src.integral;
  }
  KOKKOS_FUNCTION static void init(value_type& value) {
    value.max = Kokkos::Experimental::finite_min_v<double>;
    value.min = Kokkos::Experimental::finite_max_v<double>;
    value.sum_abs = 0.0;
    value.sum_square = 0.0;
    value.integral_abs = 0.0;
    value.integral_square = 0.0;
    value.integral = 0.0;
  }
  KOKKOS_FUNCTION value_type& reference() const { return *value_.data(); }
  KOKKOS_FUNCTION result_view_type view() const { return value_; }
  KOKKOS_FUNCTION static bool references_scalar() { return true; }

 private:
  result_view_type value_;
};

// The norm statistics of the `columns` of the elements of `range` into the
// rows of `statistics`
void norm_statistics_of_columns(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight>&
        statistics,
    const std::vector<Column>& columns, const View& det_inverse_jacobian,
    double point_weight, const ElementRange& range,
    size_t number_of_grid_points);

// Evaluates the tensors of the elements of a range (see `device_compute`),
// each once
template <typename Sources, typename Universe>
class Evaluator {
 public:
  Evaluator(const Kokkos::DefaultExecutionSpace& exec, const Sources& sources,
            const ElementRange& range, const Mode mode)
      : exec_(exec), sources_(sources), range_(range), mode_(mode) {}

  /// The batch of `Tag` for the elements of the range
  template <typename Tag>
  View get() {
    const std::string name = typeid(Tag).name();
    if (const auto it = evaluated_.find(name); it != evaluated_.end()) {
      return it->second;
    }
    const View result =
        evaluate(typename provider<Tag, Sources, Universe>::type{});
    evaluated_.emplace(name, result);
    return result;
  }

  /// A batch for intermediate data of the evaluation, named `name`
  View scratch(const std::string& name, const size_t number_of_components) {
    return sources_.scratch().get(name, sources_.number_of_grid_points(),
                                  number_of_components, range_.size);
  }

 private:
  template <typename Tag>
  View evaluate(NoProvider<Tag> /*meta*/) {
    ERROR("The tensor " << db::tag_name<Tag>()
                        << " can't be evaluated on the device.");
    return View{};
  }

  template <typename Tag>
  View evaluate(BaseProvider<Tag> /*meta*/) {
    return sources_.template base<Tag>(exec_, range_, *this);
  }

  template <typename Tag, typename DerivativeOf>
  View evaluate(DerivativeProvider<Tag, DerivativeOf> /*meta*/) {
    const Kokkos::Profiling::ScopedRegion region{"device_compute::derivative"};
    const View u = get<DerivativeOf>();
    const size_t number_of_components = DerivativeOf::type::size();
    const std::string name = typeid(Tag).name();
    const View logical_derivatives =
        scratch(name + " logical", 3 * number_of_components);
    auto& operators = sources_.scratch().logical_derivative_operators(
        name, sources_.subcell_mesh());
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(operators, d)
          .apply(exec_,
                 Kokkos::subview(logical_derivatives, Kokkos::ALL(),
                                 std::pair{d * number_of_components,
                                           (d + 1) * number_of_components},
                                 Kokkos::ALL()),
                 u);
    }
    // Where the components of the derivative come from
    using DerivativeType = typename Tag::type;
    const Kokkos::View<size_t* [2]> components{
        Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                           "device_compute derivative components"),
        DerivativeType::size()};
    const auto host_components =
        Kokkos::create_mirror_view(Kokkos::HostSpace{}, components);
    for (size_t c = 0; c < DerivativeType::size(); ++c) {
      const auto index =
          DerivativeType::structure::get_canonical_tensor_index(c);
      std::array<size_t, DerivativeOf::type::rank()> u_index{};
      for (size_t r = 0; r < u_index.size(); ++r) {
        gsl::at(u_index, r) = gsl::at(index, r + 1);
      }
      host_components(c, 0) = index[0];
      host_components(c, 1) = DerivativeOf::type::get_storage_index(u_index);
    }
    Kokkos::deep_copy(exec_, components, host_components);
    using InverseJacobianTag =
        ::domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                        Frame::Inertial>;
    Kokkos::Array<size_t, 9> inverse_jacobian_components{};
    for (size_t logical = 0; logical < 3; ++logical) {
      for (size_t i = 0; i < 3; ++i) {
        inverse_jacobian_components[logical * 3 + i] =
            InverseJacobianTag::type::get_storage_index(logical, i);
      }
    }
    const View result = scratch(name, DerivativeType::size());
    Kokkos::parallel_for(
        "device_compute::derivative",
        Kokkos::RangePolicy<>(exec_, 0,
                              sources_.number_of_grid_points() * range_.size),
        DerivativeKernel{result, logical_derivatives, get<InverseJacobianTag>(),
                         components, inverse_jacobian_components,
                         number_of_components,
                         sources_.number_of_grid_points()});
    // The host mirror is freed when this returns, so wait for the copy
    exec_.fence("device_compute::derivative");
    return result;
  }

  template <typename Tag, typename ComputeTag>
  View evaluate(ComputeProvider<Tag, ComputeTag> /*meta*/) {
    return evaluate_compute_tag<Tag, ComputeTag>(
        typename ComputeTag::argument_tags{});
  }

  template <typename Tag, typename ComputeTag, typename... ArgumentTags>
  View evaluate_compute_tag(tmpl::list<ArgumentTags...> /*meta*/) {
    // The arguments first, so that the batch of the result isn't overwritten
    // by an argument with the same name
    const DeviceArguments<ArgumentTags...> arguments{
        detail::as_batch<ArgumentTags>(get<ArgumentTags>())...};
    const View result = scratch(typeid(Tag).name(), Tag::type::size());
    const tuples::TaggedTuple<::Tags::BatchView<Tag>> results{
        detail::as_batch<Tag>(result)};
    if (mode_ == Mode::GenericOutOfLine) {
      ApplyPointwiseOnDevice_detail::apply_on_batch<
          ComputeTagAtGridPoint<ComputeTag, true>>(
          exec_, results, sources_.number_of_grid_points(), range_.size,
          tmpl::list<ArgumentTags...>{}, arguments);
    } else {
      ApplyPointwiseOnDevice_detail::apply_on_batch<
          ComputeTagAtGridPoint<ComputeTag, false>>(
          exec_, results, sources_.number_of_grid_points(), range_.size,
          tmpl::list<ArgumentTags...>{}, arguments);
    }
    return result;
  }

  const Kokkos::DefaultExecutionSpace& exec_;
  const Sources& sources_;
  ElementRange range_;
  Mode mode_;
  std::unordered_map<std::string, View> evaluated_{};
};
}  // namespace detail

template <typename Metavariables, typename Sources>
DeviceNormStatistics device_norm_statistics(
    const Kokkos::DefaultExecutionSpace& exec, const Sources& sources,
    const std::vector<std::string>& tensor_names) {
  const Kokkos::Profiling::ScopedRegion region{
      "device_compute::device_norm_statistics"};
  using Universe = universe<Metavariables, Sources>;
  using ObservableTags = observable_tags<Metavariables>;

  // The layout of the statistics: each tensor once, in the order requested
  DeviceNormStatistics result{};
  size_t number_of_columns = 0;
  for (const std::string& name : tensor_names) {
    if (alg::any_of(result.tensors, [&name](const auto& tensor) {
          return tensor.name == name;
        })) {
      continue;
    }
    bool found = false;
    tmpl::for_each<ObservableTags>([&]<typename Tag>(
                                       tmpl::type_<Tag> /*meta*/) {
      if constexpr (is_provided<base_of<Tag>, Sources, Universe>) {
        if (not found and db::tag_name<Tag>() == name) {
          found = true;
          using TensorType = typename base_of<Tag>::type;
          result.tensors.push_back({name, number_of_columns,
                                    TensorType{}.get_vector_of_data().first});
          number_of_columns += TensorType::size();
        }
      }
    });
    if (not found) {
      ERROR("The tensor '" << name << "' can't be evaluated on the device.");
    }
  }

  const size_t number_of_elements = sources.number_of_elements();
  const size_t number_of_grid_points = sources.number_of_grid_points();
  const auto& statistics =
      sources.scratch().statistics(number_of_elements, number_of_columns + 1);
  const Mode evaluation_mode = mode();
  for (size_t first = 0; first < number_of_elements; first += chunk_size()) {
    const ElementRange range{
        first, std::min(chunk_size(), number_of_elements - first)};
    detail::Evaluator<Sources, Universe> evaluator{exec, sources, range,
                                                   evaluation_mode};
    std::vector<detail::Column> columns{};
    columns.reserve(number_of_columns + 1);
    for (const auto& tensor : result.tensors) {
      tmpl::for_each<ObservableTags>(
          [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
            if constexpr (is_provided<base_of<Tag>, Sources, Universe>) {
              if (db::tag_name<Tag>() == tensor.name and
                  columns.size() == tensor.first_component) {
                const View view = evaluator.template get<base_of<Tag>>();
                for (size_t c = 0; c < view.extent(1); ++c) {
                  const auto component =
                      Kokkos::subview(view, Kokkos::ALL(), c, Kokkos::ALL());
                  columns.push_back({component.data(), component.stride(0),
                                     component.stride(1)});
                }
              }
            }
          });
    }
    // The volume
    columns.push_back({nullptr, 0, 0});
    using DetInvJacobian =
        ::domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>;
    detail::norm_statistics_of_columns(
        exec, statistics, columns, evaluator.template get<DetInvJacobian>(),
        sources.point_weight(), range, number_of_grid_points);
  }
  result.statistics =
      Kokkos::create_mirror_view(Kokkos::HostSpace{}, statistics);
  Kokkos::deep_copy(exec, result.statistics, statistics);
  exec.fence("device_compute::device_norm_statistics");
  return result;
}
}  // namespace evolution::patches::device_compute
#endif  // SPECTRE_KOKKOS
