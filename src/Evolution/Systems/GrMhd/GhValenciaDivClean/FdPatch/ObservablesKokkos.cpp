// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/ObservablesKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Expressions/AtPointKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/PatchKokkos.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::fd_patch {
namespace {
constexpr size_t dim = 3;
template <typename Tag>
using Batch = typename ::Tags::BatchView<Tag>::type;
// A tensor whose components are components of the `DeviceObservables`
template <typename TensorType>
using OutputTensor =
    TensorMetafunctions::swap_type<Kokkos::View<double**, Kokkos::LayoutStride>,
                                   TensorType>;

using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, dim>;
using Pi = gh::Tags::Pi<DataVector, dim>;
using Phi = gh::Tags::Phi<DataVector, dim>;
using InverseSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, dim>;
using GaugeConstraint =
    gh::Tags::GaugeConstraint<DataVector, dim, Frame::Inertial>;
using ThreeIndexConstraint =
    gh::Tags::ThreeIndexConstraint<DataVector, dim, Frame::Inertial>;
using InverseJacobian =
    ::domain::Tags::InverseJacobian<dim, Frame::ElementLogical,
                                    Frame::Inertial>;
using DetInvJacobian =
    ::domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>;
// The logical derivatives of the spacetime metric, as `partial_derivatives`
using LogicalDerivMetric =
    TensorMetafunctions::prepend_spatial_index<tnsr::aa<DataVector, dim>, dim,
                                               UpLo::Lo, Frame::ElementLogical>;

// The tensors computed from the variables, rather than copied
using computed_tags =
    tmpl::list<GaugeConstraint, ::Tags::PointwiseL2Norm<GaugeConstraint>,
               ThreeIndexConstraint,
               ::Tags::PointwiseL2Norm<ThreeIndexConstraint>>;

template <typename PrimitiveTagsList>
using copied_tags =
    tmpl::append<subcell::evolved_tags_on_device, PrimitiveTagsList>;

// The pointwise L2 norm of a tensor, as `pointwise_l2_norm`
template <typename TensorType, size_t... StorageIndices>
KOKKOS_FUNCTION double norm_at_point_impl(
    const TensorType& tensor, std::index_sequence<StorageIndices...> /*meta*/) {
  double result = 0.0;
  ((result +=
    static_cast<double>(TensorType::structure::multiplicity(StorageIndices)) *
    tensor[StorageIndices] * tensor[StorageIndices]),
   ...);
  return sqrt(result);
}

template <typename TensorType>
KOKKOS_FUNCTION double norm_at_point(const TensorType& tensor) {
  return norm_at_point_impl(tensor,
                            std::make_index_sequence<TensorType::size()>{});
}

// The gauge constraint and its pointwise L2 norm, as `gh::gauge_constraint`
// with the harmonic gauge and the compute tags of an element
struct GaugeConstraintKernel {
  Batch<SpacetimeMetric> spacetime_metric;
  Batch<Pi> pi;
  Batch<Phi> phi;
  Batch<InverseSpatialMetric> inverse_spatial_metric;
  bool store_constraint;
  OutputTensor<tnsr::a<DataVector, dim>> constraint;
  bool store_norm;
  OutputTensor<Scalar<DataVector>> norm;
  size_t number_of_points;

  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const size_t i = k % number_of_points;
    const size_t e = k / number_of_points;
    const auto g = tenex::at_point(spacetime_metric, i, e);
    const auto pi_at_point = tenex::at_point(pi, i, e);
    const auto phi_at_point = tenex::at_point(phi, i, e);
    const auto inv_g3 = tenex::at_point(inverse_spatial_metric, i, e);

    // The 3+1 variables, as `gr::shift`, `gr::lapse`,
    // `gr::inverse_spacetime_metric`, `gr::spacetime_normal_vector` and
    // `gr::spacetime_normal_one_form`
    const auto shift =
        tenex::evaluate<ti::I>(inv_g3(ti::I, ti::J) * g(ti::j, ti::t));
    const double lapse = sqrt(get(
        tenex::evaluate(-g(ti::t, ti::t) + shift(ti::I) * g(ti::i, ti::t))));
    const double one_over_lapse_squared = 1.0 / (lapse * lapse);
    tnsr::AA<double, dim> inverse_spacetime_metric{};
    get<0, 0>(inverse_spacetime_metric) = -one_over_lapse_squared;
    tenex::evaluate<ti::T, ti::I>(make_not_null(&inverse_spacetime_metric),
                                  shift(ti::I) * one_over_lapse_squared);
    tenex::evaluate<ti::I, ti::J>(
        make_not_null(&inverse_spacetime_metric),
        inv_g3(ti::I, ti::J) -
            shift(ti::I) * shift(ti::J) * one_over_lapse_squared);
    tnsr::A<double, dim> normal_vector{};
    get<0>(normal_vector) = 1.0 / lapse;
    tenex::evaluate<ti::I>(make_not_null(&normal_vector),
                           -shift(ti::I) / lapse);
    tnsr::a<double, dim> normal_one_form{};
    get<0>(normal_one_form) = -lapse;

    // The trace of the Christoffel symbols, as `gh::trace_christoffel`
    const auto common_term = tenex::evaluate(
        -0.5 *
        (pi_at_point(ti::b, ti::c) +
         normal_vector(ti::I) * phi_at_point(ti::i, ti::b, ti::c)) *
        inverse_spacetime_metric(ti::B, ti::C));
    tnsr::a<double, dim> gauge_constraint{};
    tenex::evaluate<ti::a>(
        make_not_null(&gauge_constraint),
        normal_one_form(ti::a) * common_term() +
            inv_g3(ti::I, ti::J) * phi_at_point(ti::i, ti::j, ti::a) +
            normal_vector(ti::B) * pi_at_point(ti::b, ti::a));
    tenex::update<ti::i>(
        make_not_null(&gauge_constraint),
        gauge_constraint(ti::i) - 0.5 * phi_at_point(ti::i, ti::b, ti::c) *
                                      inverse_spacetime_metric(ti::B, ti::C));
    // The harmonic gauge source function vanishes (patches only support the
    // harmonic gauge, see `fd_patch::InitializePatch`)

    if (store_constraint) {
      tenex::evaluate<ti::a>(tenex::at_point(constraint, i, e),
                             gauge_constraint(ti::a));
    }
    if (store_norm) {
      get(norm)(i, e) = norm_at_point(gauge_constraint);
    }
  }
};

// The three-index constraint and its pointwise L2 norm, as
// `gh::three_index_constraint` with the spatial derivatives of the spacetime
// metric of `partial_derivatives`
struct ThreeIndexConstraintKernel {
  TensorMetafunctions::swap_type<Kokkos::View<double**, Kokkos::LayoutStride>,
                                 LogicalDerivMetric>
      logical_deriv_metric;
  Batch<InverseJacobian> inverse_jacobian;
  Batch<Phi> phi;
  bool store_constraint;
  OutputTensor<tnsr::iaa<DataVector, dim>> constraint;
  bool store_norm;
  OutputTensor<Scalar<DataVector>> norm;
  size_t number_of_points;

  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const size_t i = k % number_of_points;
    const size_t e = k / number_of_points;
    const auto logical_deriv = tenex::at_point(logical_deriv_metric, i, e);
    const auto inv_jacobian = tenex::at_point(inverse_jacobian, i, e);
    const auto phi_at_point = tenex::at_point(phi, i, e);
    tnsr::iaa<double, dim> three_index_constraint{};
    tenex::evaluate<ti::i, ti::a, ti::b>(
        make_not_null(&three_index_constraint),
        inv_jacobian(ti::J, ti::i) * logical_deriv(ti::j, ti::a, ti::b) -
            phi_at_point(ti::i, ti::a, ti::b));
    if (store_constraint) {
      tenex::evaluate<ti::i, ti::a, ti::b>(
          tenex::at_point(constraint, i, e),
          three_index_constraint(ti::i, ti::a, ti::b));
    }
    if (store_norm) {
      get(norm)(i, e) = norm_at_point(three_index_constraint);
    }
  }
};

template <typename TensorType>
std::vector<std::string> component_names() {
  return TensorType{}.get_vector_of_data().first;
}

template <typename TensorType>
OutputTensor<TensorType> output_tensor(
    const Kokkos::View<double***, Kokkos::LayoutLeft>& values,
    const size_t first_component) {
  OutputTensor<TensorType> result{};
  for (size_t c = 0; c < result.size(); ++c) {
    result[c] = Kokkos::subview(values, Kokkos::ALL(), first_component + c,
                                Kokkos::ALL());
  }
  return result;
}

// The tensor `Tag` of the observables, or a tensor of empty views if it isn't
// observed
template <typename Tag>
std::pair<bool, OutputTensor<typename Tag::type>> observed_tensor(
    const evolution::patches::DeviceObservables& observables) {
  const std::string name = db::tag_name<Tag>();
  for (const auto& tensor : observables.tensors) {
    if (tensor.name == name) {
      return {true, output_tensor<typename Tag::type>(observables.values,
                                                      tensor.first_component)};
    }
  }
  return {false, {}};
}

template <typename Tag>
bool is_observed(const evolution::patches::DeviceObservables& observables) {
  return alg::any_of(observables.tensors, [](const auto& tensor) {
    return tensor.name == db::tag_name<Tag>();
  });
}

// The spatial derivatives of the spacetime metric in the logical frame, from
// the matrices that `partial_derivatives` uses on the host
TensorMetafunctions::swap_type<Kokkos::View<double**, Kokkos::LayoutStride>,
                               LogicalDerivMetric>
logical_deriv_metric(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<DeviceObservablesBuffers*> buffers,
    const BatchedVariables<subcell::evolved_tags_on_device>& evolved_variables,
    const Mesh<dim>& subcell_mesh) {
  constexpr size_t number_of_metric_components =
      tnsr::aa<DataVector, dim>::size();
  if (buffers->mesh != subcell_mesh) {
    static_assert(
        std::is_same_v<tmpl::front<subcell::evolved_tags_on_device>,
                       SpacetimeMetric>,
        "The spacetime metric must be the first tensor of the evolved "
        "variables.");
    std::vector<size_t> metric_components(number_of_metric_components);
    for (size_t c = 0; c < number_of_metric_components; ++c) {
      metric_components[c] = c;
    }
    for (size_t d = 0; d < dim; ++d) {
      std::array<Matrix, dim> matrices{};
      gsl::at(matrices, d) =
          Spectral::differentiation_matrix(subcell_mesh.slice_through(d));
      gsl::at(buffers->logical_derivative_operators, d) = DeviceApplyMatrices{
          matrices, subcell_mesh.extents(), metric_components};
    }
    buffers->mesh = subcell_mesh;
  }
  const size_t number_of_points = evolved_variables.number_of_grid_points();
  const size_t number_of_elements = evolved_variables.number_of_elements();
  auto& derivatives = buffers->logical_derivatives;
  if (derivatives.extent(0) != number_of_points or
      derivatives.extent(2) != number_of_elements) {
    derivatives = Kokkos::View<double***, Kokkos::LayoutLeft>{
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                           "GhValenciaDivClean logical derivatives"),
        number_of_points, dim * number_of_metric_components,
        number_of_elements};
  }
  for (size_t d = 0; d < dim; ++d) {
    gsl::at(buffers->logical_derivative_operators, d)
        .apply(exec,
               Kokkos::subview(derivatives, Kokkos::ALL(),
                               std::pair{d * number_of_metric_components,
                                         (d + 1) * number_of_metric_components},
                               Kokkos::ALL()),
               evolved_variables.view());
  }
  TensorMetafunctions::swap_type<Kokkos::View<double**, Kokkos::LayoutStride>,
                                 LogicalDerivMetric>
      result{};
  for (size_t storage_index = 0; storage_index < result.size();
       ++storage_index) {
    const auto multi_index = result.get_tensor_index(storage_index);
    result[storage_index] =
        Kokkos::subview(derivatives, Kokkos::ALL(),
                        multi_index[0] * number_of_metric_components +
                            tnsr::aa<DataVector, dim>::get_storage_index(
                                std::array{multi_index[1], multi_index[2]}),
                        Kokkos::ALL());
  }
  return result;
}
}  // namespace

template <typename PrimitiveTagsList>
bool is_device_observable(const std::string& name) {
  bool result = false;
  tmpl::for_each<tmpl::append<copied_tags<PrimitiveTagsList>, computed_tags>>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        result = result or db::tag_name<Tag>() == name;
      });
  return result;
}

template <typename PrimitiveTagsList>
void device_observables(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<evolution::patches::DeviceObservables*> observables,
    const gsl::not_null<DeviceObservablesBuffers*> buffers,
    const std::vector<std::string>& tensor_names,
    const BatchedVariables<subcell::evolved_tags_on_device>& evolved_variables,
    const BatchedVariables<PrimitiveTagsList>& primitives,
    const BatchedVariables<subcell::fixing_metric_tags>& fixing_metric,
    const BatchedVariables<
        evolution::dg::subcell::fd_patch::cell_geometry_tags>& cell_geometry,
    const Mesh<3>& subcell_mesh) {
  const Kokkos::Profiling::ScopedRegion region{
      "GhValenciaDivClean::fd_patch::device_observables"};
  const size_t number_of_points = evolved_variables.number_of_grid_points();
  const size_t number_of_elements = evolved_variables.number_of_elements();

  // The layout of the observables: each tensor once, in the order requested
  observables->tensors.clear();
  size_t number_of_components = 0;
  for (const std::string& name : tensor_names) {
    if (alg::any_of(observables->tensors, [&name](const auto& tensor) {
          return tensor.name == name;
        })) {
      continue;
    }
    bool found = false;
    tmpl::for_each<tmpl::append<copied_tags<PrimitiveTagsList>, computed_tags>>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          if (not found and db::tag_name<Tag>() == name) {
            found = true;
            observables->tensors.push_back(
                {name, number_of_components,
                 component_names<typename Tag::type>()});
            number_of_components += Tag::type::size();
          }
        });
    if (not found) {
      ERROR("GhValenciaDivClean FD patches can't compute the observable '"
            << name << "' on the device.");
    }
  }
  auto& values = observables->values;
  if (values.extent(0) != number_of_points or
      values.extent(1) != number_of_components or
      values.extent(2) != number_of_elements) {
    values = Kokkos::View<double***, Kokkos::LayoutLeft>{
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                           "GhValenciaDivClean device observables"),
        number_of_points, number_of_components, number_of_elements};
  }

  // The variables are copied
  std::optional<Kokkos::Profiling::ScopedRegion> copy_region{
      std::in_place, "GhValenciaDivClean::fd_patch::device_observables::copy"};
  const auto copy = [&]<typename Tag>(const auto& batch,
                                      tmpl::type_<Tag> /*meta*/) {
    const auto [observed, destination] = observed_tensor<Tag>(*observables);
    if (observed) {
      const auto& source = get<::Tags::BatchView<Tag>>(batch);
      for (size_t c = 0; c < source.size(); ++c) {
        Kokkos::deep_copy(exec, destination[c], source[c]);
      }
    }
  };
  tmpl::for_each<subcell::evolved_tags_on_device>(
      [&](auto tag_v) { copy(evolved_variables, tag_v); });
  tmpl::for_each<PrimitiveTagsList>(
      [&](auto tag_v) { copy(primitives, tag_v); });
  copy_region.reset();

  const size_t total = number_of_points * number_of_elements;
  if (is_observed<GaugeConstraint>(*observables) or
      is_observed<::Tags::PointwiseL2Norm<GaugeConstraint>>(*observables)) {
    const auto [store_constraint, constraint] =
        observed_tensor<GaugeConstraint>(*observables);
    const auto [store_norm, norm] =
        observed_tensor<::Tags::PointwiseL2Norm<GaugeConstraint>>(*observables);
    Kokkos::parallel_for(
        "GhValenciaDivClean::fd_patch::gauge_constraint",
        Kokkos::RangePolicy<>(exec, 0, total),
        GaugeConstraintKernel{
            .spacetime_metric =
                get<::Tags::BatchView<SpacetimeMetric>>(evolved_variables),
            .pi = get<::Tags::BatchView<Pi>>(evolved_variables),
            .phi = get<::Tags::BatchView<Phi>>(evolved_variables),
            .inverse_spatial_metric =
                get<::Tags::BatchView<InverseSpatialMetric>>(fixing_metric),
            .store_constraint = store_constraint,
            .constraint = constraint,
            .store_norm = store_norm,
            .norm = norm,
            .number_of_points = number_of_points});
  }
  if (is_observed<ThreeIndexConstraint>(*observables) or
      is_observed<::Tags::PointwiseL2Norm<ThreeIndexConstraint>>(
          *observables)) {
    const auto [store_constraint, constraint] =
        observed_tensor<ThreeIndexConstraint>(*observables);
    const auto [store_norm, norm] =
        observed_tensor<::Tags::PointwiseL2Norm<ThreeIndexConstraint>>(
            *observables);
    Kokkos::parallel_for(
        "GhValenciaDivClean::fd_patch::three_index_constraint",
        Kokkos::RangePolicy<>(exec, 0, total),
        ThreeIndexConstraintKernel{
            .logical_deriv_metric = logical_deriv_metric(
                exec, buffers, evolved_variables, subcell_mesh),
            .inverse_jacobian =
                get<::Tags::BatchView<InverseJacobian>>(cell_geometry),
            .phi = get<::Tags::BatchView<Phi>>(evolved_variables),
            .store_constraint = store_constraint,
            .constraint = constraint,
            .store_norm = store_norm,
            .norm = norm,
            .number_of_points = number_of_points});
  }

  observables->det_inverse_jacobian =
      get(get<::Tags::BatchView<DetInvJacobian>>(cell_geometry));
  // The same weight at all cell centers of the FD mesh
  observables->point_weight = 1.0;
  for (size_t d = 0; d < dim; ++d) {
    const DataVector& weights =
        Spectral::quadrature_weights(subcell_mesh.slice_through(d));
    if (not alg::all_of(
            weights, [&weights](const double w) { return w == weights[0]; })) {
      ERROR("The quadrature weights of the FD mesh "
            << subcell_mesh << " must all be the same in dimension " << d
            << ", but are " << weights);
    }
    observables->point_weight *= weights[0];
  }
}

#define INSTANTIATE(PRIMITIVE_TAGS)                                            \
  template bool is_device_observable<PRIMITIVE_TAGS>(const std::string& name); \
  template void device_observables<PRIMITIVE_TAGS>(                            \
      const Kokkos::DefaultExecutionSpace& exec,                               \
      gsl::not_null<evolution::patches::DeviceObservables*> observables,       \
      gsl::not_null<DeviceObservablesBuffers*> buffers,                        \
      const std::vector<std::string>& tensor_names,                            \
      const BatchedVariables<subcell::evolved_tags_on_device>&                 \
          evolved_variables,                                                   \
      const BatchedVariables<PRIMITIVE_TAGS>& primitives,                      \
      const BatchedVariables<subcell::fixing_metric_tags>& fixing_metric,      \
      const BatchedVariables<                                                  \
          evolution::dg::subcell::fd_patch::cell_geometry_tags>&               \
          cell_geometry,                                                       \
      const Mesh<3>& subcell_mesh);

INSTANTIATE(System::primitive_variables_tag::tags_list)
#undef INSTANTIATE
}  // namespace grmhd::GhValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
