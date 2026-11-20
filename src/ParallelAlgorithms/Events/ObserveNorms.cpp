// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveNorms.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"

namespace Events::ObserveNorms_impl {
void check_norm_is_observable(const std::string& tensor_name,
                              const bool tag_has_value) {
  if (UNLIKELY(not tag_has_value)) {
    ERROR("Cannot observe a norm of '"
          << tensor_name
          << "' because it is a std::optional and wasn't able to be "
             "computed. This can happen when you try to observe errors "
             "without an analytic solution.");
  }
}

namespace {
// Fills the norm of each component of the tensor (`Individual`) or of the
// whole tensor (`Sum`), where `component_norm(storage_index)` is the norm
// `tensor_norm_type` of a component
template <typename ComponentNorm>
void fill_norm_values_and_names_impl(
    const gsl::not_null<std::unordered_map<
        std::string, std::pair<std::vector<double>, std::vector<std::string>>>*>
        norm_values_and_names,
    const std::vector<std::string>& component_names,
    const std::string& tensor_name, const std::string& tensor_norm_type,
    const std::string& tensor_component, const ComponentNorm& component_norm) {
  auto& [values, names] = (*norm_values_and_names)[tensor_norm_type];
  if (tensor_component == "Individual") {
    for (size_t storage_index = 0; storage_index < component_names.size();
         ++storage_index) {
      values.push_back(component_norm(storage_index));
      names.push_back(
          tensor_norm_type + "(" +
          (component_names.size() == 1
               ? tensor_name
               : (tensor_name + "_" + component_names[storage_index])) +
          ")");
    }
  } else if (tensor_component == "Sum") {
    double value = 0.0;
    if (tensor_norm_type == "Max") {
      value = std::numeric_limits<double>::lowest();
    } else if (tensor_norm_type == "Min") {
      value = std::numeric_limits<double>::max();
    }
    for (size_t storage_index = 0; storage_index < component_names.size();
         ++storage_index) {
      if (tensor_norm_type == "Max") {
        value = std::max(value, component_norm(storage_index));
      } else if (tensor_norm_type == "Min") {
        value = std::min(value, component_norm(storage_index));
      } else {
        value += component_norm(storage_index);
      }
    }

    names.push_back(tensor_norm_type + "(" + tensor_name + ")");
    values.push_back(value);
  }
}
}  // namespace

template <size_t Dim>
void fill_norm_values_and_names(
    const gsl::not_null<std::unordered_map<
        std::string, std::pair<std::vector<double>, std::vector<std::string>>>*>
        norm_values_and_names,
    const std::pair<std::vector<std::string>, std::vector<DataVector>>&
        names_and_components,
    const Mesh<Dim>& mesh, const DataVector& det_jacobian,
    const std::string& tensor_name, const std::string& tensor_norm_type,
    const std::string& tensor_component, const size_t number_of_points) {
  const auto& components = names_and_components.second;
  if (components[0].size() != number_of_points) {
    ERROR("The number of grid points of the mesh is "
          << number_of_points << " but the tensor '" << tensor_name << "' has "
          << components[0].size()
          << " points. This means you're computing norms of tensors over "
             "different grids, which will give the wrong answer for "
             "norms that use the grid points.");
  }

  using std::abs;
  fill_norm_values_and_names_impl(
      norm_values_and_names, names_and_components.first, tensor_name,
      tensor_norm_type, tensor_component,
      [&](const size_t storage_index) -> double {
        const DataVector& component = components[storage_index];
        if (tensor_norm_type == "Max") {
          return max(component);
        } else if (tensor_norm_type == "Min") {
          return min(component);
        } else if (tensor_norm_type == "L1Norm") {
          return alg::accumulate(abs(component), 0.0);
        } else if (tensor_norm_type == "L1IntegralNorm") {
          return definite_integral(abs(component) * det_jacobian, mesh);
        } else if (tensor_norm_type == "L2Norm") {
          return alg::accumulate(square(component), 0.0);
        } else if (tensor_norm_type == "L2IntegralNorm") {
          return definite_integral(square(component) * det_jacobian, mesh);
        } else if (tensor_norm_type == "VolumeIntegral") {
          return definite_integral(component * det_jacobian, mesh);
        }
        ERROR("Unknown norm type " << tensor_norm_type);
      });
}

void fill_norm_values_and_names(
    const gsl::not_null<std::unordered_map<
        std::string, std::pair<std::vector<double>, std::vector<std::string>>>*>
        norm_values_and_names,
    const std::vector<std::string>& component_names,
    const std::vector<NormStatistics>& component_statistics,
    const std::string& tensor_name, const std::string& tensor_norm_type,
    const std::string& tensor_component) {
  if (component_statistics.size() != component_names.size()) {
    ERROR("The tensor '" << tensor_name << "' has " << component_names.size()
                         << " components but norm statistics of "
                         << component_statistics.size());
  }
  fill_norm_values_and_names_impl(
      norm_values_and_names, component_names, tensor_name, tensor_norm_type,
      tensor_component, [&](const size_t storage_index) -> double {
        const NormStatistics& statistics = component_statistics[storage_index];
        if (tensor_norm_type == "Max") {
          return statistics.max;
        } else if (tensor_norm_type == "Min") {
          return statistics.min;
        } else if (tensor_norm_type == "L1Norm") {
          return statistics.sum_abs;
        } else if (tensor_norm_type == "L1IntegralNorm") {
          return statistics.integral_abs;
        } else if (tensor_norm_type == "L2Norm") {
          return statistics.sum_square;
        } else if (tensor_norm_type == "L2IntegralNorm") {
          return statistics.integral_square;
        } else if (tensor_norm_type == "VolumeIntegral") {
          return statistics.integral;
        }
        ERROR("Unknown norm type " << tensor_norm_type);
      });
}

std::pair<std::vector<std::string>, std::vector<DataVector>>
split_complex_vector_of_data(
    std::pair<std::vector<std::string>, std::vector<DataVector>>&&
        names_and_components) {
  return names_and_components;
}

std::pair<std::vector<std::string>, std::vector<DataVector>>
split_complex_vector_of_data(
    const std::pair<std::vector<std::string>, std::vector<ComplexDataVector>>&
        names_and_components) {
  const auto& [names, components] = names_and_components;
  std::vector<std::string> result_names{};
  std::vector<DataVector> result_components{};
  result_names.reserve(2 * names.size());
  result_components.reserve(2 * names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    result_names.emplace_back("Re(" + names[i] + ")");
    result_components.emplace_back(real(components[i]));
    result_names.emplace_back("Im(" + names[i] + ")");
    result_components.emplace_back(imag(components[i]));
  }
  return std::make_pair(std::move(result_names), std::move(result_components));
}

}  // namespace Events::ObserveNorms_impl

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template void Events::ObserveNorms_impl::fill_norm_values_and_names(     \
      gsl::not_null<std::unordered_map<                                    \
          std::string,                                                     \
          std::pair<std::vector<double>, std::vector<std::string>>>*>      \
          norm_values_and_names,                                           \
      const std::pair<std::vector<std::string>, std::vector<DataVector>>&  \
          names_and_components,                                            \
      const Mesh<DIM(data)>& mesh, const DataVector& det_jacobian,         \
      const std::string& tensor_name, const std::string& tensor_norm_type, \
      const std::string& tensor_component, size_t number_of_points);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
