// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/DeviceComputeTagsKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "DataStructures/ApplyMatricesKokkos.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/Patches/DeviceComputeTagsKokkos.tpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::patches::device_compute {
Mode mode() {
  static const Mode result = []() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* const value = std::getenv("SPECTRE_DEVICE_OBSERVABLES");
    if (value == nullptr or std::string{value} == "generic") {
      return Mode::Generic;
    }
    if (std::string{value} == "generic_out_of_line") {
      return Mode::GenericOutOfLine;
    }
    ERROR(
        "SPECTRE_DEVICE_OBSERVABLES must be 'generic' or "
        "'generic_out_of_line', but is '"
        << value << "'.");
    return Mode::Generic;
  }();
  return result;
}

size_t chunk_size() {
  static const size_t result = []() -> size_t {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* const value = std::getenv("SPECTRE_DEVICE_OBSERVABLES_CHUNK");
    if (value == nullptr) {
      return 128;
    }
    const long chunk = std::strtol(value, nullptr, 10);
    if (chunk <= 0) {
      ERROR(
          "SPECTRE_DEVICE_OBSERVABLES_CHUNK must be a positive number of "
          "elements, but is '"
          << value << "'.");
    }
    return static_cast<size_t>(chunk);
  }();
  return result;
}

View Scratch::get(const std::string& name, const size_t number_of_grid_points,
                  const size_t number_of_components,
                  const size_t number_of_elements) {
  auto& view = views_[name];
  if (view.extent(0) != number_of_grid_points or
      view.extent(1) != number_of_components or
      view.extent(2) < number_of_elements) {
    view = Kokkos::View<double***, Kokkos::LayoutLeft>{
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "device_compute " + name),
        number_of_grid_points, number_of_components, number_of_elements};
  }
  return Kokkos::subview(view, Kokkos::ALL(), Kokkos::ALL(),
                         std::pair{size_t{0}, number_of_elements});
}

std::array<DeviceApplyMatrices, 3>& Scratch::logical_derivative_operators(
    const std::string& name, const Mesh<3>& mesh) {
  if (mesh != mesh_) {
    derivative_operators_.clear();
    mesh_ = mesh;
  }
  auto [it, inserted] = derivative_operators_.try_emplace(name);
  if (inserted) {
    for (size_t d = 0; d < 3; ++d) {
      std::array<Matrix, 3> matrices{};
      gsl::at(matrices, d) =
          Spectral::differentiation_matrix(mesh.slice_through(d));
      gsl::at(it->second, d) = DeviceApplyMatrices{matrices, mesh.extents()};
    }
  }
  return it->second;
}

Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight>&
Scratch::statistics(const size_t number_of_elements,
                    const size_t number_of_columns) {
  if (statistics_.extent(0) != number_of_elements or
      statistics_.extent(1) != number_of_columns) {
    statistics_ = Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight>{
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "device_compute statistics"),
        number_of_elements, number_of_columns};
  }
  return statistics_;
}

namespace detail {
void norm_statistics_of_columns(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight>&
        statistics,
    const std::vector<Column>& columns, const View& det_inverse_jacobian,
    const double point_weight, const ElementRange& range,
    const size_t number_of_grid_points) {
  const Kokkos::Profiling::ScopedRegion region{
      "device_compute::norm_statistics"};
  const size_t number_of_columns = columns.size();
  const Kokkos::View<Column*> device_columns{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "device_compute columns"),
      number_of_columns};
  const Kokkos::View<const Column*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      host_columns{columns.data(), number_of_columns};
  Kokkos::deep_copy(exec, device_columns, host_columns);
  const size_t first_element = range.first;
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  Kokkos::parallel_for(
      "device_compute::norm_statistics",
      TeamPolicy(exec, static_cast<int>(range.size * number_of_columns),
                 Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const size_t e =
            static_cast<size_t>(team.league_rank()) / number_of_columns;
        const size_t c =
            static_cast<size_t>(team.league_rank()) % number_of_columns;
        const Column column = device_columns(c);
        Events::NormStatistics result{};
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, number_of_grid_points),
            [&](const size_t i, Events::NormStatistics& update) {
              const double value =
                  column.data == nullptr
                      ? 1.0
                      : column.data[i * column.point_stride +  // NOLINT
                                    e * column.element_stride];
              const double weight =
                  point_weight / det_inverse_jacobian(i, 0, e);
              const double abs_value = Kokkos::abs(value);
              update.max = Kokkos::max(update.max, value);
              update.min = Kokkos::min(update.min, value);
              update.sum_abs += abs_value;
              update.sum_square += value * value;
              update.integral_abs += weight * abs_value;
              update.integral_square += weight * value * value;
              update.integral += weight * value;
            },
            NormStatisticsReducer(result));
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { statistics(first_element + e, c) = result; });
      });
  // The host columns are freed when the caller's vector goes away
  exec.fence("device_compute::norm_statistics");
}
}  // namespace detail
}  // namespace evolution::patches::device_compute
#endif  // SPECTRE_KOKKOS
