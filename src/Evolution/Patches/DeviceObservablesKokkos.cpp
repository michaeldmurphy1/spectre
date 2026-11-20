// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/DeviceObservablesKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <cmath>
#include <cstddef>
#include <limits>

#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace evolution::patches {
namespace {
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
    value.max = std::numeric_limits<double>::lowest();
    value.min = std::numeric_limits<double>::max();
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
}  // namespace

Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight, Kokkos::HostSpace>
norm_statistics_on_device(const Kokkos::DefaultExecutionSpace& exec,
                          const DeviceObservables& observables) {
  const Kokkos::Profiling::ScopedRegion region{
      "evolution::patches::norm_statistics_on_device"};
  const auto& values = observables.values;
  const auto& det_inverse_jacobian = observables.det_inverse_jacobian;
  const size_t number_of_points = values.extent(0);
  const size_t number_of_components = values.extent(1);
  const size_t number_of_elements = values.extent(2);
  ASSERT(det_inverse_jacobian.extent(0) == number_of_points and
             det_inverse_jacobian.extent(1) == number_of_elements,
         "The inverse Jacobian determinant has "
             << det_inverse_jacobian.extent(0) << " points and "
             << det_inverse_jacobian.extent(1)
             << " elements, but the observables have " << number_of_points
             << " points and " << number_of_elements << " elements.");
  const double point_weight = observables.point_weight;
  // The extra component is the constant 1, for the volume
  const size_t number_of_reductions = number_of_components + 1;
  const Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight> result{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "norm_statistics_on_device"),
      number_of_elements, number_of_reductions};

  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  Kokkos::parallel_for(
      "evolution::patches::norm_statistics_on_device",
      TeamPolicy(exec,
                 static_cast<int>(number_of_elements * number_of_reductions),
                 Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const size_t e =
            static_cast<size_t>(team.league_rank()) / number_of_reductions;
        const size_t c =
            static_cast<size_t>(team.league_rank()) % number_of_reductions;
        Events::NormStatistics statistics{};
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, number_of_points),
            [&](const size_t i, Events::NormStatistics& update) {
              const double value =
                  c < number_of_components ? values(i, c, e) : 1.0;
              const double weight = point_weight / det_inverse_jacobian(i, e);
              const double abs_value = Kokkos::abs(value);
              update.max = Kokkos::max(update.max, value);
              update.min = Kokkos::min(update.min, value);
              update.sum_abs += abs_value;
              update.sum_square += value * value;
              update.integral_abs += weight * abs_value;
              update.integral_square += weight * value * value;
              update.integral += weight * value;
            },
            NormStatisticsReducer(statistics));
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { result(e, c) = statistics; });
      });
  const auto host_result =
      Kokkos::create_mirror_view(Kokkos::HostSpace{}, result);
  Kokkos::deep_copy(exec, host_result, result);
  exec.fence("evolution::patches::norm_statistics_on_device");
  return host_result;
}
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
