// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/SpacetimeOnFacesKokkos.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ThreePlusOneKokkos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::GhValenciaDivClean::fd {
void spacetime_on_faces_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const typename ::Tags::BatchView<gr::Tags::Lapse<DataVector>>::type& lapse,
    const typename ::Tags::BatchView<gr::Tags::Shift<DataVector, 3>>::type&
        shift,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inverse_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SpacetimeMetric<DataVector, 3>>::type& spacetime_metric) {
  const size_t number_of_points = get<0, 0>(spacetime_metric).extent(0);
  const size_t number_of_elements = get<0, 0>(spacetime_metric).extent(1);
#ifdef SPECTRE_DEBUG
  const auto check_shape = [&](const auto& tensor) {
    for (const auto& component : tensor) {
      ASSERT(component.extent(0) == number_of_points and
                 component.extent(1) == number_of_elements,
             "All batches must have "
                 << number_of_points << " grid points and "
                 << number_of_elements << " elements, but one has "
                 << component.extent(0) << " and " << component.extent(1));
    }
  };
  check_shape(lapse);
  check_shape(shift);
  check_shape(spatial_metric);
  check_shape(sqrt_det_spatial_metric);
  check_shape(inverse_spatial_metric);
  check_shape(spacetime_metric);
#endif  // SPECTRE_DEBUG
  // The components as arrays of views, since tensors of views aren't
  // device-callable
  using View = Kokkos::View<double**, Kokkos::LayoutStride>;
  const View lapse_view = get(lapse);
  const View sqrt_det_view = get(sqrt_det_spatial_metric);
  Kokkos::Array<View, 3> shift_views{};
  Kokkos::Array<View, 6> spatial_metric_views{};
  Kokkos::Array<View, 6> inverse_views{};
  Kokkos::Array<View, 10> spacetime_metric_views{};
  for (size_t i = 0; i < 3; ++i) {
    shift_views[i] = shift.get(i);
  }
  for (size_t component = 0; component < 6; ++component) {
    spatial_metric_views[component] = spatial_metric[component];
    inverse_views[component] = inverse_spatial_metric[component];
  }
  for (size_t component = 0; component < 10; ++component) {
    spacetime_metric_views[component] = spacetime_metric[component];
  }
  Kokkos::parallel_for(
      "grmhd::GhValenciaDivClean::fd::spacetime_on_faces_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>,
                            Kokkos::IndexType<size_t>>(
          exec, {0, 0}, {number_of_points, number_of_elements}),
      KOKKOS_LAMBDA(const size_t point, const size_t element) {
        tnsr::aa<double, 3> g{};
        for (size_t component = 0; component < 10; ++component) {
          g[component] = spacetime_metric_views[component](point, element);
        }
        const tnsr::ii<double, 3> local_spatial_metric =
            gr::device::spatial_metric(g);
        const auto [det, inverse] =
            gr::device::determinant_and_inverse(local_spatial_metric);
        const tnsr::I<double, 3> local_shift = gr::device::shift(g, inverse);

        lapse_view(point, element) = gr::device::lapse(g, local_shift);
        sqrt_det_view(point, element) = sqrt(det);
        for (size_t i = 0; i < 3; ++i) {
          shift_views[i](point, element) = local_shift[i];
        }
        for (size_t component = 0; component < 6; ++component) {
          spatial_metric_views[component](point, element) =
              local_spatial_metric[component];
          inverse_views[component](point, element) = inverse[component];
        }
      });
}
}  // namespace grmhd::GhValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
