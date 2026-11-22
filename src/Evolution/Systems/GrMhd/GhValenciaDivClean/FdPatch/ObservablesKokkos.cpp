// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/ObservablesKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintGammasKokkos.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ThreePlusOneKokkos.hpp"

namespace grmhd::GhValenciaDivClean::fd_patch::ObservablesKokkos_detail {
namespace {
constexpr size_t dim = 3;
template <typename Tag>
using Batch = typename ::Tags::BatchView<Tag>::type;
template <typename TensorType>
using BatchTensor = TensorMetafunctions::swap_type<
    Kokkos::View<double**, Kokkos::LayoutStride,
                 typename Kokkos::DefaultExecutionSpace::memory_space>,
    TensorType>;

// The tensor `TensorType` in the components of `view`
template <typename TensorType>
BatchTensor<TensorType> as_batch(const View& view) {
  BatchTensor<TensorType> result{};
  for (size_t c = 0; c < result.size(); ++c) {
    result[c] = Kokkos::subview(view, Kokkos::ALL(), c, Kokkos::ALL());
  }
  return result;
}

struct DetSpatialMetricKernel {
  Batch<gr::Tags::SpatialMetric<DataVector, dim>> spatial_metric;
  View result;
  size_t number_of_points;

  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const size_t i = k % number_of_points;
    const size_t e = k / number_of_points;
    tnsr::ii<double, dim> metric{};
    for (size_t c = 0; c < metric.size(); ++c) {
      metric[c] = spatial_metric[c](i, e);
    }
    result(i, 0, e) = gr::device::determinant_and_inverse(metric).determinant;
  }
};
}  // namespace

void det_spatial_metric(const Kokkos::DefaultExecutionSpace& exec,
                        const View& result, const View& spatial_metric) {
  Kokkos::parallel_for(
      "GhValenciaDivClean::fd_patch::det_spatial_metric",
      Kokkos::RangePolicy<>(exec, 0, result.extent(0) * result.extent(2)),
      DetSpatialMetricKernel{
          .spatial_metric = as_batch<tnsr::ii<DataVector, dim>>(spatial_metric),
          .result = result,
          .number_of_points = result.extent(0)});
}

void constraint_gammas(
    const Kokkos::DefaultExecutionSpace& exec, const View& gamma0,
    const View& gamma1, const View& gamma2, const View& grid_coordinates,
    const Kokkos::View<const ConstraintDamping::GaussianSum<3>*>&
        damping_functions) {
  gh::constraint_gammas_on_device(
      exec, as_batch<Scalar<DataVector>>(gamma0),
      as_batch<Scalar<DataVector>>(gamma1),
      as_batch<Scalar<DataVector>>(gamma2),
      as_batch<tnsr::I<DataVector, dim, Frame::Grid>>(grid_coordinates),
      damping_functions);
}

}  // namespace grmhd::GhValenciaDivClean::fd_patch::ObservablesKokkos_detail
#endif  // SPECTRE_KOKKOS
