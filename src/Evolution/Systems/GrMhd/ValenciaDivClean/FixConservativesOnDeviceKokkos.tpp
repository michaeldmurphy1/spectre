// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.hpp"

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"

namespace grmhd::ValenciaDivClean {
namespace FixConservativesOnDevice_detail {
// `FixConservatives::AtGridPoint` with whether to assume a non-zero magnetic
// field given for each element of the batch
struct AtGridPointOfBatch {
  using AtGridPoint = FixConservatives::AtGridPoint;
  using return_tags = AtGridPoint::return_tags;
  using argument_tags = tmpl::push_back<
      tmpl::pop_back<AtGridPoint::argument_tags>,
      ::Tags::PerElement<FixConservatives::AssumeNonZeroMagneticField>>;

  template <typename... Args>
  KOKKOS_FUNCTION static PointwiseStatus apply(const Args&... args) {
    return AtGridPoint::apply(args...);
  }
};

// Whether to assume a non-zero magnetic field on each element, as
// `FixConservatives::assume_non_zero_magnetic_field` decides for an element
Kokkos::View<const bool*> assume_non_zero_magnetic_field_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const typename ::Tags::BatchView<Tags::TildeB<>>::type& tilde_b,
    const FixConservatives& fix_conservatives, Kokkos::View<bool*>* scratch);

// Fixes with a `PointwiseStatusSummary` result if `status` is `nullptr`, and
// else records the status in `*status`
template <typename BatchTags, typename StatusPointer>
PointwiseStatusSummary fix_conservatives_impl(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<BatchTags>*> conservatives,
    const StatusPointer status,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* const per_element_scratch) {
  const auto& tilde_b = get<::Tags::BatchView<Tags::TildeB<>>>(*conservatives);
  const Kokkos::View<const bool*> assume_non_zero_magnetic_field =
      assume_non_zero_magnetic_field_on_device(exec, tilde_b, fix_conservatives,
                                               per_element_scratch);
  // The variables are both arguments and results. This is safe because the
  // arguments at a grid point are read before the results there are written.
  return ApplyPointwiseOnDevice_detail::apply_with_status<AtGridPointOfBatch>(
      exec, conservatives, status, AtGridPointOfBatch::return_tags{},
      AtGridPointOfBatch::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, get<::Tags::BatchView<Tags::TildeD>>(*conservatives),
            get<::Tags::BatchView<Tags::TildeYe>>(*conservatives),
            get<::Tags::BatchView<Tags::TildeTau>>(*conservatives),
            get<::Tags::BatchView<Tags::TildeS<>>>(*conservatives), tilde_b,
            spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
            fix_conservatives, assume_non_zero_magnetic_field);
      }(AtGridPointOfBatch::argument_tags{}));
}

}  // namespace FixConservativesOnDevice_detail

template <typename BatchTags>
PointwiseStatusSummary fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<BatchTags>*> conservatives,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* const per_element_scratch) {
  return FixConservativesOnDevice_detail::fix_conservatives_impl(
      exec, conservatives, nullptr, spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, fix_conservatives, per_element_scratch);
}

template <typename BatchTags>
void fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<BatchTags>*> conservatives,
    const gsl::not_null<
        DeferredPointwiseStatus<fix_conservatives_snapshot_tags>*>
        status,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* const per_element_scratch) {
  FixConservativesOnDevice_detail::fix_conservatives_impl(
      exec, conservatives, status.get(), spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, fix_conservatives, per_element_scratch);
}

}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
