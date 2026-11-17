// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
class DataVector;
/// \endcond

namespace grmhd::GhValenciaDivClean::fd {
/*!
 * \brief The lapse, shift, spatial metric, square root of its determinant and
 * inverse spatial metric from the spacetime metric on one side of the faces of
 * a batch of elements, in a Kokkos kernel on `exec`.
 *
 * \details This computes the same values as the reconstructors of
 * `grmhd::GhValenciaDivClean::fd` do on the host from the reconstructed
 * spacetime metric (`gr::spatial_metric`, `determinant_and_inverse`,
 * `gr::shift` and `gr::lapse`), with the same operations in the same order.
 *
 * All arguments are batches with the same number of grid points and elements,
 * e.g. the tags of a `BatchedVariables` of face variables.
 */
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
        gr::Tags::SpacetimeMetric<DataVector, 3>>::type& spacetime_metric);
}  // namespace grmhd::GhValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
