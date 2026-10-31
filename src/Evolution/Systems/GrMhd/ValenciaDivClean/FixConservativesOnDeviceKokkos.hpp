// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace grmhd::ValenciaDivClean {
class FixConservatives;
}  // namespace grmhd::ValenciaDivClean
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean {
/// The evolved variables of a batch whose conservative variables
/// `fix_conservatives_on_device` fixes.
using FixConservativesOnDeviceTags =
    tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
               Tags::TildeB<>, Tags::TildePhi>;

/*!
 * \brief Fixes the conservative variables of a batch of elements whose data is
 * in device memory, like `FixConservatives` does for each element, in a
 * single Kokkos kernel.
 *
 * `FixConservatives::AtGridPoint` is applied at every grid point. Like
 * `FixConservatives`, whether to assume a non-zero magnetic field (see
 * `FixConservatives::assume_non_zero_magnetic_field`) is decided for each
 * element, here on the device.
 *
 * Unlike `FixConservatives`, a failed root find for the Lorentz factor is
 * never an error, and the variables at grid points where it failed are left
 * unfixed. This returns whether any variables were fixed (`any_flag`) and the
 * number of failed grid points and the element and grid point of the first
 * one (see `PointwiseStatusSummary`). Report a failure with
 * `error_fixing_conservatives_on_device`.
 *
 * The `conservatives` are fixed in place. Nothing is copied between the host
 * and the device, but since the result is needed on the host, this function
 * waits for the kernels on `exec` to complete.
 */
PointwiseStatusSummary fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<FixConservativesOnDeviceTags>*>
        conservatives,
    const typename ::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>::
        type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives);

/*!
 * \brief ERRORs with the variables at the first grid point where
 * `fix_conservatives_on_device` failed to find the Lorentz factor.
 *
 * The arguments are those passed to `fix_conservatives_on_device` and its
 * result `status`, which must have failures. The variables there are still
 * unfixed. Only that grid point is copied from the device. The
 * `element_description` identifies the element in the error, e.g. its
 * `ElementId`.
 */
[[noreturn]] void error_fixing_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const PointwiseStatusSummary& status,
    const BatchedVariables<FixConservativesOnDeviceTags>& conservatives,
    const typename ::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>::
        type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const std::string& element_description);
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
