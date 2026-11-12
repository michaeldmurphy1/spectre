// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
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

/// The snapshot of a `DeferredPointwiseStatus` of
/// `fix_conservatives_on_device`: the variables before fixing
using fix_conservatives_snapshot_tags =
    tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
               Tags::TildeB<>, gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>>;

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
 * unfixed. The first overload returns whether any variables were fixed
 * (`any_flag`) and the number of failed grid points and the element and grid
 * point of the first one (see `PointwiseStatusSummary`), so it waits for the
 * kernels on `exec` to complete. The second records them in `status` (see
 * `DeferredPointwiseStatus`) and only enqueues the kernels. Report a failure
 * with `error_fixing_conservatives_on_device`.
 *
 * The `conservatives` are fixed in place. Nothing is copied between the host
 * and the device.
 *
 * `per_element_scratch`, if given, holds the per-element decision between
 * calls (see `scratch_view`), so that it isn't allocated every time.
 */
/// @{
PointwiseStatusSummary fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<FixConservativesOnDeviceTags>*>
        conservatives,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* per_element_scratch = nullptr);

void fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<FixConservativesOnDeviceTags>*>
        conservatives,
    gsl::not_null<DeferredPointwiseStatus<fix_conservatives_snapshot_tags>*>
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
    Kokkos::View<bool*>* per_element_scratch = nullptr);
/// @}

/// @{
/*!
 * \brief ERRORs with the variables at a grid point where
 * `fix_conservatives_on_device` failed to find the Lorentz factor.
 *
 * In the first overload, the arguments are those passed to
 * `fix_conservatives_on_device` and its result `status`, which must have
 * failures. The variables there are still unfixed. Only that grid point is
 * copied from the device. In the second, `status` is the
 * `DeferredPointwiseStatus` of `fix_conservatives_on_device` after its
 * `copy_to_host()` and a fence, and the variables are its snapshot. The
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

[[noreturn]] void error_fixing_conservatives_on_device(
    const DeferredPointwiseStatus<fix_conservatives_snapshot_tags>& status,
    size_t number_of_grid_points, const std::string& element_description);
/// @}
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
