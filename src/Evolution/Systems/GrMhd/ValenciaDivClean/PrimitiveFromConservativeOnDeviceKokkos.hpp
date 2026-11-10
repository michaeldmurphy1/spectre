// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace grmhd::ValenciaDivClean {
class PrimitiveFromConservativeOptions;
}  // namespace grmhd::ValenciaDivClean
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean {
/*!
 * \brief Recovers the primitive variables of a batch of elements whose data is
 * in device memory, like `PrimitiveFromConservative` does for each element.
 *
 * All elements are recovered in one Kokkos kernel on the execution space
 * instance `exec`, on data that stays on the device, e.g. the elements of an FD
 * patch.
 *
 * Like `PrimitiveFromConservative`:
 * - The pressure in `primitives` is the initial guess of the recovery schemes
 *   that use one.
 * - If `tilde_d` is below the cutoff for the inversion on the whole grid of an
 *   element, the primitive variables of that element are set to the atmosphere
 *   before the recovery, which also changes the initial guess of the pressure.
 *   This is decided for each element separately.
 *
 * Unlike `PrimitiveFromConservative`:
 * - The EOS is evaluated through its `device_view()` (see
 *   `EquationsOfState::visit_device_view`). It is an error if the EOS has none.
 * - The recovery doesn't stop at the first grid point where all recovery
 *   schemes fail. The primitive variables at such grid points keep their
 *   values from before the recovery.
 * - A failure is never an error. Instead this returns the number of failed
 *   grid points and the element and grid point of the first one (see
 *   `PointwiseStatusSummary`). Report it with
 *   `error_recovering_primitives_on_device`.
 *
 * The `primitives` are overwritten in place. Nothing is copied between the
 * host and the device, but since the failures are needed on the host, this
 * function waits for the kernels on `exec` to complete.
 *
 * `per_element_scratch`, if given, holds the per-element decision whether
 * \f$\tilde{D}\f$ is below the cutoff between calls (see `scratch_view`), so
 * that it isn't allocated every time.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool EnforcePhysicality>
PointwiseStatusSummary primitive_from_conservative_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*> primitives,
    const typename ::Tags::BatchView<Tags::TildeD>::type& tilde_d,
    const typename ::Tags::BatchView<Tags::TildeYe>::type& tilde_ye,
    const typename ::Tags::BatchView<Tags::TildeTau>::type& tilde_tau,
    const typename ::Tags::BatchView<Tags::TildeS<>>::type& tilde_s,
    const typename ::Tags::BatchView<Tags::TildeB<>>::type& tilde_b,
    const typename ::Tags::BatchView<Tags::TildePhi>::type& tilde_phi,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const PrimitiveFromConservativeOptions& primitive_from_conservative_options,
    Kokkos::View<bool*>* per_element_scratch = nullptr);

/*!
 * \brief ERRORs with the variables at the first grid point where
 * `primitive_from_conservative_on_device` failed, like the error of
 * `PrimitiveFromConservative`.
 *
 * The arguments are those passed to `primitive_from_conservative_on_device`
 * and its result `status`, which must have failures. The primitive variables
 * there still hold their values from before the recovery. Only that grid point
 * is copied from the device. The `element_description` identifies the element
 * in the error, e.g. its `ElementId`.
 */
[[noreturn]] void error_recovering_primitives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const PointwiseStatusSummary& status,
    const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives,
    const typename ::Tags::BatchView<Tags::TildeD>::type& tilde_d,
    const typename ::Tags::BatchView<Tags::TildeTau>::type& tilde_tau,
    const typename ::Tags::BatchView<Tags::TildeS<>>::type& tilde_s,
    const typename ::Tags::BatchView<Tags::TildeB<>>::type& tilde_b,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const std::string& element_description);
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
