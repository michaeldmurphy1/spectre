// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/// The evolved variables, which hold the boundary corrections computed by
/// `boundary_terms_on_device`.
using BoundaryCorrectionTags =
    tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
               Tags::TildeB<>, Tags::TildePhi>;

/// The normal dot the mesh velocity on the faces of a batch, if the mesh is
/// moving (see `package_data_on_device`).
using NormalDotMeshVelocityBatch =
    Scalar<Kokkos::View<double**, Kokkos::LayoutStride,
                        Kokkos::DefaultExecutionSpace::memory_space>>;

/*!
 * \brief Computes the packaged data of the boundary correction `Correction`
 * (`Rusanov` or `Hll`) on one side of the faces of a batch of FD grids in
 * device memory, like `Correction::dg_package_data` does for each face set, in
 * a single Kokkos kernel.
 *
 * The arguments are the variables on the `faces` (see `fd::face_tags`), with
 * the normal covector `evolution::dg::Tags::NormalCovector` pointing out of
 * the side. `normal_dot_mesh_velocity` is `nullptr` if the mesh isn't moving.
 * Like `Hll::dg_package_data`, `Hll` decides for each face set (each element
 * of the batch) whether to use the hydro characteristic speeds (see
 * `Hll::use_hydro_speeds`), here on the device.
 *
 * The equation of state is evaluated through its `device_view()` (see
 * `EquationsOfState::visit_device_view`). It is an error if it has no
 * `device_view()`.
 *
 * \note Unlike `dg_package_data`, this doesn't return the largest
 * characteristic speed, so it doesn't wait for the kernel, which is only
 * enqueued on `exec`. The FD time derivative (`subcell::TimeDerivative`)
 * discards that speed, as do the other callers of
 * `evolution::dg::Actions::detail::dg_package_data` for ValenciaDivClean. A
 * caller that needs it can reduce the packaged characteristic speeds
 * (`Rusanov::AbsCharSpeed`, or `Hll::LargestOutgoingCharSpeed` and
 * `Hll::LargestIngoingCharSpeed`) on the device.
 */
template <typename Correction>
void package_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<typename Correction::dg_package_field_tags>*>
        packaged_data,
    const BatchedVariables<fd::face_tags>& faces, const Correction& correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const NormalDotMeshVelocityBatch* normal_dot_mesh_velocity = nullptr);

/// @{
/// A covector or a vector on the faces of a batch (see
/// `fluxes_and_package_data_on_device`)
using FaceCovectorBatch =
    tnsr::i<Kokkos::View<double**, Kokkos::LayoutStride,
                         Kokkos::DefaultExecutionSpace::memory_space>,
            3>;
using FaceVectorBatch =
    tnsr::I<Kokkos::View<double**, Kokkos::LayoutStride,
                         Kokkos::DefaultExecutionSpace::memory_space>,
            3>;
/// @}

/*!
 * \brief Computes the packaged data of the boundary correction `Correction`
 * (`Rusanov` or `Hll`) on one side of the faces of a batch of FD grids in
 * device memory, starting from the conserved and primitive variables on the
 * faces, in a single Kokkos kernel.
 *
 * \details At every grid point this computes, keeping all intermediate
 * results on the stack:
 * 1. the fluxes (see `ComputeFluxes`),
 * 2. if the mesh is moving, the fluxes minus the `mesh_velocity` times the
 *    evolved variables,
 * 3. the normal covector, i.e. `sign` times the `conormal` normalized with the
 *    inverse spatial metric (`sign` is \f$+1\f$ on the lower side of the faces
 *    and \f$-1\f$ on the upper side, so that the normal points out of the
 *    side),
 * 4. if the mesh is moving, the normal dot the mesh velocity, and
 * 5. the packaged data, as `package_data_on_device` does from these.
 *
 * This is equivalent to computing these steps in separate kernels and passing
 * the results to `package_data_on_device`, with the same operations at each
 * grid point, but it doesn't write the fluxes and the normals to memory. The
 * `faces` (`fd::face_tags` or `fd::device_face_tags`) must hold the conserved
 * variables, the primitive variables and the spacetime variables (e.g. from
 * `fd::compute_conservatives_on_device`); fluxes and normals in them are
 * neither used nor set. `mesh_velocity` is `nullptr` if
 * the mesh isn't moving.
 *
 * The kernel is only enqueued on `exec`, except for `Hll`, which first decides
 * for each face set whether to use the hydro characteristic speeds (see
 * `package_data_on_device`), also on the device.
 */
template <typename Correction, typename FaceTags>
void fluxes_and_package_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<typename Correction::dg_package_field_tags>*>
        packaged_data,
    const BatchedVariables<FaceTags>& faces, const FaceCovectorBatch& conormal,
    double sign, const FaceVectorBatch* mesh_velocity,
    const Correction& correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state);

/*!
 * \brief Computes the packaged data of `Rusanov` on one side of the faces of a
 * batch of FD grids in device memory, starting from the reconstructed
 * primitive variables, in a single Kokkos kernel.
 *
 * \details At every grid point this computes, keeping all intermediate
 * results on the stack, the variables on the face as
 * `fd::compute_conservatives_on_device` does (with the fix to the atmosphere
 * if `fix_to_atmosphere` isn't `nullptr`), and then the fluxes, the normals
 * and the packaged data as `fluxes_and_package_data_on_device` does. The
 * `faces` hold the reconstructed and spacetime variables
 * (`fd::reconstruction_face_tags`) and aren't modified.
 *
 * As `fd::compute_conservatives_on_device`, the first overload returns the
 * number of grid points where the root find of the fix to the atmosphere
 * failed and the first one (see `PointwiseStatusSummary`), so it waits for the
 * kernel on `exec`, and the second records them in `status` (see
 * `DeferredPointwiseStatus`) and only enqueues the kernel. The packaged data
 * at failed grid points isn't computed. Report a failure with
 * `VariableFixing::error_fixing_to_atmosphere_on_device`, on the `faces`
 * (which hold the reconstructed variables there) or the `status`.
 *
 * Only `Rusanov` is supported: `Hll` decides for each face set from the fixed
 * variables on all its faces whether to use the hydro characteristic speeds,
 * so it needs those variables first (see `fluxes_and_package_data_on_device`).
 */
/// @{
PointwiseStatusSummary package_data_from_reconstruction_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<Rusanov::dg_package_field_tags>*>
        packaged_data,
    const BatchedVariables<fd::reconstruction_face_tags>& faces,
    const FaceCovectorBatch& conormal, double sign,
    const FaceVectorBatch* mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* fix_to_atmosphere);

void package_data_from_reconstruction_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<Rusanov::dg_package_field_tags>*>
        packaged_data,
    gsl::not_null<DeferredPointwiseStatus<fd::face_atmosphere_snapshot_tags>*>
        status,
    const BatchedVariables<fd::reconstruction_face_tags>& faces,
    const FaceCovectorBatch& conormal, double sign,
    const FaceVectorBatch* mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* fix_to_atmosphere);
/// @}

/*!
 * \brief Computes the boundary terms of the boundary correction `Correction`
 * (`Rusanov` or `Hll`) on the faces of a batch of FD grids in device memory
 * from the packaged data on both sides, like `Correction::dg_boundary_terms`
 * does for each face set, in a single Kokkos kernel.
 *
 * The kernel is only enqueued on `exec`.
 */
template <typename Correction>
void boundary_terms_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<BoundaryCorrectionTags>*>
        boundary_corrections,
    const BatchedVariables<typename Correction::dg_package_field_tags>&
        interior_packaged_data,
    const BatchedVariables<typename Correction::dg_package_field_tags>&
        exterior_packaged_data,
    dg::Formulation dg_formulation);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
#endif  // SPECTRE_KOKKOS
