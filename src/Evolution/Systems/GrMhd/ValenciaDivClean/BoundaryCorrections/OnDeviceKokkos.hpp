// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
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
