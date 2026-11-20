// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "DataStructures/ApplyMatricesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/Patches/DeviceObservablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariablesKokkos.hpp"

namespace grmhd::GhValenciaDivClean::fd_patch {
/*!
 * \brief Whether GhValenciaDivClean FD patches can compute the observable
 * tensor named `name` (its `db::tag_name`) on the device (see
 * `device_observables`)
 *
 * \details These are the evolved variables, the primitive variables
 * (`PrimitiveTagsList`), the gauge and three-index constraints of the GH
 * system and their `::Tags::PointwiseL2Norm`s.
 */
template <typename PrimitiveTagsList>
bool is_device_observable(const std::string& name);

/// Buffers of `device_observables` that are kept between calls
struct DeviceObservablesBuffers {
  /// The mesh that the `logical_derivative_operators` are for
  Mesh<3> mesh{};
  /// The operators that compute the logical derivatives of the spacetime
  /// metric in each dimension
  std::array<DeviceApplyMatrices, 3> logical_derivative_operators{};
  /// The logical derivatives of the spacetime metric in the dimensions one
  /// after the other, indexed by grid point, component and element
  Kokkos::View<double***, Kokkos::LayoutLeft> logical_derivatives{};
};

/*!
 * \brief Compute the observable tensors `tensor_names` (see
 * `is_device_observable`) of all elements of a patch on the device
 *
 * \details This computes the same values as the compute tags of the DataBox
 * of an element on the host (see `Patch::make_element_box` and the observed
 * fields of the GhValenciaDivClean executable):
 * - The evolved and primitive variables are copied.
 * - The gauge constraint is computed as `gh::gauge_constraint` with the
 *   harmonic gauge (the only gauge of FD patches), from the lapse, shift and
 *   inverse spacetime metric of the spacetime metric, with the inverse
 *   spatial metric of `fixing_metric` (which the patch computes as the
 *   compute tags of an element).
 * - The three-index constraint uses the spatial derivatives of the spacetime
 *   metric, computed on each element with its finite-difference
 *   differentiation matrices and the inverse Jacobian, as
 *   `::Tags::DerivCompute` with the observer mesh does.
 *
 * The values agree with the host to roundoff: the operations are in a
 * different order. Tensors are only computed once even if they appear several
 * times in `tensor_names`.
 *
 * \param exec the execution space instance that runs the kernels.
 * \param observables the observables, whose memory is reused between calls.
 * \param buffers buffers that are kept between calls.
 * \param tensor_names the names of the observable tensors to compute.
 * \param evolved_variables the evolved variables of the elements.
 * \param primitives the primitive variables of the elements.
 * \param fixing_metric the spatial metric, its inverse and determinant from
 * the spacetime metric of `evolved_variables`.
 * \param cell_geometry the inertial geometry at the cell centers at the time
 * of the variables.
 * \param subcell_mesh the FD mesh of the elements.
 */
template <typename PrimitiveTagsList>
void device_observables(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<evolution::patches::DeviceObservables*> observables,
    gsl::not_null<DeviceObservablesBuffers*> buffers,
    const std::vector<std::string>& tensor_names,
    const BatchedVariables<subcell::evolved_tags_on_device>& evolved_variables,
    const BatchedVariables<PrimitiveTagsList>& primitives,
    const BatchedVariables<subcell::fixing_metric_tags>& fixing_metric,
    const BatchedVariables<
        evolution::dg::subcell::fd_patch::cell_geometry_tags>& cell_geometry,
    const Mesh<3>& subcell_mesh);
}  // namespace grmhd::GhValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
