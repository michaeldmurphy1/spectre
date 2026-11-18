// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructorOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivativesKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace grmhd::GhValenciaDivClean::fd {
template <typename System>
class Reconstructor;
}  // namespace grmhd::GhValenciaDivClean::fd
/// \endcond

namespace grmhd::GhValenciaDivClean::subcell {
/// The GH variables, which come first in the evolved variables
using gh_tags_on_device =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
               gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;

/// The evolved variables on the device, as in the `Variables` of the system:
/// the GH variables and then those of ValenciaDivClean
using evolved_tags_on_device =
    tmpl::append<gh_tags_on_device,
                 ValenciaDivClean::subcell::evolved_tags_on_device>;

/// The time derivatives of the `evolved_tags_on_device`
using dt_tags_on_device = db::wrap_tags_in<::Tags::dt, evolved_tags_on_device>;

/*!
 * \brief The reconstruction of the primitive variables that the FD time
 * derivative on the device uses for the GhValenciaDivClean `reconstructor`
 * (`MonotonisedCentralPrim` or `PositivityPreservingAdaptiveOrderPrim`, other
 * reconstructors are an error).
 */
template <typename System>
ValenciaDivClean::fd::ReconstructorOnDevice hydro_reconstructor_on_device(
    const fd::Reconstructor<System>& reconstructor);

/*!
 * \brief The degree of the unlimited reconstruction of the spacetime metric
 * that the GhValenciaDivClean `reconstructor` uses: 2 with
 * `MonotonisedCentralPrim` and \f$2 g - 2\f$ with
 * `PositivityPreservingAdaptiveOrderPrim`, whose ghost zones have width
 * \f$g\f$.
 */
template <typename System>
size_t spacetime_reconstruction_degree(
    const fd::Reconstructor<System>& reconstructor);

/*!
 * \brief Whether the FD time derivative on the device fixes the reconstructed
 * states on the faces to the atmosphere, as the GhValenciaDivClean
 * `reconstructor` does for elements doing FD on the host: with
 * `AtmosphereTreatment` `Always` or `OnFdOnly`.
 */
template <typename System>
bool fix_reconstructed_states_to_atmosphere(
    const fd::Reconstructor<System>& reconstructor);

/*!
 * \brief The FD time derivative of the GhValenciaDivClean system for a batch
 * of elements whose data is on the device, as `subcell::TimeDerivative`
 * computes it for one element on the host, with the harmonic gauge.
 *
 * \details This supports what `ValenciaDivClean::subcell::
 * TimeDerivativeOnDevice` supports for the GRMHD part, with `Correction` the
 * boundary correction of the GRMHD variables (`Rusanov` or `Hll`, e.g. of a
 * product of GH and GRMHD corrections). The time derivative is computed as on
 * the host:
 * 1. if the spacetime dissipation is given, apply the Kreiss-Oliger filter of
 *    order \f$2 g\f$ (with ghost zones of width \f$g\f$) to the GH variables,
 *    replacing them in the evolved variables and in the padded ghost
 *    variables;
 * 2. compute the derivatives of the GH variables with FD of order \f$2 g\f$;
 * 3. compute the constraint damping parameters from the damping functions;
 * 4. compute the GH time derivatives (`gh::TimeDerivativeOnDevice`) and, if
 *    the mesh moves, add the mesh velocity times the derivatives;
 * 5. compute the spacetime variables at the cell centers that the GRMHD
 *    sources need (`grmhd_spacetime_on_device`);
 * 6. reconstruct the spacetime metric to both sides of the faces with
 *    unlimited reconstruction and compute the lapse, shift and spatial metric
 *    there (`fd::spacetime_on_faces_on_device`);
 * 7. compute the GRMHD time derivatives
 *    (`ValenciaDivClean::subcell::TimeDerivativeOnDevice`);
 * 8. add the stress energy of the fluid to \f$\partial_t\Pi_{ab}\f$
 *    (`add_stress_energy_to_dt_pi_on_device`).
 *
 * The intermediate results are in device memory that the constructor
 * allocates once.
 */
template <typename Correction>
class TimeDerivativeOnDevice {
 public:
  static constexpr size_t dim = 3;
  template <typename Tag>
  using Batch = typename ::Tags::BatchView<Tag>::type;

  /// The derivatives of the GH variables
  using deriv_tags = db::wrap_tags_in<::Tags::deriv, gh_tags_on_device,
                                      tmpl::size_t<dim>, Frame::Inertial>;
  /// The constraint damping parameters
  using gamma_tags =
      tmpl::list<gh::Tags::ConstraintGamma0, gh::Tags::ConstraintGamma1,
                 gh::Tags::ConstraintGamma2>;
  /// The spacetime metric on one side of the faces
  using face_spacetime_metric_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, dim>>;
  /// The mesh velocity at the cell centers
  using MeshVelocity = gh::TimeDerivativeOnDevice::MeshVelocity;

  TimeDerivativeOnDevice() = default;

  /*!
   * \brief Allocate the intermediate results for `number_of_elements`
   * elements with the `dg_mesh` on `exec`.
   *
   * \param exec the execution space instance that allocates the memory.
   * \param number_of_elements the number of elements in the batch.
   * \param dg_mesh the DG mesh of the elements.
   * \param hydro_reconstructor the reconstruction of the primitive variables,
   * see `hydro_reconstructor_on_device`.
   * \param spacetime_reconstruction_degree the degree of the unlimited
   * reconstruction of the spacetime metric (2, 4, 6 or 8), see
   * `spacetime_reconstruction_degree`.
   * \param spacetime_dissipation the dissipation of the Kreiss-Oliger filter
   * of the GH variables, if they are filtered.
   */
  TimeDerivativeOnDevice(
      const Kokkos::DefaultExecutionSpace& exec, size_t number_of_elements,
      const Mesh<dim>& dg_mesh,
      ValenciaDivClean::fd::ReconstructorOnDevice hydro_reconstructor,
      size_t spacetime_reconstruction_degree,
      std::optional<double> spacetime_dissipation);

  /*!
   * \brief Compute the time derivatives `dt_vars` of a batch of elements.
   *
   * \param exec the execution space instance that runs the kernels.
   * \param dt_vars the time derivatives of the evolved variables.
   * \param evolved_variables the evolved variables at the cell centers. Their
   * GH variables are filtered if the spacetime dissipation is given.
   * \param face_status where the failures of the fix to the atmosphere on
   * both sides of the faces are added (see
   * `ValenciaDivClean::subcell::TimeDerivativeOnDevice`).
   * \param padded_ghost_variables the
   * `Tags::primitive_grmhd_and_spacetime_reconstruction_tags` of the elements
   * padded with ghost zones of the width of the reconstruction. The GH
   * variables in the interior are replaced by the filtered ones if the
   * spacetime dissipation is given.
   * \param primitives the primitive variables at the cell centers.
   * \param inverse_jacobian the inverse Jacobian from the element-logical to
   * the inertial frame at the cell centers.
   * \param grid_coordinates the grid coordinates of the cell centers, at
   * which the damping functions are evaluated.
   * \param mesh_velocity the mesh velocity at the cell centers, if the mesh
   * moves.
   * \param face_geometry the normal covectors, determinants of the inverse
   * Jacobian and mesh velocities on the faces (a batch of \f$3N\f$ face sets).
   * \param cell_geometry the determinant of the inverse Jacobian at the cell
   * centers.
   * \param div_mesh_velocity the divergence of the mesh velocity of each
   * element.
   * \param damping_functions the damping functions of \f$\gamma_0\f$,
   * \f$\gamma_1\f$ and \f$\gamma_2\f$ at the current time, in device memory.
   * \param correction the boundary correction of the GRMHD variables.
   * \param equation_of_state the equation of state.
   * \param fix_to_atmosphere the device view of the fix to the atmosphere
   * that is applied to the reconstructed primitive variables on the faces, or
   * `nullptr` if `fix_reconstructed_states_to_atmosphere()` of the
   * reconstructor is false.
   * \param constraint_damping_parameter the damping parameter of the
   * divergence cleaning field.
   * \param packaged_data_from_dg where the data from DG neighbors goes on
   * the faces, skipping their GH components.
   * \param received_mortar_data the data received from DG neighbors.
   *
   * All work is only enqueued on `exec`.
   */
  void apply(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<BatchedVariables<dt_tags_on_device>*> dt_vars,
      gsl::not_null<BatchedVariables<evolved_tags_on_device>*>
          evolved_variables,
      gsl::not_null<DeferredPointwiseStatus<
          ValenciaDivClean::fd::face_atmosphere_snapshot_tags>*>
          face_status,
      const ::fd::device::VolumeView& padded_ghost_variables,
      const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives,
      const ::fd::device::BatchInverseJacobian<dim>& inverse_jacobian,
      const Batch<domain::Tags::Coordinates<dim, Frame::Grid>>&
          grid_coordinates,
      const std::optional<Batch<MeshVelocity>>& mesh_velocity,
      const BatchedVariables<
          evolution::dg::subcell::fd_patch::inertial_face_tags>& face_geometry,
      const BatchedVariables<
          evolution::dg::subcell::fd_patch::inertial_cell_tags>& cell_geometry,
      const Kokkos::View<const double*>& div_mesh_velocity,
      const Kokkos::View<const ConstraintDamping::GaussianSum<dim>*>&
          damping_functions,
      const Correction& correction,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const VariableFixing::FixToAtmosphere<3>::DeviceView* fix_to_atmosphere,
      double constraint_damping_parameter,
      const evolution::dg::subcell::fd_patch::PackagedDataFromDg<dim>&
          packaged_data_from_dg,
      const Kokkos::View<const double*>& received_mortar_data);

  /// The number of points of each face set (see
  /// `ValenciaDivClean::subcell::TimeDerivativeOnDevice`)
  size_t number_of_face_points() const {
    return grmhd_.number_of_face_points();
  }

  /// The spacetime variables at the cell centers that the GRMHD sources used
  /// in the last `apply()`, e.g. for tests
  const BatchedVariables<ValenciaDivClean::subcell::cell_spacetime_tags>&
  cell_spacetime_variables() const {
    return cell_spacetime_variables_;
  }

 private:
  size_t number_of_elements_ = 0;
  Index<dim> subcell_extents_{};
  size_t ghost_zone_size_ = 0;
  size_t spacetime_reconstruction_degree_ = 0;
  std::optional<double> spacetime_dissipation_{};
  ValenciaDivClean::subcell::TimeDerivativeOnDevice<Correction> grmhd_{};
  gh::TimeDerivativeOnDevice gh_{};
  BatchedVariables<deriv_tags> derivatives_{};
  BatchedVariables<gamma_tags> gammas_{};
  BatchedVariables<ValenciaDivClean::subcell::cell_spacetime_tags>
      cell_spacetime_variables_{};
  BatchedVariables<face_spacetime_metric_tags> lower_face_spacetime_metric_{};
  BatchedVariables<face_spacetime_metric_tags> upper_face_spacetime_metric_{};
};
}  // namespace grmhd::GhValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
