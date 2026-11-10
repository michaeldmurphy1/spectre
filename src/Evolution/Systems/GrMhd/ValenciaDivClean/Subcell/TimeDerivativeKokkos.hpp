// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
class PackagedDataFromDg;
}  // namespace evolution::dg::subcell::fd_patch
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/// The evolved variables of a batch of FD elements on the device
using evolved_tags_on_device =
    tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau, Tags::TildeS<>,
               Tags::TildeB<>, Tags::TildePhi>;
/// The time derivatives of the `evolved_tags_on_device`
using dt_tags_on_device = db::wrap_tags_in<::Tags::dt, evolved_tags_on_device>;

/// The spacetime variables on the faces that the FD time derivative copies to
/// the face variables, as `subcell::TimeDerivative` does from
/// `evolution::dg::subcell::Tags::OnSubcellFaces`
using face_spacetime_tags =
    tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
               gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>>;

/// The spacetime variables at the cell centers that the sources need, and the
/// inertial coordinates (only used by Cartoon meshes, which FD patches don't
/// support)
using cell_spacetime_tags =
    tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
               ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                             Frame::Inertial>,
               ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                             Frame::Inertial>,
               gr::Tags::SpatialMetric<DataVector, 3>,
               ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                             tmpl::size_t<3>, Frame::Inertial>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>,
               gr::Tags::ExtrinsicCurvature<DataVector, 3>,
               evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>;

/*!
 * \brief The FD time derivative of the ValenciaDivClean system for a batch of
 * elements whose data is on the device, as `subcell::TimeDerivative` computes
 * it for one element on the host.
 *
 * \details This supports what FD patches need: monotonised central
 * reconstruction of the temperature (not \f$\rho T\f$), finite difference
 * derivative order 2 (no high-order flux corrections), the boundary correction
 * `Correction` (`Rusanov` or `Hll`), and an affine grid to inertial map (see
 * `evolution::dg::subcell::fd_patch::AffineGridToInertial`). Unlike the host,
 * it supports a moving mesh, following `GhValenciaDivClean`.
 *
 * The time derivative is computed as on the host:
 * 1. reconstruct the ghost variables to both sides of all faces,
 * 2. compute the other primitive variables, with the fix to the atmosphere if
 *    given, and the conserved variables on the faces,
 * 3. compute the fluxes on the faces, including the mesh velocity terms,
 * 4. compute the packaged data with the normalized normal covector, replacing
 *    the data on faces next to DG elements with the data they sent
 *    (`evolution::dg::subcell::fd_patch::PackagedDataFromDg`),
 * 5. compute the boundary corrections and multiply them by the normalization
 *    of the normal covector and the inverse of the determinant of the inverse
 *    Jacobian on the faces,
 * 6. compute the sources at the cell centers (zero for \f$\tilde{D}\f$ and
 *    \f$\tilde{Y}_e\f$), subtract the divergence of the mesh velocity times the
 *    evolved variables, and add the flux divergence.
 *
 * All face data is in batches of \f$3N\f$ face sets ordered by direction and
 * then element (see `fd::reconstructed_face_views`). The face variables and
 * the packaged data on both sides and the boundary corrections are allocated
 * once by the constructor.
 *
 * Steps 2 to 4 are one kernel for each side of the faces with `Rusanov`, and
 * two with `Hll` (see `face_tags`), and step 5 is one kernel, so that the
 * intermediate results stay on the stack instead of in memory.
 */
template <typename Correction>
class TimeDerivativeOnDevice {
 public:
  using packaged_tags = typename Correction::dg_package_field_tags;
  /// The variables on each side of the faces. With `Rusanov` these are only
  /// the reconstructed and spacetime variables, from which one kernel computes
  /// the packaged data (see
  /// `BoundaryCorrections::package_data_from_reconstruction_on_device`).
  /// `Hll` needs the variables on all faces of a face set before computing the
  /// packaged data, so they are kept too (see `fd::device_face_tags`).
  using face_tags = tmpl::conditional_t<
      std::is_same_v<Correction, BoundaryCorrections::Rusanov>,
      fd::reconstruction_face_tags, fd::device_face_tags>;

  TimeDerivativeOnDevice() = default;

  /// Allocate the face data for `number_of_elements` elements with the
  /// `dg_mesh` on `exec`.
  TimeDerivativeOnDevice(const Kokkos::DefaultExecutionSpace& exec,
                         size_t number_of_elements, const Mesh<3>& dg_mesh);

  /// Copy the spacetime variables on the faces (a batch of \f$3N\f$ face
  /// sets) to both sides of the faces. They aren't modified by `apply()`, so
  /// this is only needed when the spacetime changes.
  void set_face_spacetime_variables(
      const Kokkos::DefaultExecutionSpace& exec,
      const BatchedVariables<face_spacetime_tags>& face_spacetime_variables);

  /*!
   * \brief Compute the time derivatives `dt_vars` of a batch of elements.
   *
   * \param exec the execution space instance that runs the kernels.
   * \param dt_vars the time derivatives of the evolved variables.
   * \param padded_ghost_variables the ghost variables of the elements padded
   * with ghost zones of width at least 2, filled by
   * `evolution::dg::subcell::fd_patch::GhostZones`.
   * \param evolved_variables the evolved variables at the cell centers.
   * \param primitives the primitive variables at the cell centers.
   * \param cell_spacetime_variables the spacetime variables at the cell
   * centers, which the sources need.
   * \param face_geometry the normal covectors, determinants of the inverse
   * Jacobian and mesh velocities on the faces (a batch of \f$3N\f$ face sets).
   * \param cell_geometry the determinant of the inverse Jacobian at the cell
   * centers.
   * \param grid_to_inertial the affine map from the grid to the inertial
   * frame, for the divergence of the mesh velocity.
   * \param correction the boundary correction.
   * \param equation_of_state the equation of state, for the reconstruction
   * and the boundary corrections.
   * \param fix_to_atmosphere the device view of the fix to the atmosphere
   * that is applied to the reconstructed primitive variables on the faces.
   * \param constraint_damping_parameter the damping parameter of the
   * divergence cleaning field.
   * \param packaged_data_from_dg where the data from DG neighbors goes on
   * the faces.
   * \param received_mortar_data the data received from DG neighbors, laid
   * out as described by `packaged_data_from_dg`.
   *
   * Returns the status of the fix to the atmosphere on the lower and upper
   * sides of the faces (see `fd::compute_conservatives_on_device`), which
   * waits for the face computations to complete. The rest of the work is only
   * enqueued on `exec`.
   */
  std::array<PointwiseStatusSummary, 2> apply(
      const Kokkos::DefaultExecutionSpace& exec,
      gsl::not_null<BatchedVariables<dt_tags_on_device>*> dt_vars,
      const ::fd::device::VolumeView& padded_ghost_variables,
      const BatchedVariables<evolved_tags_on_device>& evolved_variables,
      const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives,
      const BatchedVariables<cell_spacetime_tags>& cell_spacetime_variables,
      const BatchedVariables<
          evolution::dg::subcell::fd_patch::inertial_face_tags>& face_geometry,
      const BatchedVariables<
          evolution::dg::subcell::fd_patch::inertial_cell_tags>& cell_geometry,
      const evolution::dg::subcell::fd_patch::AffineGridToInertial&
          grid_to_inertial,
      const Correction& correction,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const VariableFixing::FixToAtmosphere<3>::DeviceView* fix_to_atmosphere,
      double constraint_damping_parameter,
      const evolution::dg::subcell::fd_patch::PackagedDataFromDg<3>&
          packaged_data_from_dg,
      const Kokkos::View<const double*>& received_mortar_data);

  /// The variables on the lower side of the faces, e.g. to report a failure
  const BatchedVariables<face_tags>& lower_faces() const {
    return lower_faces_;
  }
  /// The variables on the upper side of the faces
  const BatchedVariables<face_tags>& upper_faces() const {
    return upper_faces_;
  }

 private:
  size_t number_of_elements_ = 0;
  Mesh<3> dg_mesh_{};
  Index<3> subcell_extents_{};
  std::array<double, 3> one_over_delta_xi_{};
  BatchedVariables<face_tags> lower_faces_{};
  BatchedVariables<face_tags> upper_faces_{};
  BatchedVariables<packaged_tags> lower_packaged_data_{};
  BatchedVariables<packaged_tags> upper_packaged_data_{};
  BatchedVariables<BoundaryCorrections::BoundaryCorrectionTags>
      boundary_corrections_{};
};
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
