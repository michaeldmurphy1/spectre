// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDevice.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDevice.hpp"

namespace grmhd::ValenciaDivClean {
bool fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives) {
  using AtGridPoint = FixConservatives::AtGridPoint;
  const size_t number_of_points = get(*tilde_d).size();
  Variables<AtGridPoint::return_tags> fixed{number_of_points};
  const PointwiseStatusSummary status =
      copy_and_apply_pointwise_on_device<AtGridPoint>(
          exec, make_not_null(&fixed), AtGridPoint::return_tags{},
          AtGridPoint::argument_tags{}, *tilde_d, *tilde_ye, *tilde_tau,
          *tilde_s, tilde_b, spatial_metric, inv_spatial_metric,
          sqrt_det_spatial_metric, fix_conservatives,
          fix_conservatives.assume_non_zero_magnetic_field(tilde_b));
  if (status.number_of_failures > 0) {
    // The failed grid points keep their unfixed values, which are also still
    // in the inputs
    const size_t i = status.first_failed_grid_point;
    ERROR(
        "Failed to fix conserved variables on the device because the root "
        "finder failed to find the lorentz factor at "
        << status.number_of_failures << " of " << number_of_points
        << " grid points. At the first one, grid point " << i
        << ":\n  tilde_d = " << extract_point(*tilde_d, i)
        << "\n  tilde_ye = " << extract_point(*tilde_ye, i)
        << "\n  tilde_tau = " << extract_point(*tilde_tau, i)
        << "\n  tilde_s =\n"
        << extract_point(*tilde_s, i) << "\n  tilde_b =\n"
        << extract_point(tilde_b, i) << "\n  spatial_metric =\n"
        << extract_point(spatial_metric, i) << "\n  inv_spatial_metric =\n"
        << extract_point(inv_spatial_metric, i)
        << "\n  sqrt_det_spatial_metric = "
        << extract_point(sqrt_det_spatial_metric, i)
        << "\nRun on the host for the details of the root find.");
  }
  *tilde_d = get<Tags::TildeD>(fixed);
  *tilde_ye = get<Tags::TildeYe>(fixed);
  *tilde_tau = get<Tags::TildeTau>(fixed);
  *tilde_s = get<Tags::TildeS<Frame::Inertial>>(fixed);
  return status.any_flag;
}

bool fix_conservatives_on_device(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives) {
  return fix_conservatives_on_device(
      Kokkos::DefaultExecutionSpace{}, tilde_d, tilde_ye, tilde_tau, tilde_s,
      tilde_b, spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
      fix_conservatives);
}
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
