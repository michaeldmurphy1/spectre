// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
namespace {
// Copies the spacetime variables to the face variables
struct CopyFaceSpacetime {
  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> lapse_on_face,
      const gsl::not_null<tnsr::I<double, 3>*> shift_on_face,
      const gsl::not_null<tnsr::ii<double, 3>*> spatial_metric_on_face,
      const gsl::not_null<Scalar<double>*> sqrt_det_spatial_metric_on_face,
      const gsl::not_null<tnsr::II<double, 3>*> inv_spatial_metric_on_face,
      const Scalar<double>& lapse, const tnsr::I<double, 3>& shift,
      const tnsr::ii<double, 3>& spatial_metric,
      const Scalar<double>& sqrt_det_spatial_metric,
      const tnsr::II<double, 3>& inv_spatial_metric) {
    *lapse_on_face = lapse;
    *shift_on_face = shift;
    *spatial_metric_on_face = spatial_metric;
    *sqrt_det_spatial_metric_on_face = sqrt_det_spatial_metric;
    *inv_spatial_metric_on_face = inv_spatial_metric;
  }
};
}  // namespace

template <typename Correction>
TimeDerivativeOnDevice<Correction>::TimeDerivativeOnDevice(
    const Kokkos::DefaultExecutionSpace& exec, const size_t number_of_elements,
    const Mesh<3>& dg_mesh, fd::ReconstructorOnDevice reconstructor)
    : number_of_elements_(number_of_elements),
      reconstructor_(std::move(reconstructor)),
      dg_mesh_(dg_mesh),
      subcell_extents_(evolution::dg::subcell::fd::mesh(dg_mesh).extents()) {
  // As `subcell::TimeDerivative`, which assumes isotropic extents
  const auto cell_logical_coords =
      logical_coordinates(evolution::dg::subcell::fd::mesh(dg_mesh));
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(one_over_delta_xi_, d) =
        1.0 / (get<0>(cell_logical_coords)[1] - get<0>(cell_logical_coords)[0]);
  }
  Index<3> face_extents = subcell_extents_;
  ++face_extents[0];
  const size_t number_of_face_points = face_extents.product();
  const size_t number_of_face_sets = 3 * number_of_elements;
  lower_faces_ = BatchedVariables<face_tags>{exec, number_of_face_points,
                                             number_of_face_sets};
  upper_faces_ = BatchedVariables<face_tags>{exec, number_of_face_points,
                                             number_of_face_sets};
  lower_packaged_data_ = BatchedVariables<packaged_tags>{
      exec, number_of_face_points, number_of_face_sets};
  upper_packaged_data_ = BatchedVariables<packaged_tags>{
      exec, number_of_face_points, number_of_face_sets};
  boundary_corrections_ =
      BatchedVariables<BoundaryCorrections::BoundaryCorrectionTags>{
          exec, number_of_face_points, number_of_face_sets};
}

template <typename Correction>
void TimeDerivativeOnDevice<Correction>::set_face_spacetime_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const BatchedVariables<face_spacetime_tags>& face_spacetime_variables) {
  for (auto* faces : {&lower_faces_, &upper_faces_}) {
    apply_pointwise_on_device<CopyFaceSpacetime>(
        exec, make_not_null(faces), face_spacetime_tags{},
        face_spacetime_tags{},
        [&]<typename... SpacetimeTags>(tmpl::list<SpacetimeTags...> /*meta*/) {
          return make_device_arguments<SpacetimeTags...>(
              exec, get<::Tags::BatchView<SpacetimeTags>>(
                        face_spacetime_variables)...);
        }(face_spacetime_tags{}));
  }
}

template <typename Correction>
gsl::not_null<
    BatchedVariables<typename TimeDerivativeOnDevice<Correction>::face_tags>*>
TimeDerivativeOnDevice<Correction>::faces(const Side side) {
  return make_not_null(side == Side::Lower ? &lower_faces_ : &upper_faces_);
}

template class TimeDerivativeOnDevice<BoundaryCorrections::Rusanov>;
template class TimeDerivativeOnDevice<BoundaryCorrections::Hll>;

#define CORRECTION(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                                 \
  template void TimeDerivativeOnDevice<CORRECTION(data)>::apply(               \
      const Kokkos::DefaultExecutionSpace& exec,                               \
      gsl::not_null<BatchedVariables<dt_tags_on_device>*> dt_vars,             \
      gsl::not_null<                                                           \
          DeferredPointwiseStatus<fd::face_atmosphere_snapshot_tags>*>         \
          face_status,                                                         \
      const ::fd::device::SplitVolumeView& ghost_variables,                    \
      const BatchedVariables<evolved_tags_on_device>& evolved_variables,       \
      const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives,       \
      const CellSpacetimeViews& cell_spacetime,                                \
      const BatchedVariables<                                                  \
          evolution::dg::subcell::fd_patch::inertial_face_tags>&               \
          face_geometry,                                                       \
      const BatchedVariables<                                                  \
          evolution::dg::subcell::fd_patch::inertial_cell_tags>&               \
          cell_geometry,                                                       \
      bool mesh_is_moving, const DivergenceOfMeshVelocity& div_mesh_velocity,  \
      const CORRECTION(data) & correction,                                     \
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,     \
      const VariableFixing::FixToAtmosphere<3>::DeviceView* fix_to_atmosphere, \
      double constraint_damping_parameter,                                     \
      const evolution::dg::subcell::fd_patch::PackagedDataFromDg<3>&           \
          packaged_data_from_dg,                                               \
      const Kokkos::View<const double*>& received_mortar_data);

GENERATE_INSTANTIATIONS(INSTANTIATION, (BoundaryCorrections::Rusanov,
                                        BoundaryCorrections::Hll))

#undef INSTANTIATION
#undef CORRECTION
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
