// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/OnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceViewsKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructorOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/FiniteDifference/FluxDivergenceKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
namespace {
namespace fd_patch = evolution::dg::subcell::fd_patch;
// Plain arguments of the pointwise functions
struct DivMeshVelocity : db::SimpleTag {
  using type = double;
};

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

// Multiplies the boundary corrections by the normalization of the normal
// covector and divides by the determinant of the inverse Jacobian on the faces
struct ScaleBoundaryCorrections {
  using return_tags = BoundaryCorrections::BoundaryCorrectionTags;
  using argument_tags =
      tmpl::append<BoundaryCorrections::BoundaryCorrectionTags,
                   tmpl::list<fd_patch::Tags::FaceConormal,
                              gr::Tags::InverseSpatialMetric<DataVector, 3>,
                              fd_patch::Tags::FaceDetInverseJacobian>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> result_tilde_d,
      const gsl::not_null<Scalar<double>*> result_tilde_ye,
      const gsl::not_null<Scalar<double>*> result_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3>*> result_tilde_s,
      const gsl::not_null<tnsr::I<double, 3>*> result_tilde_b,
      const gsl::not_null<Scalar<double>*> result_tilde_phi,
      const Scalar<double>& tilde_d, const Scalar<double>& tilde_ye,
      const Scalar<double>& tilde_tau, const tnsr::i<double, 3>& tilde_s,
      const tnsr::I<double, 3>& tilde_b, const Scalar<double>& tilde_phi,
      const tnsr::i<double, 3>& conormal,
      const tnsr::II<double, 3>& inv_spatial_metric,
      const Scalar<double>& det_inverse_jacobian) {
    double magnitude_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        magnitude_squared +=
            conormal.get(i) * conormal.get(j) * inv_spatial_metric.get(i, j);
      }
    }
    // Same order of operations as the host
    const double normalization = sqrt(magnitude_squared);
    const double inverse_det = 1.0 / get(det_inverse_jacobian);
    const auto scale = [normalization, inverse_det](const double value) {
      return (value * normalization) * inverse_det;
    };
    get(*result_tilde_d) = scale(get(tilde_d));
    get(*result_tilde_ye) = scale(get(tilde_ye));
    get(*result_tilde_tau) = scale(get(tilde_tau));
    get(*result_tilde_phi) = scale(get(tilde_phi));
    for (size_t i = 0; i < 3; ++i) {
      result_tilde_s->get(i) = scale(tilde_s.get(i));
      result_tilde_b->get(i) = scale(tilde_b.get(i));
    }
  }
};

// The packaged data on the exterior side of the faces, so that it has a
// different tag than the interior one
template <typename Tag>
struct Exterior : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = typename Tag::type;
};

// The boundary corrections of `Correction` from the packaged data on both
// sides of the faces (`Correction::dg_boundary_terms_impl` with the weak
// form), scaled as `ScaleBoundaryCorrections` does, in one kernel
template <typename Correction>
struct ScaledBoundaryTerms {
  using return_tags = BoundaryCorrections::BoundaryCorrectionTags;
  using argument_tags = tmpl::append<
      typename Correction::dg_package_field_tags,
      db::wrap_tags_in<Exterior, typename Correction::dg_package_field_tags>,
      tmpl::list<fd_patch::Tags::FaceConormal,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 fd_patch::Tags::FaceDetInverseJacobian>>;

  // The arguments are the packaged data on both sides, followed by the three
  // arguments of the scaling
  template <typename... Args>
  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> result_tilde_d,
      const gsl::not_null<Scalar<double>*> result_tilde_ye,
      const gsl::not_null<Scalar<double>*> result_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3>*> result_tilde_s,
      const gsl::not_null<tnsr::I<double, 3>*> result_tilde_b,
      const gsl::not_null<Scalar<double>*> result_tilde_phi,
      const Args&... args) {
    apply_impl(result_tilde_d, result_tilde_ye, result_tilde_tau,
               result_tilde_s, result_tilde_b, result_tilde_phi,
               std::make_index_sequence<sizeof...(Args) - 3>{},
               std::forward_as_tuple(args...));
  }

 private:
  template <size_t... Is, typename Arguments>
  KOKKOS_FUNCTION static void apply_impl(
      const gsl::not_null<Scalar<double>*> result_tilde_d,
      const gsl::not_null<Scalar<double>*> result_tilde_ye,
      const gsl::not_null<Scalar<double>*> result_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3>*> result_tilde_s,
      const gsl::not_null<tnsr::I<double, 3>*> result_tilde_b,
      const gsl::not_null<Scalar<double>*> result_tilde_phi,
      std::index_sequence<Is...> /*meta*/, const Arguments& arguments) {
    constexpr size_t number_of_packaged_arguments = sizeof...(Is);
    Scalar<double> tilde_d{};
    Scalar<double> tilde_ye{};
    Scalar<double> tilde_tau{};
    tnsr::i<double, 3> tilde_s{};
    tnsr::I<double, 3> tilde_b{};
    Scalar<double> tilde_phi{};
    Correction::dg_boundary_terms_impl(
        make_not_null(&tilde_d), make_not_null(&tilde_ye),
        make_not_null(&tilde_tau), make_not_null(&tilde_s),
        make_not_null(&tilde_b), make_not_null(&tilde_phi),
        std::get<Is>(arguments)..., ::dg::Formulation::WeakInertial);
    ScaleBoundaryCorrections::apply(
        result_tilde_d, result_tilde_ye, result_tilde_tau, result_tilde_s,
        result_tilde_b, result_tilde_phi, tilde_d, tilde_ye, tilde_tau, tilde_s,
        tilde_b, tilde_phi, std::get<number_of_packaged_arguments>(arguments),
        std::get<number_of_packaged_arguments + 1>(arguments),
        std::get<number_of_packaged_arguments + 2>(arguments));
  }
};

// The sources as the time derivatives, as `subcell::TimeDerivative` sets them
// before adding the flux divergence, including the mesh velocity term
struct SourcesAsTimeDerivatives {
  using return_tags = dt_tags_on_device;
  using argument_tags =
      tmpl::push_back<ComputeSources::argument_tags, DivMeshVelocity>;

  template <typename... SourceArgs>
  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> dt_tilde_d,
      const gsl::not_null<Scalar<double>*> dt_tilde_ye,
      const gsl::not_null<Scalar<double>*> dt_tilde_tau,
      const gsl::not_null<tnsr::i<double, 3>*> dt_tilde_s,
      const gsl::not_null<tnsr::I<double, 3>*> dt_tilde_b,
      const gsl::not_null<Scalar<double>*> dt_tilde_phi,
      const Scalar<double>& tilde_d, const Scalar<double>& tilde_ye,
      const Scalar<double>& tilde_tau, const tnsr::i<double, 3>& tilde_s,
      const tnsr::I<double, 3>& tilde_b, const Scalar<double>& tilde_phi,
      const SourceArgs&... source_args) {
    // The last argument is the divergence of the mesh velocity
    const auto call = [&](const auto&... args) {
      ComputeSources::apply(dt_tilde_tau, dt_tilde_s, dt_tilde_b, dt_tilde_phi,
                            tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b,
                            tilde_phi, args...);
    };
    const double div_mesh_velocity = apply_all_but_last(call, source_args...);
    // No sources for these
    get(*dt_tilde_d) = 0.0;
    get(*dt_tilde_ye) = 0.0;
    if (div_mesh_velocity != 0.0) {
      get(*dt_tilde_d) -= div_mesh_velocity * get(tilde_d);
      get(*dt_tilde_ye) -= div_mesh_velocity * get(tilde_ye);
      get(*dt_tilde_tau) -= div_mesh_velocity * get(tilde_tau);
      get(*dt_tilde_phi) -= div_mesh_velocity * get(tilde_phi);
      for (size_t i = 0; i < 3; ++i) {
        dt_tilde_s->get(i) -= div_mesh_velocity * tilde_s.get(i);
        dt_tilde_b->get(i) -= div_mesh_velocity * tilde_b.get(i);
      }
    }
  }

 private:
  // Calls `function` with all arguments but the last, and returns the last
  template <typename Function, typename... Args>
  KOKKOS_FUNCTION static double apply_all_but_last(const Function& function,
                                                   const Args&... args) {
    return apply_all_but_last_impl(
        function, std::make_index_sequence<sizeof...(Args) - 1>{}, args...);
  }

  template <typename Function, size_t... Is, typename... Args>
  KOKKOS_FUNCTION static double apply_all_but_last_impl(
      const Function& function, std::index_sequence<Is...> /*meta*/,
      const Args&... args) {
    const auto arguments = std::forward_as_tuple(args...);
    function(std::get<Is>(arguments)...);
    return std::get<sizeof...(Args) - 1>(arguments);
  }
};

// A `VolumeView` of `count` entries of a batch starting at `first`, with the
// points of each entry laid out with `extents`
template <typename View>
View entries_as_volume_view(const auto& storage, const size_t first,
                            const size_t count, const Index<3>& extents) {
  const size_t components = storage.extent(1);
  const size_t points = storage.extent(0);
  ASSERT(
      points == extents.product(),
      "The batch has " << points << " points but the extents are " << extents);
  // The entries are contiguous since the storage is `LayoutLeft`
  return View(
      Kokkos::subview(storage, Kokkos::ALL(), Kokkos::ALL(),
                      Kokkos::pair<size_t, size_t>(first, first + count))
          .data(),
      count, components, extents[2], extents[1], extents[0]);
}
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
void TimeDerivativeOnDevice<Correction>::apply(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<dt_tags_on_device>*> dt_vars,
    const gsl::not_null<
        DeferredPointwiseStatus<fd::face_atmosphere_snapshot_tags>*>
        face_status,
    const ::fd::device::VolumeView& padded_ghost_variables,
    const BatchedVariables<evolved_tags_on_device>& evolved_variables,
    const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives,
    const BatchedVariables<cell_spacetime_tags>& cell_spacetime_variables,
    const BatchedVariables<fd_patch::inertial_face_tags>& face_geometry,
    const BatchedVariables<fd_patch::inertial_cell_tags>& cell_geometry,
    const fd_patch::AffineGridToInertial& grid_to_inertial,
    const Correction& correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* fix_to_atmosphere,
    const double constraint_damping_parameter,
    const fd_patch::PackagedDataFromDg<3>& packaged_data_from_dg,
    const Kokkos::View<const double*>& received_mortar_data) {
  ASSERT(dt_vars->number_of_elements() == number_of_elements_ and
             face_geometry.number_of_elements() == 3 * number_of_elements_,
         "The batches don't match the " << number_of_elements_
                                        << " elements of the time derivative");

  // 1. Reconstruct
  reconstructor_.reconstruct(
      exec, fd::reconstructed_face_views(upper_faces_, subcell_extents_),
      fd::reconstructed_face_views(lower_faces_, subcell_extents_),
      padded_ghost_variables, subcell_extents_);

  // 2. Primitive and conserved variables, fluxes, normals and packaged data
  // on the faces
  const auto conormal =
      get<::Tags::BatchView<fd_patch::Tags::FaceConormal>>(face_geometry);
  const auto mesh_velocity =
      get<::Tags::BatchView<fd_patch::Tags::FaceMeshVelocity>>(face_geometry);
  const auto* const moving_mesh_velocity =
      grid_to_inertial.is_moving ? &mesh_velocity : nullptr;
  const std::array<double, 2> signs{{1.0, -1.0}};
  if constexpr (std::is_same_v<Correction, BoundaryCorrections::Rusanov>) {
    // In one kernel for each side
    (void)correction;
    for (size_t side = 0; side < 2; ++side) {
      BoundaryCorrections::package_data_from_reconstruction_on_device(
          exec,
          make_not_null(side == 0 ? &lower_packaged_data_
                                  : &upper_packaged_data_),
          face_status, side == 0 ? lower_faces_ : upper_faces_, conormal,
          gsl::at(signs, side), moving_mesh_velocity, equation_of_state,
          fix_to_atmosphere);
    }
  } else {
    // `Hll` needs the variables on all faces of a face set first
    for (size_t side = 0; side < 2; ++side) {
      fd::compute_conservatives_on_device(
          exec, make_not_null(side == 0 ? &lower_faces_ : &upper_faces_),
          face_status, equation_of_state, fix_to_atmosphere);
    }
    for (size_t side = 0; side < 2; ++side) {
      BoundaryCorrections::fluxes_and_package_data_on_device(
          exec,
          make_not_null(side == 0 ? &lower_packaged_data_
                                  : &upper_packaged_data_),
          side == 0 ? lower_faces_ : upper_faces_, conormal,
          gsl::at(signs, side), moving_mesh_velocity, correction,
          equation_of_state);
    }
  }
  packaged_data_from_dg.apply(
      exec,
      Kokkos::View<double*>(lower_packaged_data_.view().data(),
                            lower_packaged_data_.view().span()),
      Kokkos::View<double*>(upper_packaged_data_.view().data(),
                            upper_packaged_data_.view().span()),
      received_mortar_data);

  // 3. Boundary corrections, with the upper side as the interior, scaled by
  // the normalization of the normal and the inverse Jacobian, in one kernel
  apply_pointwise_on_device<ScaledBoundaryTerms<Correction>>(
      exec, make_not_null(&boundary_corrections_),
      typename ScaledBoundaryTerms<Correction>::return_tags{},
      typename ScaledBoundaryTerms<Correction>::argument_tags{},
      [&]<typename... PackagedTags>(tmpl::list<PackagedTags...> /*meta*/) {
        return make_device_arguments<
            PackagedTags..., Exterior<PackagedTags>...,
            fd_patch::Tags::FaceConormal,
            gr::Tags::InverseSpatialMetric<DataVector, 3>,
            fd_patch::Tags::FaceDetInverseJacobian>(
            exec, get<::Tags::BatchView<PackagedTags>>(upper_packaged_data_)...,
            get<::Tags::BatchView<PackagedTags>>(lower_packaged_data_)...,
            conormal,
            get<::Tags::BatchView<
                gr::Tags::InverseSpatialMetric<DataVector, 3>>>(lower_faces_),
            get<::Tags::BatchView<fd_patch::Tags::FaceDetInverseJacobian>>(
                face_geometry));
      }(packaged_tags{}));

  // 4. Sources, mesh velocity term and flux divergence
  const auto argument = [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    if constexpr (tmpl::list_contains_v<evolved_tags_on_device, Tag>) {
      return get<::Tags::BatchView<Tag>>(evolved_variables);
    } else if constexpr (tmpl::list_contains_v<hydro::grmhd_tags<DataVector>,
                                               Tag>) {
      return get<::Tags::BatchView<Tag>>(primitives);
    } else if constexpr (tmpl::list_contains_v<cell_spacetime_tags, Tag>) {
      return get<::Tags::BatchView<Tag>>(cell_spacetime_variables);
    } else if constexpr (std::is_same_v<Tag,
                                        Tags::ConstraintDampingParameter>) {
      return constraint_damping_parameter;
    } else if constexpr (std::is_same_v<Tag, domain::Tags::Mesh<3>>) {
      return dg_mesh_;
    } else {
      static_assert(std::is_same_v<Tag, DivMeshVelocity>,
                    "Unexpected argument of the sources");
      return grid_to_inertial.div_mesh_velocity;
    }
  };
  apply_pointwise_on_device<SourcesAsTimeDerivatives>(
      exec, dt_vars, SourcesAsTimeDerivatives::return_tags{},
      SourcesAsTimeDerivatives::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, argument(tmpl::type_<ArgumentTags>{})...);
      }(SourcesAsTimeDerivatives::argument_tags{}));

  std::array<::fd::device::ConstVolumeView, 3> corrections{};
  for (size_t d = 0; d < 3; ++d) {
    Index<3> face_extents = subcell_extents_;
    ++face_extents[d];
    gsl::at(corrections, d) =
        entries_as_volume_view<::fd::device::ConstVolumeView>(
            boundary_corrections_.view(), d * number_of_elements_,
            number_of_elements_, face_extents);
  }
  ::fd::device::add_cartesian_flux_divergence<3>(
      exec,
      entries_as_volume_view<::fd::device::VolumeView>(
          dt_vars->view(), 0, number_of_elements_, subcell_extents_),
      corrections,
      entries_as_volume_view<::fd::device::ConstVolumeView>(
          cell_geometry.view(), 0, number_of_elements_, subcell_extents_),
      one_over_delta_xi_, subcell_extents_);
}

template class TimeDerivativeOnDevice<BoundaryCorrections::Rusanov>;
template class TimeDerivativeOnDevice<BoundaryCorrections::Hll>;
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
