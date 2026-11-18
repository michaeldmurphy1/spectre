// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintGammasKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/SpacetimeOnFacesKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.tpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/FilterKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivativesKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/UnlimitedKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
namespace {
constexpr size_t dim = 3;
template <typename Tag>
using Batch = typename ::Tags::BatchView<Tag>::type;

constexpr size_t number_of_hydro_ghost_variables =
    Variables<Tags::primitive_grmhd_reconstruction_tags>::
        number_of_independent_components;
constexpr size_t number_of_gh_variables =
    Variables<gh_tags_on_device>::number_of_independent_components;
constexpr size_t number_of_spacetime_metric_components = 10;
static_assert(
    std::is_same_v<Tags::spacetime_reconstruction_tags, gh_tags_on_device>,
    "The GH variables must be in the same order in the ghost data and the "
    "evolved variables");
static_assert(
    std::is_same_v<Tags::primitive_grmhd_reconstruction_tags,
                   ValenciaDivClean::fd::reconstructed_face_tags>,
    "The hydro ghost variables must be those that the hydro reconstruction "
    "reconstructs");
static_assert(
    std::is_same_v<evolved_tags_on_device,
                   typename System<RadiationTransport::NoNeutrinos::System>::
                       variables_tag::tags_list>,
    "The evolved variables must be those of the system");

// The components [`first`, `first + count`) of a batch of elements with
// `extents` as a volume view, with the points of each component laid out with
// `extents` (the layout of a `Variables` for each element)
template <typename TagsList>
::fd::device::StridedVolumeView component_range(
    const BatchedVariables<TagsList>& batch, const size_t first,
    const size_t count, const Index<dim>& extents) {
  constexpr size_t number_of_components =
      BatchedVariables<TagsList>::number_of_independent_components;
  const size_t number_of_points = batch.number_of_grid_points();
  ASSERT(number_of_points == extents.product(),
         "The batch has " << number_of_points << " points but the extents are "
                          << extents);
  ASSERT(first + count <= number_of_components,
         "The components [" << first << ", " << first + count
                            << ") aren't in the batch of "
                            << number_of_components << " components");
  return ::fd::device::StridedVolumeView(
      // The address of the first point of component `first` of element 0
      Kokkos::subview(batch.view(), Kokkos::ALL(), first, Kokkos::ALL()).data(),
      Kokkos::LayoutStride(
          batch.number_of_elements(), number_of_components * number_of_points,
          count, number_of_points, extents[2], extents[1] * extents[0],
          extents[1], extents[0], extents[0], 1));
}

// The faces in each direction of a batch of face sets (see
// `ValenciaDivClean::fd::reconstructed_face_views`) as volume views of all
// components
template <typename TagsList>
std::array<::fd::device::StridedVolumeView, dim> face_views(
    const BatchedVariables<TagsList>& faces, const Index<dim>& volume_extents) {
  constexpr size_t number_of_components =
      BatchedVariables<TagsList>::number_of_independent_components;
  const size_t points_per_face_set = faces.number_of_grid_points();
  const size_t number_of_elements = faces.number_of_elements() / dim;
  std::array<::fd::device::StridedVolumeView, dim> result{};
  for (size_t d = 0; d < dim; ++d) {
    Index<dim> face_extents = volume_extents;
    ++face_extents[d];
    ASSERT(face_extents.product() == points_per_face_set,
           "The face sets have " << points_per_face_set
                                 << " points, but the faces in direction " << d
                                 << " have " << face_extents.product());
    gsl::at(result, d) = ::fd::device::StridedVolumeView(
        // The first face set in direction `d`
        Kokkos::subview(faces.view(), Kokkos::ALL(), Kokkos::ALL(),
                        d * number_of_elements)
            .data(),
        Kokkos::LayoutStride(
            number_of_elements, number_of_components * points_per_face_set,
            number_of_components, points_per_face_set, face_extents[2],
            face_extents[1] * face_extents[0], face_extents[1], face_extents[0],
            face_extents[0], 1));
  }
  return result;
}

// The variables [`begin`, `end`) of the padded ghost variables
::fd::device::StridedVolumeView padded_range(
    const ::fd::device::VolumeView& padded_ghost_variables, const size_t begin,
    const size_t end) {
  return Kokkos::subview(padded_ghost_variables, Kokkos::ALL(),
                         Kokkos::make_pair(begin, end), Kokkos::ALL(),
                         Kokkos::ALL(), Kokkos::ALL());
}
}  // namespace

template <typename System>
ValenciaDivClean::fd::ReconstructorOnDevice hydro_reconstructor_on_device(
    const fd::Reconstructor<System>& reconstructor) {
  if (dynamic_cast<const fd::MonotonisedCentralPrim<System>*>(&reconstructor) !=
      nullptr) {
    return ValenciaDivClean::fd::ReconstructorOnDevice{
        reconstructor.ghost_zone_size(),
        false,
        ::fd::reconstruction::FallbackReconstructorType::None,
        std::numeric_limits<double>::signaling_NaN(),
        std::nullopt,
        std::nullopt,
        reconstructor.reconstruct_rho_times_temperature()};
  }
  if (const auto* const adaptive_order = dynamic_cast<
          const fd::PositivityPreservingAdaptiveOrderPrim<System>*>(
          &reconstructor);
      adaptive_order != nullptr) {
    return ValenciaDivClean::fd::ReconstructorOnDevice{
        adaptive_order->ghost_zone_size(),
        true,
        adaptive_order->low_order_reconstructor(),
        adaptive_order->four_to_the_alpha_5(),
        adaptive_order->six_to_the_alpha_7(),
        adaptive_order->eight_to_the_alpha_9(),
        adaptive_order->reconstruct_rho_times_temperature()};
  }
  ERROR(
      "FD patches of GhValenciaDivClean only support the "
      "MonotonisedCentralPrim and PositivityPreservingAdaptiveOrderPrim "
      "reconstructors.");
}

template <typename System>
size_t spacetime_reconstruction_degree(
    const fd::Reconstructor<System>& reconstructor) {
  // As the reconstructors on the host
  if (dynamic_cast<const fd::MonotonisedCentralPrim<System>*>(&reconstructor) !=
      nullptr) {
    return 2;
  }
  if (dynamic_cast<const fd::PositivityPreservingAdaptiveOrderPrim<System>*>(
          &reconstructor) != nullptr) {
    return 2 * reconstructor.ghost_zone_size() - 2;
  }
  ERROR(
      "FD patches of GhValenciaDivClean only support the "
      "MonotonisedCentralPrim and PositivityPreservingAdaptiveOrderPrim "
      "reconstructors.");
}

template <typename System>
bool fix_reconstructed_states_to_atmosphere(
    const fd::Reconstructor<System>& reconstructor) {
  using ::VariableFixing::FixReconstructedStateToAtmosphere;
  // As the reconstructors on the host for elements doing FD
  const auto fixes = [](const FixReconstructedStateToAtmosphere where) {
    return where == FixReconstructedStateToAtmosphere::Always or
           where == FixReconstructedStateToAtmosphere::OnFdOnly;
  };
  if (const auto* const monotonised_central =
          dynamic_cast<const fd::MonotonisedCentralPrim<System>*>(
              &reconstructor);
      monotonised_central != nullptr) {
    return fixes(monotonised_central->fix_reconstructed_state_to_atmosphere());
  }
  if (const auto* const adaptive_order = dynamic_cast<
          const fd::PositivityPreservingAdaptiveOrderPrim<System>*>(
          &reconstructor);
      adaptive_order != nullptr) {
    return fixes(adaptive_order->fix_reconstructed_state_to_atmosphere());
  }
  ERROR(
      "FD patches of GhValenciaDivClean only support the "
      "MonotonisedCentralPrim and PositivityPreservingAdaptiveOrderPrim "
      "reconstructors.");
}

template <typename Correction>
TimeDerivativeOnDevice<Correction>::TimeDerivativeOnDevice(
    const Kokkos::DefaultExecutionSpace& exec, const size_t number_of_elements,
    const Mesh<dim>& dg_mesh,
    ValenciaDivClean::fd::ReconstructorOnDevice hydro_reconstructor,
    const size_t spacetime_reconstruction_degree,
    std::optional<double> spacetime_dissipation)
    : number_of_elements_(number_of_elements),
      subcell_extents_(evolution::dg::subcell::fd::mesh(dg_mesh).extents()),
      ghost_zone_size_(hydro_reconstructor.ghost_zone_size()),
      spacetime_reconstruction_degree_(spacetime_reconstruction_degree),
      spacetime_dissipation_(std::move(spacetime_dissipation)),
      grmhd_(exec, number_of_elements, dg_mesh,
             std::move(hydro_reconstructor)) {
  ASSERT(spacetime_reconstruction_degree_ / 2 + 1 <= ghost_zone_size_,
         "The ghost zones of width "
             << ghost_zone_size_
             << " are too small for the unlimited reconstruction of degree "
             << spacetime_reconstruction_degree_);
  const size_t number_of_points = subcell_extents_.product();
  derivatives_ =
      BatchedVariables<deriv_tags>{exec, number_of_points, number_of_elements};
  gammas_ =
      BatchedVariables<gamma_tags>{exec, number_of_points, number_of_elements};
  cell_spacetime_variables_ =
      BatchedVariables<ValenciaDivClean::subcell::cell_spacetime_tags>{
          exec, number_of_points, number_of_elements};
  Index<dim> face_extents = subcell_extents_;
  ++face_extents[0];
  lower_face_spacetime_metric_ = BatchedVariables<face_spacetime_metric_tags>{
      exec, face_extents.product(), dim * number_of_elements};
  upper_face_spacetime_metric_ = BatchedVariables<face_spacetime_metric_tags>{
      exec, face_extents.product(), dim * number_of_elements};
}

template <typename Correction>
void TimeDerivativeOnDevice<Correction>::apply(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<dt_tags_on_device>*> dt_vars,
    const gsl::not_null<BatchedVariables<evolved_tags_on_device>*>
        evolved_variables,
    const gsl::not_null<DeferredPointwiseStatus<
        ValenciaDivClean::fd::face_atmosphere_snapshot_tags>*>
        face_status,
    const ::fd::device::VolumeView& padded_ghost_variables,
    const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives,
    const ::fd::device::BatchInverseJacobian<dim>& inverse_jacobian,
    const Batch<domain::Tags::Coordinates<dim, Frame::Grid>>& grid_coordinates,
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
    const double constraint_damping_parameter,
    const evolution::dg::subcell::fd_patch::PackagedDataFromDg<dim>&
        packaged_data_from_dg,
    const Kokkos::View<const double*>& received_mortar_data) {
  const Kokkos::Profiling::ScopedRegion region{
      "GhValenciaDivClean::subcell::TimeDerivativeOnDevice::apply"};
  ASSERT(dt_vars->number_of_elements() == number_of_elements_ and
             evolved_variables->number_of_elements() == number_of_elements_ and
             padded_ghost_variables.extent(0) == number_of_elements_ and
             padded_ghost_variables.extent(1) ==
                 number_of_hydro_ghost_variables + number_of_gh_variables,
         "The batches don't match the " << number_of_elements_
                                        << " elements of the time derivative");
  const size_t fd_order = 2 * ghost_zone_size_;
  const ::fd::device::StridedVolumeView padded_gh_variables =
      padded_range(padded_ghost_variables, number_of_hydro_ghost_variables,
                   number_of_hydro_ghost_variables + number_of_gh_variables);

  // 1. Filter the GH variables, as the host does before computing the time
  // derivative. The filtered variables replace the evolved ones and the
  // interior of the padded ones; the ghost zones keep the unfiltered data of
  // the neighbors.
  if (spacetime_dissipation_.has_value()) {
    const ::fd::device::StridedVolumeView evolved_gh_variables =
        component_range(*evolved_variables, 0, number_of_gh_variables,
                        subcell_extents_);
    ::fd::device::kreiss_oliger_filter(exec, evolved_gh_variables,
                                       padded_gh_variables, subcell_extents_,
                                       fd_order, *spacetime_dissipation_);
    // The padded view may have wider ghost zones than the reconstruction needs
    const auto ghost_width = static_cast<int>(::fd::device::detail::ghost_width(
        padded_ghost_variables, subcell_extents_));
    const ::fd::device::detail::BatchCells cells{
        number_of_elements_,
        number_of_gh_variables,
        {static_cast<int>(subcell_extents_[0]),
         static_cast<int>(subcell_extents_[1]),
         static_cast<int>(subcell_extents_[2])}};
    Kokkos::parallel_for(
        "GhValenciaDivClean::subcell::TimeDerivativeOnDevice::filtered_"
        "interior",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                            Kokkos::IndexType<int>>(exec, 0, cells.size()),
        KOKKOS_LAMBDA(const int index) {
          const auto [element, var, cell] = cells(index);
          padded_gh_variables(element, var, cell[2] + ghost_width,
                              cell[1] + ghost_width, cell[0] + ghost_width) =
              evolved_gh_variables(element, var, cell[2], cell[1], cell[0]);
        });
  }

  // 2. The derivatives of the GH variables
  ::fd::device::partial_derivatives(
      exec,
      component_range(
          derivatives_, 0,
          BatchedVariables<deriv_tags>::number_of_independent_components,
          subcell_extents_),
      padded_gh_variables, inverse_jacobian, subcell_extents_, fd_order);

  // 3. The constraint damping parameters
  gh::constraint_gammas_on_device(
      exec, get<::Tags::BatchView<gh::Tags::ConstraintGamma0>>(gammas_),
      get<::Tags::BatchView<gh::Tags::ConstraintGamma1>>(gammas_),
      get<::Tags::BatchView<gh::Tags::ConstraintGamma2>>(gammas_),
      grid_coordinates, damping_functions);

  // 4. The GH time derivatives, and the mesh velocity times the derivatives
  using spacetime_metric_tag = gr::Tags::SpacetimeMetric<DataVector, dim>;
  using pi_tag = gh::Tags::Pi<DataVector, dim>;
  using phi_tag = gh::Tags::Phi<DataVector, dim>;
  const auto& evolved = *evolved_variables;
  const auto spacetime_metric =
      get<::Tags::BatchView<spacetime_metric_tag>>(evolved);
  const auto pi = get<::Tags::BatchView<pi_tag>>(evolved);
  const auto phi = get<::Tags::BatchView<phi_tag>>(evolved);
  gh_.apply(exec,
            get<::Tags::BatchView<::Tags::dt<spacetime_metric_tag>>>(*dt_vars),
            get<::Tags::BatchView<::Tags::dt<pi_tag>>>(*dt_vars),
            get<::Tags::BatchView<::Tags::dt<phi_tag>>>(*dt_vars),
            get<::Tags::BatchView<tmpl::at_c<deriv_tags, 0>>>(derivatives_),
            get<::Tags::BatchView<tmpl::at_c<deriv_tags, 1>>>(derivatives_),
            get<::Tags::BatchView<tmpl::at_c<deriv_tags, 2>>>(derivatives_),
            spacetime_metric, pi, phi,
            get<::Tags::BatchView<gh::Tags::ConstraintGamma0>>(gammas_),
            get<::Tags::BatchView<gh::Tags::ConstraintGamma1>>(gammas_),
            get<::Tags::BatchView<gh::Tags::ConstraintGamma2>>(gammas_),
            mesh_velocity);
  if (mesh_velocity.has_value()) {
    // As the host: d_t u += v^j d_j u, adding the directions in order
    const auto dt_storage = dt_vars->view();
    const auto derivative_storage = derivatives_.view();
    const Kokkos::Array<Kokkos::View<double**, Kokkos::LayoutStride>, dim>
        velocity{get<0>(*mesh_velocity), get<1>(*mesh_velocity),
                 get<2>(*mesh_velocity)};
    Kokkos::parallel_for(
        "GhValenciaDivClean::subcell::TimeDerivativeOnDevice::mesh_velocity",
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>,
                              Kokkos::IndexType<size_t>>(
            exec, {0, 0}, {subcell_extents_.product(), number_of_elements_}),
        KOKKOS_LAMBDA(const size_t point, const size_t element) {
          const double v_x = velocity[0](point, element);
          const double v_y = velocity[1](point, element);
          const double v_z = velocity[2](point, element);
          for (size_t component = 0; component < number_of_gh_variables;
               ++component) {
            double& dt_var = dt_storage(point, component, element);
            dt_var += v_x * derivative_storage(point, dim * component, element);
            dt_var +=
                v_y * derivative_storage(point, dim * component + 1, element);
            dt_var +=
                v_z * derivative_storage(point, dim * component + 2, element);
          }
        });
  }

  // 5. The spacetime variables at the cell centers for the GRMHD sources
  const auto& gh_cache = gh_.cache();
  const auto lapse =
      get<::Tags::BatchView<gr::Tags::Lapse<DataVector>>>(gh_cache);
  const auto shift =
      get<::Tags::BatchView<gr::Tags::Shift<DataVector, dim>>>(gh_cache);
  grmhd_spacetime_on_device(
      exec, make_not_null(&cell_spacetime_variables_), spacetime_metric, pi,
      phi, lapse, shift,
      get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, dim>>>(
          gh_cache));

  // 6. The spacetime metric and the 3+1 variables on both sides of the faces
  const ::fd::device::ConstStridedVolumeView padded_spacetime_metric =
      padded_range(padded_ghost_variables, number_of_hydro_ghost_variables,
                   number_of_hydro_ghost_variables +
                       number_of_spacetime_metric_components);
  const auto upper_metric_views =
      face_views(upper_face_spacetime_metric_, subcell_extents_);
  const auto lower_metric_views =
      face_views(lower_face_spacetime_metric_, subcell_extents_);
  switch (spacetime_reconstruction_degree_) {
    case 2:
      ::fd::device::unlimited<2>(exec, upper_metric_views, lower_metric_views,
                                 padded_spacetime_metric, subcell_extents_);
      break;
    case 4:
      ::fd::device::unlimited<4>(exec, upper_metric_views, lower_metric_views,
                                 padded_spacetime_metric, subcell_extents_);
      break;
    case 6:
      ::fd::device::unlimited<6>(exec, upper_metric_views, lower_metric_views,
                                 padded_spacetime_metric, subcell_extents_);
      break;
    case 8:
      ::fd::device::unlimited<8>(exec, upper_metric_views, lower_metric_views,
                                 padded_spacetime_metric, subcell_extents_);
      break;
    default:
      ERROR("Unsupported degree of the reconstruction of the spacetime metric: "
            << spacetime_reconstruction_degree_);
  }
  for (const Side side : {Side::Lower, Side::Upper}) {
    const auto& faces = *grmhd_.faces(side);
    fd::spacetime_on_faces_on_device(
        exec, get<::Tags::BatchView<gr::Tags::Lapse<DataVector>>>(faces),
        get<::Tags::BatchView<gr::Tags::Shift<DataVector, dim>>>(faces),
        get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, dim>>>(faces),
        get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
            faces),
        get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, dim>>>(
            faces),
        get<::Tags::BatchView<spacetime_metric_tag>>(
            side == Side::Lower ? lower_face_spacetime_metric_
                                : upper_face_spacetime_metric_));
  }

  // 7. The GRMHD time derivatives
  grmhd_.apply(exec, dt_vars, face_status, padded_ghost_variables, evolved,
               primitives, cell_spacetime_variables_, face_geometry,
               cell_geometry, mesh_velocity.has_value(), div_mesh_velocity,
               correction, equation_of_state, fix_to_atmosphere,
               constraint_damping_parameter, packaged_data_from_dg,
               received_mortar_data);

  // 8. The stress energy of the fluid in d_t Pi
  add_stress_energy_to_dt_pi_on_device(
      exec, get<::Tags::BatchView<::Tags::dt<pi_tag>>>(*dt_vars),
      get<::Tags::BatchView<hydro::Tags::RestMassDensity<DataVector>>>(
          primitives),
      get<::Tags::BatchView<hydro::Tags::SpecificInternalEnergy<DataVector>>>(
          primitives),
      get<::Tags::BatchView<hydro::Tags::Pressure<DataVector>>>(primitives),
      get<::Tags::BatchView<hydro::Tags::LorentzFactor<DataVector>>>(
          primitives),
      get<::Tags::BatchView<hydro::Tags::SpatialVelocity<DataVector, dim>>>(
          primitives),
      get<::Tags::BatchView<hydro::Tags::MagneticField<DataVector, dim>>>(
          primitives),
      spacetime_metric, lapse, shift);
}

template ValenciaDivClean::fd::ReconstructorOnDevice
hydro_reconstructor_on_device(
    const fd::Reconstructor<System<RadiationTransport::NoNeutrinos::System>>&
        reconstructor);
template bool fix_reconstructed_states_to_atmosphere(
    const fd::Reconstructor<System<RadiationTransport::NoNeutrinos::System>>&
        reconstructor);
template size_t spacetime_reconstruction_degree(
    const fd::Reconstructor<System<RadiationTransport::NoNeutrinos::System>>&
        reconstructor);
template class TimeDerivativeOnDevice<
    ValenciaDivClean::BoundaryCorrections::Rusanov>;
template class TimeDerivativeOnDevice<
    ValenciaDivClean::BoundaryCorrections::Hll>;
}  // namespace grmhd::GhValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
