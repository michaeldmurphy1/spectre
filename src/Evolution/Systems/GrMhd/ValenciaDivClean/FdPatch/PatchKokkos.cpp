// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/PatchKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructorOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/RdmpTciDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepperOnDeviceKokkos.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd_patch {
namespace {
namespace patch_ns = evolution::dg::subcell::fd_patch;

// Copies host `Variables` of each element into a batch on the device
template <typename Tags, typename HostVars>
BatchedVariables<Tags> to_device(const Kokkos::DefaultExecutionSpace& exec,
                                 const std::vector<HostVars>& host_vars,
                                 const size_t number_of_points) {
  BatchedVariables<Tags, Kokkos::HostSpace> host_batch{
      Kokkos::DefaultHostExecutionSpace{}, number_of_points, host_vars.size()};
  for (size_t e = 0; e < host_vars.size(); ++e) {
    Variables<Tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    element.assign_subset(host_vars[e]);
  }
  BatchedVariables<Tags> result{exec, number_of_points, host_vars.size()};
  deep_copy(exec, make_not_null(&result), host_batch);
  // The host batch must stay valid until the copy is done
  exec.fence("grmhd::ValenciaDivClean::fd_patch::to_device");
  return result;
}

Kokkos::View<const double*> flat(const auto& batch) {
  return Kokkos::View<const double*>(batch.view().data(), batch.view().span());
}
}  // namespace

template <typename OrderedListOfPrimitiveRecoverySchemes>
Patch<OrderedListOfPrimitiveRecoverySchemes>::Patch(
    const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
    const patch_ns::FdPatches<3>& fd_patches, const size_t patch,
    const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const fd::Reconstructor& reconstructor,
    const evolution::BoundaryCorrection& boundary_correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
    const FixConservatives& fix_conservatives,
    const PrimitiveFromConservativeOptions& primitive_from_conservative_options,
    const double constraint_damping_parameter)
    : equation_of_state_(equation_of_state.get_clone()),
      fix_to_atmosphere_(fix_to_atmosphere.device_view()),
      fix_conservatives_(fix_conservatives),
      primitive_from_conservative_options_(primitive_from_conservative_options),
      constraint_damping_parameter_(constraint_damping_parameter) {
  // Errors if the reconstructor isn't supported
  const fd::ReconstructorOnDevice reconstructor_on_device{reconstructor};
  const Mesh<3>& dg_mesh = fd_patches.dg_mesh(patch);
  const size_t number_of_elements = fd_patches.element_ids(patch).size();

  // Boundary correction and time derivative
  size_t number_of_packaged_variables = 0;
  if (const auto* const rusanov =
          dynamic_cast<const BoundaryCorrections::Rusanov*>(
              &boundary_correction);
      rusanov != nullptr) {
    boundary_correction_ = *rusanov;
    time_derivative_ =
        subcell::TimeDerivativeOnDevice<BoundaryCorrections::Rusanov>{
            exec, number_of_elements, dg_mesh, reconstructor_on_device};
    number_of_packaged_variables =
        Variables<typename BoundaryCorrections::Rusanov::
                      dg_package_field_tags>::number_of_independent_components;
  } else if (const auto* const hll =
                 dynamic_cast<const BoundaryCorrections::Hll*>(
                     &boundary_correction);
             hll != nullptr) {
    boundary_correction_ = *hll;
    time_derivative_ =
        subcell::TimeDerivativeOnDevice<BoundaryCorrections::Hll>{
            exec, number_of_elements, dg_mesh, reconstructor_on_device};
    number_of_packaged_variables =
        Variables<typename BoundaryCorrections::Hll::dg_package_field_tags>::
            number_of_independent_components;
  } else {
    ERROR("FD patches only support the Rusanov and Hll boundary corrections.");
  }

  using ghost_tags = subcell::PrimitiveGhostVariables::ghost_variables_tag_list;
  patch_elements_ = patch_ns::PatchElements{
      exec,
      domain,
      fd_patches,
      patch,
      initial_refinement_levels,
      grid_to_inertial_map,
      time,
      functions_of_time,
      reconstructor_on_device.ghost_zone_size(),
      Variables<ghost_tags>::number_of_independent_components,
      number_of_packaged_variables};

  const size_t number_of_cells =
      patch_elements_.subcell_mesh().number_of_grid_points();
  ghost_variables_ =
      BatchedVariables<ghost_tags>{exec, number_of_cells, number_of_elements};
  dt_variables_ = BatchedVariables<subcell::dt_tags_on_device>{
      exec, number_of_cells, number_of_elements};
  stepper_errors_.resize(number_of_elements);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
auto Patch<OrderedListOfPrimitiveRecoverySchemes>::initial_state(
    const Kokkos::DefaultExecutionSpace& exec,
    const InitialDataFunction& initial_data, const double initial_time,
    const domain::FunctionsOfTimeMap& functions_of_time) const
    -> tuples::tagged_tuple_from_typelist<state_tags> {
  const Mesh<3>& subcell_mesh = patch_elements_.subcell_mesh();
  const auto& element_maps = patch_elements_.element_maps();
  const auto& grid_to_inertial_map = patch_elements_.grid_to_inertial_map();
  const size_t number_of_elements = patch_elements_.number_of_elements();
  const size_t number_of_cells = subcell_mesh.number_of_grid_points();
  Index<3> face_extents = subcell_mesh.extents();
  ++face_extents[0];

  // Initial data at the cell centers and the spacetime on the faces, as
  // `SetVariables` and `subcell::BackgroundGrVars` do
  const auto cell_logical_coords = logical_coordinates(subcell_mesh);
  std::vector<Variables<initial_data_tags>> cell_data{};
  std::vector<Variables<subcell::cell_spacetime_tags>> cell_spacetime{};
  std::vector<Variables<subcell::face_spacetime_tags>> face_spacetime(
      3 * number_of_elements);
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    const auto inertial_coords =
        grid_to_inertial_map(element_maps[slot](cell_logical_coords),
                             initial_time, functions_of_time);
    cell_data.push_back(initial_data(inertial_coords));
    auto& spacetime = cell_spacetime.emplace_back(number_of_cells);
    spacetime.assign_subset(cell_data.back());
    get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
        spacetime) = inertial_coords;
    for (size_t d = 0; d < 3; ++d) {
      auto extents = subcell_mesh.extents().indices();
      gsl::at(extents, d) += 1;
      auto quadrature = subcell_mesh.quadrature();
      gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
      const Mesh<3> face_mesh{extents, subcell_mesh.basis(), quadrature};
      auto& face_data = face_spacetime[d * number_of_elements + slot];
      face_data.initialize(face_mesh.number_of_grid_points());
      face_data.assign_subset(initial_data(grid_to_inertial_map(
          element_maps[slot](logical_coordinates(face_mesh)), initial_time,
          functions_of_time)));
    }
  }
  tuples::tagged_tuple_from_typelist<state_tags> result{
      EvolvedVariables{exec, number_of_cells, number_of_elements},
      to_device<hydro::grmhd_tags<DataVector>>(exec, cell_data,
                                               number_of_cells),
      to_device<subcell::cell_spacetime_tags>(exec, cell_spacetime,
                                              number_of_cells),
      to_device<subcell::face_spacetime_tags>(exec, face_spacetime,
                                              face_extents.product())};
  auto& primitives = get<primitive_variables_tag>(result);
  const auto& cell_spacetime_variables =
      get<cell_spacetime_variables_tag>(result);
  const auto atmosphere_status = VariableFixing::fix_to_atmosphere_on_device(
      exec, make_not_null(&primitives),
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
          cell_spacetime_variables),
      *equation_of_state_, fix_to_atmosphere_);
  if (atmosphere_status.number_of_failures > 0) {
    VariableFixing::error_fixing_to_atmosphere_on_device(
        exec, atmosphere_status, primitives,
        patch_elements_.element_description(
            atmosphere_status.first_failed_element));
  }
  compute_conservatives(exec,
                        make_not_null(&get<evolved_variables_tag>(result)),
                        primitives, cell_spacetime_variables);
  return result;
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::set_face_spacetime_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const FaceSpacetimeVariables& face_spacetime_variables) {
  std::visit(
      [&exec, &face_spacetime_variables](auto& time_derivative) {
        time_derivative.set_face_spacetime_variables(exec,
                                                     face_spacetime_variables);
      },
      time_derivative_);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
std::vector<patch_ns::OutgoingBoundaryData<3>>
Patch<OrderedListOfPrimitiveRecoverySchemes>::prepare_send(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& primitives,
    const CellSpacetimeVariables& /*cell_spacetime_variables*/,
    const FaceSpacetimeVariables& /*face_spacetime_variables*/,
    const TimeStepId& next_time_step_id, const size_t integration_order) {
  const Kokkos::Profiling::ScopedRegion region{
      "ValenciaDivClean::fd_patch::Patch::prepare_send"};
  subcell::primitive_ghost_variables_on_device(
      exec, make_not_null(&ghost_variables_), primitives);
  patch_elements_.fill_ghost_zones(exec, flat(ghost_variables_));
  if (not patch_elements_.has_external_faces()) {
    return {};
  }
  return patch_elements_.boundary_data_to_send(
      exec, flat(ghost_variables_),
      subcell::rdmp_tci_data_on_device(
          exec, get<::Tags::BatchView<Tags::TildeD>>(evolved_variables),
          get<::Tags::BatchView<Tags::TildeYe>>(evolved_variables),
          get<::Tags::BatchView<Tags::TildeTau>>(evolved_variables),
          get<::Tags::BatchView<Tags::TildeB<>>>(evolved_variables),
          &rdmp_tci_data_buffers_),
      next_time_step_id, tci_status_sent, integration_order);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
std::vector<patch_ns::OutgoingInitialRdmpData<3>>
Patch<OrderedListOfPrimitiveRecoverySchemes>::initial_rdmp_data(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& /*primitives*/,
    const CellSpacetimeVariables& /*cell_spacetime_variables*/,
    const FaceSpacetimeVariables& /*face_spacetime_variables*/) const {
  // As `subcell::SetInitialRdmpData` on the subcell grid
  return patch_elements_.initial_rdmp_data(subcell::rdmp_tci_data_on_device(
      exec, get<::Tags::BatchView<Tags::TildeD>>(evolved_variables),
      get<::Tags::BatchView<Tags::TildeYe>>(evolved_variables),
      get<::Tags::BatchView<Tags::TildeTau>>(evolved_variables),
      get<::Tags::BatchView<Tags::TildeB<>>>(evolved_variables)));
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::take_step(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const gsl::not_null<PrimitiveVariables*> primitives,
    const gsl::not_null<CellSpacetimeVariables*> cell_spacetime_variables,
    const gsl::not_null<FaceSpacetimeVariables*> /*face_spacetime_variables*/,
    const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
    const TimeStepper& time_stepper, const TimeStepId& time_step_id,
    const TimeStepId& next_time_step_id, const TimeDelta& time_step,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  patch_elements_.update_geometry(exec, time_step_id.substep_time(),
                                  functions_of_time);

  // Time derivative
  std::optional<Kokkos::Profiling::ScopedRegion> region{};
  region.emplace("ValenciaDivClean::fd_patch::Patch::time_derivative");
  const auto status = std::visit(
      [&]<typename Correction>(
          subcell::TimeDerivativeOnDevice<Correction>& time_derivative) {
        return time_derivative.apply(
            exec, make_not_null(&dt_variables_),
            patch_elements_.padded_ghost_variables(), *evolved_variables,
            *primitives, *cell_spacetime_variables,
            patch_elements_.face_geometry(), patch_elements_.cell_geometry(),
            patch_elements_.grid_to_inertial(),
            std::get<Correction>(boundary_correction_), *equation_of_state_,
            &fix_to_atmosphere_, constraint_damping_parameter_,
            patch_elements_.packaged_data_from_dg(),
            patch_elements_.received_mortar_data());
      },
      time_derivative_);
  for (size_t side = 0; side < 2; ++side) {
    if (gsl::at(status, side).number_of_failures > 0) {
      std::visit(
          [&](const auto& time_derivative) {
            VariableFixing::error_fixing_to_atmosphere_on_device(
                exec, gsl::at(status, side),
                side == 0 ? time_derivative.lower_faces()
                          : time_derivative.upper_faces(),
                "a face of " + patch_elements_.element_description(
                                   gsl::at(status, side).first_failed_element %
                                   patch_elements_.number_of_elements()));
          },
          time_derivative_);
    }
  }

  // Time stepping
  region.reset();
  region.emplace("ValenciaDivClean::fd_patch::Patch::update_u");
  TimeSteppers::record_on_device(exec, history, time_step_id,
                                 *evolved_variables, dt_variables_);
  TimeSteppers::update_u_on_device(
      exec, evolved_variables, make_not_null(&stepper_errors_), *history,
      time_stepper, time_step_id, next_time_step_id, time_step, false,
      StepperErrorTolerances{});
  history->clean(time_stepper);

  // Fix the conserved variables and recover the primitive variables
  region.reset();
  region.emplace("ValenciaDivClean::fd_patch::Patch::fix_conservatives");
  const auto& spatial_metric =
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
          *cell_spacetime_variables);
  const auto& inv_spatial_metric =
      get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, 3>>>(
          *cell_spacetime_variables);
  const auto& sqrt_det_spatial_metric =
      get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
          *cell_spacetime_variables);
  const auto fix_status = fix_conservatives_on_device(
      exec, evolved_variables, spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, fix_conservatives_, &fix_conservatives_scratch_);
  if (fix_status.number_of_failures > 0) {
    error_fixing_conservatives_on_device(
        exec, fix_status, *evolved_variables, spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric,
        patch_elements_.element_description(fix_status.first_failed_element));
  }
  region.reset();
  region.emplace("ValenciaDivClean::fd_patch::Patch::primitive_recovery");
  const auto recovery_status = primitive_from_conservative_on_device<
      OrderedListOfPrimitiveRecoverySchemes, true>(
      exec, primitives,
      get<::Tags::BatchView<Tags::TildeD>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeYe>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeTau>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeS<>>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeB<>>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildePhi>>(*evolved_variables),
      spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
      *equation_of_state_, primitive_from_conservative_options_,
      &primitive_recovery_scratch_);
  if (recovery_status.number_of_failures > 0) {
    error_recovering_primitives_on_device(
        exec, recovery_status, *primitives,
        get<::Tags::BatchView<Tags::TildeD>>(*evolved_variables),
        get<::Tags::BatchView<Tags::TildeTau>>(*evolved_variables),
        get<::Tags::BatchView<Tags::TildeS<>>>(*evolved_variables),
        get<::Tags::BatchView<Tags::TildeB<>>>(*evolved_variables),
        spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
        patch_elements_.element_description(
            recovery_status.first_failed_element));
  }
  region.reset();
  region.emplace("ValenciaDivClean::fd_patch::Patch::fix_to_atmosphere");
  const auto atmosphere_status = VariableFixing::fix_to_atmosphere_on_device(
      exec, primitives, spatial_metric, *equation_of_state_,
      fix_to_atmosphere_);
  if (atmosphere_status.number_of_failures > 0) {
    VariableFixing::error_fixing_to_atmosphere_on_device(
        exec, atmosphere_status, *primitives,
        patch_elements_.element_description(
            atmosphere_status.first_failed_element));
  }
  region.reset();
  region.emplace("ValenciaDivClean::fd_patch::Patch::compute_conservatives");
  compute_conservatives(exec, evolved_variables, *primitives,
                        *cell_spacetime_variables);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
auto Patch<OrderedListOfPrimitiveRecoverySchemes>::element_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& primitives,
    const CellSpacetimeVariables& cell_spacetime_variables,
    const FaceSpacetimeVariables& /*face_spacetime_variables*/) const
    -> std::vector<ElementVariables> {
  const auto to_host = [&exec](const auto& batch) {
    using TagsList = typename std::decay_t<decltype(batch)>::tags_list;
    BatchedVariables<TagsList, Kokkos::HostSpace> host{
        Kokkos::DefaultHostExecutionSpace{}, batch.number_of_grid_points(),
        batch.number_of_elements()};
    deep_copy(exec, make_not_null(&host), batch);
    return host;
  };
  auto host_evolved = to_host(evolved_variables);
  auto host_primitives = to_host(primitives);
  auto host_spacetime = to_host(cell_spacetime_variables);
  exec.fence("grmhd::ValenciaDivClean::fd_patch::Patch::element_variables");

  // Copies, since the batches are deallocated on return
  const auto element = [](auto& host_batch, const size_t slot) {
    using TagsList = typename std::decay_t<decltype(host_batch)>::tags_list;
    Variables<TagsList> view{};
    set_data_ref_to_element(make_not_null(&view), host_batch, slot);
    return Variables<TagsList>{view};
  };
  std::vector<ElementVariables> result(patch_elements_.number_of_elements());
  for (size_t slot = 0; slot < result.size(); ++slot) {
    result[slot].evolved = element(host_evolved, slot);
    result[slot].primitives = element(host_primitives, slot);
    const auto spacetime = element(host_spacetime, slot);
    result[slot].spacetime.initialize(spacetime.number_of_grid_points());
    result[slot].spacetime.assign_subset(spacetime);
  }
  return result;
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
size_t Patch<OrderedListOfPrimitiveRecoverySchemes>::
    number_of_evolved_values_per_element() const {
  return patch_elements_.subcell_mesh().number_of_grid_points() *
         Variables<
             subcell::evolved_tags_on_device>::number_of_independent_components;
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
TimeSteppers::DeviceHistory<>
Patch<OrderedListOfPrimitiveRecoverySchemes>::initial_history(
    const TimeStepper& time_stepper) const {
  return patch_elements_.initial_history(
      time_stepper, number_of_evolved_values_per_element());
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::save_initial_values(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& primitives,
    const CellSpacetimeVariables& /*cell_spacetime_variables*/,
    const FaceSpacetimeVariables& /*face_spacetime_variables*/) {
  // The background spacetime doesn't change
  initial_values_.save(exec, evolved_variables, primitives);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::restore_initial_values(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const gsl::not_null<PrimitiveVariables*> primitives,
    const gsl::not_null<CellSpacetimeVariables*> /*cell_spacetime_variables*/,
    const gsl::not_null<FaceSpacetimeVariables*> /*face_spacetime_variables*/)
    const {
  initial_values_.restore(exec, evolved_variables, primitives);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::discard_initial_values() {
  initial_values_.discard();
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::pup(PUP::er& /*p*/) {}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::compute_conservatives(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const PrimitiveVariables& primitives,
    const CellSpacetimeVariables& cell_spacetime_variables) const {
  const auto argument = [&primitives, &cell_spacetime_variables]<typename Tag>(
                            tmpl::type_<Tag> /*meta*/) {
    if constexpr (tmpl::list_contains_v<hydro::grmhd_tags<DataVector>, Tag>) {
      return get<::Tags::BatchView<Tag>>(primitives);
    } else {
      return get<::Tags::BatchView<Tag>>(cell_spacetime_variables);
    }
  };
  apply_pointwise_on_device<ConservativeFromPrimitive>(
      exec, evolved_variables, ConservativeFromPrimitive::return_tags{},
      ConservativeFromPrimitive::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, argument(tmpl::type_<ArgumentTags>{})...);
      }(ConservativeFromPrimitive::argument_tags{}));
}

template class Patch<tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl,
                                PrimitiveRecoverySchemes::NewmanHamlin,
                                PrimitiveRecoverySchemes::PalenzuelaEtAl>>;
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
