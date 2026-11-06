// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/PatchKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/RdmpTciDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/VariableFixing/FixToAtmosphereOnDeviceKokkos.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepperOnDeviceKokkos.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ConstantExpressions.hpp"
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
    const InitialDataFunction& initial_data,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const double initial_time,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const fd::Reconstructor& reconstructor,
    const evolution::BoundaryCorrection& boundary_correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
    const FixConservatives& fix_conservatives,
    const PrimitiveFromConservativeOptions& primitive_from_conservative_options,
    const double constraint_damping_parameter, const TimeStepper& time_stepper)
    : patch_(patch),
      dg_mesh_(fd_patches.dg_mesh(patch)),
      element_ids_(fd_patches.element_ids(patch)),
      equation_of_state_(equation_of_state.get_clone()),
      fix_to_atmosphere_(fix_to_atmosphere.device_view()),
      fix_conservatives_(fix_conservatives),
      primitive_from_conservative_options_(primitive_from_conservative_options),
      constraint_damping_parameter_(constraint_damping_parameter),
      grid_to_inertial_map_(grid_to_inertial_map.get_clone()) {
  // Check what is supported
  const auto* const monotonised_central =
      dynamic_cast<const fd::MonotonisedCentralPrim*>(&reconstructor);
  if (monotonised_central == nullptr or
      monotonised_central->reconstruct_rho_times_temperature()) {
    ERROR(
        "FD patches only support the MonotonisedCentralPrim reconstructor "
        "without ReconstructRhoTimesTemperature.");
  }
  if (time_stepper.number_of_past_steps() != 0) {
    ERROR("FD patches only support time steppers that need no past steps.");
  }
  const size_t number_of_elements = element_ids_.size();
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh_);
  const size_t number_of_cells = subcell_mesh.number_of_grid_points();

  // Elements and their maps
  std::vector<ElementMap<3, Frame::Grid>> element_maps{};
  elements_.reserve(number_of_elements);
  element_maps.reserve(number_of_elements);
  for (const auto& element_id : element_ids_) {
    elements_.push_back(domain::create_initial_element(
        element_id, domain.blocks(), initial_refinement_levels));
    element_maps.emplace_back(element_id,
                              domain.blocks()[element_id.block_id()]);
  }

  // Ghost zones and data from DG neighbors
  using ghost_tags = subcell::PrimitiveGhostVariables::ghost_variables_tag_list;
  constexpr size_t number_of_ghost_variables =
      Variables<ghost_tags>::number_of_independent_components;
  const size_t ghost_width = monotonised_central->ghost_zone_size();
  ghost_zones_ = patch_ns::GhostZones<3>{
      exec,
      elements_,
      std::vector<DirectionalIdMap<3, std::optional<intrp::Irregular<3>>>>(
          number_of_elements),
      fd_patches,
      patch,
      ghost_width,
      number_of_ghost_variables};
  ghost_variables_ =
      BatchedVariables<ghost_tags>{exec, number_of_cells, number_of_elements};
  padded_ghost_variables_ = ::fd::device::make_volume_view(
      exec, "FD patch ghost variables", number_of_elements,
      number_of_ghost_variables, subcell_mesh.extents(), ghost_width);
  received_ghost_data_ = Kokkos::View<double*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch received ghost data"),
      ghost_zones_.external_data_size());
  send_data_ = Kokkos::View<double*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch send data"),
      ghost_zones_.send_data_size());

  // Boundary correction and time derivative
  using packaged_tags_rusanov =
      typename BoundaryCorrections::Rusanov::dg_package_field_tags;
  if (const auto* const rusanov =
          dynamic_cast<const BoundaryCorrections::Rusanov*>(
              &boundary_correction);
      rusanov != nullptr) {
    boundary_correction_ = *rusanov;
    time_derivative_ =
        subcell::TimeDerivativeOnDevice<BoundaryCorrections::Rusanov>{
            exec, number_of_elements, dg_mesh_};
    packaged_data_from_dg_ = patch_ns::PackagedDataFromDg<3>{
        exec,
        elements_,
        fd_patches,
        patch,
        Variables<packaged_tags_rusanov>::number_of_independent_components,
        Variables<packaged_tags_rusanov>::number_of_independent_components,
        0};
  } else if (const auto* const hll =
                 dynamic_cast<const BoundaryCorrections::Hll*>(
                     &boundary_correction);
             hll != nullptr) {
    using packaged_tags_hll =
        typename BoundaryCorrections::Hll::dg_package_field_tags;
    boundary_correction_ = *hll;
    time_derivative_ =
        subcell::TimeDerivativeOnDevice<BoundaryCorrections::Hll>{
            exec, number_of_elements, dg_mesh_};
    packaged_data_from_dg_ = patch_ns::PackagedDataFromDg<3>{
        exec,
        elements_,
        fd_patches,
        patch,
        Variables<packaged_tags_hll>::number_of_independent_components,
        Variables<packaged_tags_hll>::number_of_independent_components,
        0};
  } else {
    ERROR("FD patches only support the Rusanov and Hll boundary corrections.");
  }
  received_mortar_data_ = Kokkos::View<double*>(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "FD patch received mortar data"),
      packaged_data_from_dg_.received_data_size());

  // Geometry
  logical_to_grid_ = patch_ns::logical_to_grid(exec, element_maps, dg_mesh_);
  Index<3> face_extents = subcell_mesh.extents();
  ++face_extents[0];
  face_geometry_ = BatchedVariables<patch_ns::inertial_face_tags>{
      exec, face_extents.product(), 3 * number_of_elements};
  cell_geometry_ = BatchedVariables<patch_ns::inertial_cell_tags>{
      exec, number_of_cells, number_of_elements};
  update_geometry(exec, initial_time, functions_of_time);

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
  primitives_ = to_device<hydro::grmhd_tags<DataVector>>(exec, cell_data,
                                                         number_of_cells);
  cell_spacetime_variables_ = to_device<subcell::cell_spacetime_tags>(
      exec, cell_spacetime, number_of_cells);
  const auto face_spacetime_variables = to_device<subcell::face_spacetime_tags>(
      exec, face_spacetime, face_extents.product());
  std::visit(
      [&exec, &face_spacetime_variables](auto& time_derivative) {
        time_derivative.set_face_spacetime_variables(exec,
                                                     face_spacetime_variables);
      },
      time_derivative_);

  evolved_variables_ = BatchedVariables<subcell::evolved_tags_on_device>{
      exec, number_of_cells, number_of_elements};
  dt_variables_ = BatchedVariables<subcell::dt_tags_on_device>{
      exec, number_of_cells, number_of_elements};
  const auto atmosphere_status = VariableFixing::fix_to_atmosphere_on_device(
      exec, make_not_null(&primitives_),
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
          cell_spacetime_variables_),
      *equation_of_state_, fix_to_atmosphere_);
  if (atmosphere_status.number_of_failures > 0) {
    VariableFixing::error_fixing_to_atmosphere_on_device(
        exec, atmosphere_status, primitives_,
        element_description(atmosphere_status.first_failed_element));
  }
  compute_conservatives(exec);

  // The history starts at the order that needs no past steps, as
  // `Initialization::TimeStepperHistory`
  const size_t starting_order = visit(
      []<typename Tag>(
          const std::pair<tmpl::type_<Tag>, typename Tag::type&&> order) {
        if constexpr (std::is_same_v<Tag, TimeSteppers::Tags::FixedOrder>) {
          return order.second;
        } else {
          return order.second.minimum;
        }
      },
      time_stepper.order());
  history_ = TimeSteppers::DeviceHistory<>{
      starting_order,
      Variables<
          subcell::evolved_tags_on_device>::number_of_independent_components *
          number_of_cells,
      number_of_elements};
  stepper_errors_.resize(number_of_elements);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
std::vector<patch_ns::OutgoingBoundaryData<3>>
Patch<OrderedListOfPrimitiveRecoverySchemes>::prepare_send(
    const Kokkos::DefaultExecutionSpace& exec,
    const TimeStepId& next_time_step_id) {
  subcell::primitive_ghost_variables_on_device(
      exec, make_not_null(&ghost_variables_), primitives_);
  ghost_zones_.fill_from_patch(exec, padded_ghost_variables_,
                               flat(ghost_variables_));
  if (ghost_zones_.external_faces().empty()) {
    return {};
  }
  const auto rdmp_tci_data = subcell::rdmp_tci_data_on_device(
      exec, get<::Tags::BatchView<Tags::TildeD>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildeYe>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildeTau>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildeB<>>>(evolved_variables_));
  ghost_zones_.slice_for_external(exec, send_data_, flat(ghost_variables_));
  const auto host_send_data =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, send_data_);
  Kokkos::deep_copy(exec, host_send_data, send_data_);
  exec.fence("grmhd::ValenciaDivClean::fd_patch::Patch::prepare_send");
  return patch_ns::pack_boundary_data(
      ghost_zones_,
      gsl::span<const double>{host_send_data.data(), host_send_data.size()},
      element_ids_, dg_mesh_, rdmp_tci_data, next_time_step_id, tci_status_sent,
      history_.integration_order());
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
bool Patch<OrderedListOfPrimitiveRecoverySchemes>::received_all(
    const patch_ns::Tags::BoundaryDataInbox<3>::messages_type& messages) const {
  return patch_ns::received_all_boundary_data(ghost_zones_, element_ids_,
                                              messages);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::receive(
    const Kokkos::DefaultExecutionSpace& exec,
    const patch_ns::Tags::BoundaryDataInbox<3>::messages_type& messages) {
  if (ghost_zones_.external_faces().empty()) {
    return;
  }
  const auto host_ghost_data = Kokkos::create_mirror_view(
      Kokkos::WithoutInitializing, received_ghost_data_);
  const auto host_mortar_data = Kokkos::create_mirror_view(
      Kokkos::WithoutInitializing, received_mortar_data_);
  patch_ns::unpack_boundary_data(
      gsl::span<double>{host_ghost_data.data(), host_ghost_data.size()},
      gsl::span<double>{host_mortar_data.data(), host_mortar_data.size()},
      ghost_zones_, packaged_data_from_dg_, element_ids_, messages,
      number_of_rdmp_values);
  Kokkos::deep_copy(exec, received_ghost_data_, host_ghost_data);
  Kokkos::deep_copy(exec, received_mortar_data_, host_mortar_data);
  ghost_zones_.fill_from_external(exec, padded_ghost_variables_,
                                  received_ghost_data_);
  // The host mirrors must stay valid until the copies are done
  exec.fence("grmhd::ValenciaDivClean::fd_patch::Patch::receive");
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::take_step(
    const Kokkos::DefaultExecutionSpace& exec, const TimeStepper& time_stepper,
    const TimeStepId& time_step_id, const TimeStepId& next_time_step_id,
    const TimeDelta& time_step,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  update_geometry(exec, time_step_id.substep_time(), functions_of_time);

  // Time derivative
  const auto status = std::visit(
      [&]<typename Correction>(
          subcell::TimeDerivativeOnDevice<Correction>& time_derivative) {
        return time_derivative.apply(
            exec, make_not_null(&dt_variables_), padded_ghost_variables_,
            evolved_variables_, primitives_, cell_spacetime_variables_,
            face_geometry_, cell_geometry_, grid_to_inertial_.value(),
            std::get<Correction>(boundary_correction_), *equation_of_state_,
            &fix_to_atmosphere_, constraint_damping_parameter_,
            packaged_data_from_dg_, received_mortar_data_);
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
                "a face of " + element_description(
                                   gsl::at(status, side).first_failed_element %
                                   number_of_elements()));
          },
          time_derivative_);
    }
  }

  // Time stepping
  TimeSteppers::record_on_device(exec, make_not_null(&history_), time_step_id,
                                 evolved_variables_, dt_variables_);
  TimeSteppers::update_u_on_device(
      exec, make_not_null(&evolved_variables_), make_not_null(&stepper_errors_),
      history_, time_stepper, time_step_id, next_time_step_id, time_step, false,
      StepperErrorTolerances{});
  history_.clean(time_stepper);

  // Fix the conserved variables and recover the primitive variables
  const auto& spatial_metric =
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
          cell_spacetime_variables_);
  const auto& inv_spatial_metric =
      get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, 3>>>(
          cell_spacetime_variables_);
  const auto& sqrt_det_spatial_metric =
      get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
          cell_spacetime_variables_);
  const auto fix_status = fix_conservatives_on_device(
      exec, make_not_null(&evolved_variables_), spatial_metric,
      inv_spatial_metric, sqrt_det_spatial_metric, fix_conservatives_);
  if (fix_status.number_of_failures > 0) {
    error_fixing_conservatives_on_device(
        exec, fix_status, evolved_variables_, spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric,
        element_description(fix_status.first_failed_element));
  }
  const auto recovery_status = primitive_from_conservative_on_device<
      OrderedListOfPrimitiveRecoverySchemes, true>(
      exec, make_not_null(&primitives_),
      get<::Tags::BatchView<Tags::TildeD>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildeYe>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildeTau>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildeS<>>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildeB<>>>(evolved_variables_),
      get<::Tags::BatchView<Tags::TildePhi>>(evolved_variables_),
      spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
      *equation_of_state_, primitive_from_conservative_options_);
  if (recovery_status.number_of_failures > 0) {
    error_recovering_primitives_on_device(
        exec, recovery_status, primitives_,
        get<::Tags::BatchView<Tags::TildeD>>(evolved_variables_),
        get<::Tags::BatchView<Tags::TildeTau>>(evolved_variables_),
        get<::Tags::BatchView<Tags::TildeS<>>>(evolved_variables_),
        get<::Tags::BatchView<Tags::TildeB<>>>(evolved_variables_),
        spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
        element_description(recovery_status.first_failed_element));
  }
  const auto atmosphere_status = VariableFixing::fix_to_atmosphere_on_device(
      exec, make_not_null(&primitives_), spatial_metric, *equation_of_state_,
      fix_to_atmosphere_);
  if (atmosphere_status.number_of_failures > 0) {
    VariableFixing::error_fixing_to_atmosphere_on_device(
        exec, atmosphere_status, primitives_,
        element_description(atmosphere_status.first_failed_element));
  }
  compute_conservatives(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
std::string Patch<OrderedListOfPrimitiveRecoverySchemes>::norms(
    const Kokkos::DefaultExecutionSpace& exec) const {
  BatchedVariables<subcell::evolved_tags_on_device, Kokkos::HostSpace> host{
      Kokkos::DefaultHostExecutionSpace{},
      evolved_variables_.number_of_grid_points(),
      evolved_variables_.number_of_elements()};
  deep_copy(exec, make_not_null(&host), evolved_variables_);
  exec.fence("grmhd::ValenciaDivClean::fd_patch::Patch::norms");
  std::ostringstream result{};
  result << std::setprecision(16) << std::scientific;
  tmpl::for_each<subcell::evolved_tags_on_device>([&]<typename Tag>(
                                                      tmpl::type_<Tag>
                                                      /*meta*/) {
    const auto& tensor = get<::Tags::BatchView<Tag, Kokkos::HostSpace>>(host);
    for (size_t component = 0; component < tensor.size(); ++component) {
      double max = std::numeric_limits<double>::lowest();
      double sum_of_squares = 0.0;
      const auto& values = tensor[component];
      for (size_t e = 0; e < values.extent(1); ++e) {
        for (size_t i = 0; i < values.extent(0); ++i) {
          max = std::max(max, values(i, e));
          sum_of_squares += square(values(i, e));
        }
      }
      result << "FD patch " << patch_ << " " << db::tag_name<Tag>()
             << tensor.component_suffix(component) << ": Max " << max
             << " L2Norm "
             << sqrt(sum_of_squares / static_cast<double>(values.size()))
             << "\n";
    }
  });
  return result.str();
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::pup(PUP::er& p) {
  if (not p.isSizing()) {
    ERROR("FD patches can't be serialized yet, e.g. for checkpoints.");
  }
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
std::string Patch<OrderedListOfPrimitiveRecoverySchemes>::element_description(
    const size_t slot) const {
  std::ostringstream description{};
  description << "element " << element_ids_.at(slot) << " of FD patch "
              << patch_;
  return description.str();
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::update_geometry(
    const Kokkos::DefaultExecutionSpace& exec, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  if (grid_to_inertial_.has_value() and grid_to_inertial_map_->is_identity()) {
    return;
  }
  grid_to_inertial_ = patch_ns::affine_grid_to_inertial(
      *grid_to_inertial_map_, time, functions_of_time,
      logical_to_grid_.corners);
  patch_ns::inertial_geometry_on_device(
      exec, make_not_null(&face_geometry_), make_not_null(&cell_geometry_),
      logical_to_grid_, grid_to_inertial_.value());
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::compute_conservatives(
    const Kokkos::DefaultExecutionSpace& exec) {
  const auto argument = [this]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    if constexpr (tmpl::list_contains_v<hydro::grmhd_tags<DataVector>, Tag>) {
      return get<::Tags::BatchView<Tag>>(primitives_);
    } else {
      return get<::Tags::BatchView<Tag>>(cell_spacetime_variables_);
    }
  };
  apply_pointwise_on_device<ConservativeFromPrimitive>(
      exec, make_not_null(&evolved_variables_),
      ConservativeFromPrimitive::return_tags{},
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
