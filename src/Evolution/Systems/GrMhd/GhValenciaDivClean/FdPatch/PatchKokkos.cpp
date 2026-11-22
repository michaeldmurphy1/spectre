// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/PatchKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <array>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatchElementsKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/FixingAndRecoveryKokkos.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/RdmpTciDataKokkos.hpp"
#include "Evolution/VariableFixing/LimitLorentzFactorOnDeviceKokkos.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepperOnDeviceKokkos.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::fd_patch {
namespace {
namespace patch_ns = evolution::dg::subcell::fd_patch;
using ValenciaDivClean::BoundaryCorrections::Hll;
using ValenciaDivClean::BoundaryCorrections::Rusanov;

Kokkos::View<const double*> flat(const auto& batch) {
  return Kokkos::View<const double*>(batch.view().data(), batch.view().span());
}

// Copies the variables `Tags` of the batch `source` into `destination`, which
// may hold other variables
template <typename Tags, typename DestinationTags, typename SourceTags>
void copy_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<DestinationTags>*> destination,
    const BatchedVariables<SourceTags>& source) {
  tmpl::for_each<Tags>([&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    const auto& from = get<::Tags::BatchView<Tag>>(source);
    const auto& to = get<::Tags::BatchView<Tag>>(*destination);
    for (size_t component = 0; component < from.size(); ++component) {
      Kokkos::deep_copy(exec, to[component], from[component]);
    }
  });
}
}  // namespace

template <typename OrderedListOfPrimitiveRecoverySchemes>
Patch<OrderedListOfPrimitiveRecoverySchemes>::Patch(
    const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
    const evolution::patches::PatchTable<3>& patch_table, const size_t patch,
    const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const grmhd::GhValenciaDivClean::fd::Reconstructor<System>& reconstructor,
    const evolution::BoundaryCorrection& boundary_correction,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
    const VariableFixing::LimitLorentzFactor& limit_lorentz_factor,
    const ValenciaDivClean::FixConservatives& fix_conservatives,
    const ValenciaDivClean::PrimitiveFromConservativeOptions&
        primitive_from_conservative_options,
    const double constraint_damping_parameter,
    const std::array<const ConstraintDamping::DampingFunction<3, Frame::Grid>*,
                     3>& damping_functions,
    std::optional<double> spacetime_dissipation)
    : fixing_(exec, equation_of_state, fix_to_atmosphere, fix_conservatives,
              primitive_from_conservative_options),
      limit_lorentz_factor_(limit_lorentz_factor),
      constraint_damping_parameter_(constraint_damping_parameter),
      fix_reconstructed_states_(
          subcell::fix_reconstructed_states_to_atmosphere(reconstructor)) {
  // Errors if the reconstructor isn't supported
  const ValenciaDivClean::fd::ReconstructorOnDevice hydro_reconstructor =
      subcell::hydro_reconstructor_on_device(reconstructor);
  if (hydro_reconstructor.ghost_zone_size() !=
      reconstructor.ghost_zone_size()) {
    ERROR("The ghost zones of the reconstruction on the device have width "
          << hydro_reconstructor.ghost_zone_size() << " but those of the "
          << pretty_type::get_runtime_type_name(reconstructor)
          << " reconstruction have width " << reconstructor.ghost_zone_size());
  }
  const size_t spacetime_reconstruction_degree =
      subcell::spacetime_reconstruction_degree(reconstructor);
  const Mesh<3>& dg_mesh = patch_table.dg_mesh(patch);
  const size_t number_of_elements = patch_table.element_ids(patch).size();

  // Errors if a damping function isn't a sum of Gaussians
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(damping_functions_, i) = gsl::at(damping_functions, i)->get_clone();
    (void)gsl::at(damping_functions_, i)
        ->as_gaussian_sum(time, functions_of_time);
  }

  // Boundary correction and time derivative. The FD elements only use the
  // GRMHD part of the product of the GH and GRMHD corrections; DG neighbors
  // send the packaged data of both, the GH part first.
  size_t number_of_packaged_variables = 0;
  size_t number_of_dg_packaged_variables = 0;
  size_t dg_packaged_variables_offset = 0;
  bool is_supported = false;
  call_with_dynamic_type<void,
                         BoundaryCorrections::standard_boundary_corrections>(
      &boundary_correction, [&](const auto* const product_correction) {
        using ProductCorrection = std::decay_t<decltype(*product_correction)>;
        using GhCorrection =
            std::decay_t<decltype(product_correction->gh_correction())>;
        using GrmhdCorrection =
            std::decay_t<decltype(product_correction->valencia_correction())>;
        if constexpr (std::is_same_v<GrmhdCorrection, Rusanov> or
                      std::is_same_v<GrmhdCorrection, Hll>) {
          is_supported = true;
          boundary_correction_ = product_correction->valencia_correction();
          time_derivative_ = subcell::TimeDerivativeOnDevice<GrmhdCorrection>{
              exec,
              number_of_elements,
              dg_mesh,
              hydro_reconstructor,
              spacetime_reconstruction_degree,
              std::move(spacetime_dissipation)};
          number_of_packaged_variables =
              Variables<typename GrmhdCorrection::dg_package_field_tags>::
                  number_of_independent_components;
          number_of_dg_packaged_variables =
              Variables<typename ProductCorrection::dg_package_field_tags>::
                  number_of_independent_components;
          dg_packaged_variables_offset =
              Variables<typename GhCorrection::dg_package_field_tags>::
                  number_of_independent_components;
        }
      });
  if (not is_supported) {
    ERROR(
        "FD patches only support the Rusanov and Hll boundary corrections of "
        "the GRMHD variables, but got "
        << pretty_type::get_runtime_type_name(boundary_correction));
  }

  // The ghost variables are those of `PrimitiveGhostVariables`: the hydro
  // variables to reconstruct, which are computed into `ghost_variables_`, and
  // the GH variables, which are read from the evolved variables (where they
  // come first)
  constexpr size_t number_of_hydro_ghost_variables =
      Variables<hydro_ghost_tags>::number_of_independent_components;
  constexpr size_t number_of_gh_variables =
      Variables<subcell::gh_tags_on_device>::number_of_independent_components;
  static_assert(std::is_same_v<
                subcell::PrimitiveGhostVariables::ghost_variables_tag_list,
                tmpl::append<hydro_ghost_tags, subcell::gh_tags_on_device>>);
  static_assert(std::is_same_v<tmpl::front<subcell::evolved_tags_on_device>,
                               tmpl::front<subcell::gh_tags_on_device>>);
  // The cell geometry includes the inverse Jacobian of the FD derivatives of
  // the GH variables and the grid coordinates of the damping functions
  this->set_patch_elements(
      patch_ns::FdPatchElements{
          exec,
          domain,
          patch_table,
          patch,
          initial_refinement_levels,
          time,
          functions_of_time,
          hydro_reconstructor.ghost_zone_size(),
          {patch_ns::GhostVariablesPart{
               .number_of_variables = number_of_hydro_ghost_variables,
               .variables_per_element = number_of_hydro_ghost_variables},
           patch_ns::GhostVariablesPart{
               .number_of_variables = number_of_gh_variables,
               .variables_per_element =
                   EvolvedVariables::number_of_independent_components}},
          number_of_packaged_variables,
          number_of_dg_packaged_variables,
          dg_packaged_variables_offset,
          true},
      number_of_rdmp_values);

  const size_t number_of_cells =
      patch_elements().subcell_mesh().number_of_grid_points();
  ghost_variables_ = BatchedVariables<hydro_ghost_tags>{exec, number_of_cells,
                                                        number_of_elements};
  fixing_metric_ = BatchedVariables<subcell::fixing_metric_tags>{
      exec, number_of_cells, number_of_elements};
  stepper_errors_.resize(number_of_elements);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
auto Patch<OrderedListOfPrimitiveRecoverySchemes>::initial_state(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::vector<Variables<initial_data_tags>>& initial_data)
    -> tuples::tagged_tuple_from_typelist<state_tags> {
  const size_t number_of_elements = patch_elements().number_of_elements();
  const size_t number_of_cells =
      patch_elements().subcell_mesh().number_of_grid_points();
  if (initial_data.size() != number_of_elements) {
    ERROR("Got the initial data of " << initial_data.size()
                                     << " elements, but the patch has "
                                     << number_of_elements);
  }
  BatchedVariables<initial_data_tags, Kokkos::HostSpace> host_batch{
      Kokkos::DefaultHostExecutionSpace{}, number_of_cells, number_of_elements};
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    Variables<initial_data_tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, slot);
    element.assign_subset(initial_data[slot]);
  }
  BatchedVariables<initial_data_tags> device_batch{exec, number_of_cells,
                                                   number_of_elements};
  deep_copy(exec, make_not_null(&device_batch), host_batch);

  tuples::tagged_tuple_from_typelist<state_tags> result{
      EvolvedVariables{exec, number_of_cells, number_of_elements},
      PrimitiveVariables{exec, number_of_cells, number_of_elements}};
  auto& evolved_variables = get<evolved_variables_tag>(result);
  auto& primitives = get<primitive_variables_tag>(result);
  copy_variables<subcell::gh_tags_on_device>(
      exec, make_not_null(&evolved_variables), device_batch);
  copy_variables<hydro::grmhd_tags<DataVector>>(
      exec, make_not_null(&primitives), device_batch);
  update_fixing_metric(exec, evolved_variables);
  // As `initialize_initial_data_dependent_quantities_actions` of the
  // executable: before and after the initial TCI
  for (size_t i = 0; i < 2; ++i) {
    fixing_.fix_to_atmosphere_now(exec, make_not_null(&primitives),
                                  fixing_metric_, patch_elements());
    VariableFixing::limit_lorentz_factor_on_device(
        exec, make_not_null(&primitives), limit_lorentz_factor_);
    fixing_.compute_conservatives(exec, make_not_null(&evolved_variables),
                                  primitives, fixing_metric_);
  }
  // The host batch must stay valid until the copy is done
  exec.fence("grmhd::GhValenciaDivClean::fd_patch::Patch::initial_state");
  return result;
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
auto Patch<OrderedListOfPrimitiveRecoverySchemes>::element_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& primitives) -> std::vector<ElementVariables> {
  if (dt_variables_.number_of_elements() == 0) {
    // Before the first time derivative, uninitialized as on the host
    dt_variables_ = BatchedVariables<subcell::dt_tags_on_device>{
        exec, patch_elements().subcell_mesh().number_of_grid_points(),
        patch_elements().number_of_elements()};
  }
  return this->copy_element_variables(exec, evolved_variables, primitives,
                                      dt_variables_);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
std::vector<patch_ns::OutgoingBoundaryData<3>>
Patch<OrderedListOfPrimitiveRecoverySchemes>::prepare_send(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& primitives, const TimeStepId& next_time_step_id,
    const size_t integration_order) {
  const Kokkos::Profiling::ScopedRegion region{
      "GhValenciaDivClean::fd_patch::Patch::prepare_send"};
  subcell::primitive_ghost_variables_on_device(
      exec, make_not_null(&ghost_variables_), primitives);
  if (patch_elements().has_external_faces()) {
    patch_elements().enqueue_boundary_data(
        exec, {flat(ghost_variables_), flat(evolved_variables)});
    ValenciaDivClean::subcell::enqueue_rdmp_tci_data_on_device(
        exec,
        get<::Tags::BatchView<ValenciaDivClean::Tags::TildeD>>(
            evolved_variables),
        get<::Tags::BatchView<ValenciaDivClean::Tags::TildeYe>>(
            evolved_variables),
        get<::Tags::BatchView<ValenciaDivClean::Tags::TildeTau>>(
            evolved_variables),
        get<::Tags::BatchView<ValenciaDivClean::Tags::TildeB<>>>(
            evolved_variables),
        make_not_null(&rdmp_tci_data_buffers_));
  }
  // The only wait for the device in a substep: the failures of the last
  // substep are reported before its state is sent
  check_last_step(exec);
  // The ghost zones between the elements of the patch are filled after the
  // wait, so that the data for the neighbors outside the patch is sent without
  // waiting for them
  patch_elements().fill_ghost_zones(
      exec, {flat(ghost_variables_), flat(evolved_variables)});
  if (not patch_elements().has_external_faces()) {
    return {};
  }
  return patch_elements().boundary_data_to_send(
      ValenciaDivClean::subcell::rdmp_tci_data_from_buffers(
          rdmp_tci_data_buffers_),
      next_time_step_id, tci_status_sent, integration_order);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
std::vector<patch_ns::OutgoingInitialRdmpData<3>>
Patch<OrderedListOfPrimitiveRecoverySchemes>::initial_rdmp_data(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& /*primitives*/) const {
  // As `ValenciaDivClean::subcell::SetInitialRdmpData` on the subcell grid
  return patch_elements().initial_rdmp_data(
      ValenciaDivClean::subcell::rdmp_tci_data_on_device(
          exec,
          get<::Tags::BatchView<ValenciaDivClean::Tags::TildeD>>(
              evolved_variables),
          get<::Tags::BatchView<ValenciaDivClean::Tags::TildeYe>>(
              evolved_variables),
          get<::Tags::BatchView<ValenciaDivClean::Tags::TildeTau>>(
              evolved_variables),
          get<::Tags::BatchView<ValenciaDivClean::Tags::TildeB<>>>(
              evolved_variables)));
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
bool Patch<OrderedListOfPrimitiveRecoverySchemes>::dense_output(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& /*evolved_variables*/,
    const PrimitiveVariables& primitives,
    const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
    const TimeStepper& time_stepper, const double time) {
  const Kokkos::Profiling::ScopedRegion region{
      "GhValenciaDivClean::fd_patch::Patch::dense_output"};
  if (dense_evolved_variables_.number_of_elements() == 0) {
    const size_t number_of_elements = patch_elements().number_of_elements();
    const size_t number_of_cells =
        patch_elements().subcell_mesh().number_of_grid_points();
    dense_evolved_variables_ =
        EvolvedVariables{exec, number_of_cells, number_of_elements};
    dense_primitives_ =
        PrimitiveVariables{exec, number_of_cells, number_of_elements};
    dense_fixing_metric_ = BatchedVariables<subcell::fixing_metric_tags>{
        exec, number_of_cells, number_of_elements};
    dense_statuses_ = typename ValenciaDivClean::fd_patch::FixingAndRecovery<
        OrderedListOfPrimitiveRecoverySchemes>::Statuses{exec};
  }
  if (not TimeSteppers::dense_update_u_on_device(
          exec, make_not_null(&dense_evolved_variables_), history, time_stepper,
          time)) {
    return false;
  }
  deep_copy(exec, make_not_null(&dense_primitives_), primitives);
  dense_statuses_.reset(exec);
  // With the metric of the dense spacetime metric, as
  // `FixConservativesAndComputePrims`
  subcell::fixing_metric_on_device(
      exec, make_not_null(&dense_fixing_metric_),
      get<::Tags::BatchView<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          dense_evolved_variables_));
  fixing_.fix_conservatives_and_recover_primitives(
      exec, make_not_null(&dense_evolved_variables_),
      make_not_null(&dense_primitives_), dense_fixing_metric_,
      make_not_null(&dense_statuses_));
  dense_statuses_.copy_to_host(exec);
  return true;
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
auto Patch<OrderedListOfPrimitiveRecoverySchemes>::dense_element_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& /*evolved_variables*/,
    const PrimitiveVariables& /*primitives*/, const double time)
    -> std::vector<ElementVariables> {
  auto result = this->copy_element_variables(exec, dense_evolved_variables_,
                                             dense_primitives_, dt_variables_);
  dense_statuses_.check(
      exec, patch_elements(), number_of_face_points(),
      MakeString{} << "the dense output at time " << time << " of ");
  return result;
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
auto Patch<OrderedListOfPrimitiveRecoverySchemes>::observable_sources(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& primitives, const bool dense_output,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time)
    -> device_observable_sources {
  if (dense_output) {
    dense_statuses_.check(
        exec, patch_elements(), number_of_face_points(),
        MakeString{} << "the dense output at time " << time << " of ");
  }
  const Mesh<3>& mesh = patch_elements().subcell_mesh();
  // The same weight at all cell centers of the FD mesh
  double point_weight = 1.0;
  for (size_t d = 0; d < 3; ++d) {
    const DataVector& weights =
        Spectral::quadrature_weights(mesh.slice_through(d));
    if (not alg::all_of(
            weights, [&weights](const double w) { return w == weights[0]; })) {
      ERROR("The quadrature weights of the FD mesh "
            << mesh << " must all be the same in dimension " << d
            << ", but are " << weights);
    }
    point_weight *= weights[0];
  }
  // The damping functions at the time of the observation
  std::array<ConstraintDamping::GaussianSum<3>, 3> gaussian_sums{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(gaussian_sums, i) = gsl::at(damping_functions_, i)
                                    ->as_gaussian_sum(time, functions_of_time);
  }
  observation_gaussian_sums_.set(exec, gaussian_sums);
  const auto& grid_geometry = patch_elements().cell_logical_to_grid_geometry();
  return device_observable_sources{
      .evolved_variables =
          dense_output ? &dense_evolved_variables_ : &evolved_variables,
      .primitives = dense_output ? &dense_primitives_ : &primitives,
      .fixing_metric = dense_output ? &dense_fixing_metric_ : &fixing_metric_,
      .cell_geometry = &patch_elements().observation_cell_geometry(
          exec, time, functions_of_time),
      .grid_coordinates = ObservablesKokkos_detail::components<
          ::domain::Tags::Coordinates<3, Frame::Grid>>(
          grid_geometry, {0, grid_geometry.number_of_elements()}),
      .damping_functions = observation_gaussian_sums_.device_array(),
      .mesh = mesh,
      .weight = point_weight,
      .buffers = &device_observables_scratch_};
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::compute_time_derivative(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const gsl::not_null<PrimitiveVariables*> primitives,
    const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
    const TimeStepId& time_step_id,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  const double time = time_step_id.substep_time();
  patch_elements().update_geometry(exec, time, functions_of_time);
  // The failures of this substep are recorded on the device and reported by
  // `check_last_step()`, so nothing here waits for the device
  fixing_.reset_statuses(exec);

  // The damping functions at the time of the substep
  std::array<ConstraintDamping::GaussianSum<3>, 3> gaussian_sums{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(gaussian_sums, i) = gsl::at(damping_functions_, i)
                                    ->as_gaussian_sum(time, functions_of_time);
  }
  gaussian_sums_.set(exec, gaussian_sums);
  // The array of the three sums, whose address doesn't change between substeps
  const Kokkos::View<const ConstraintDamping::GaussianSum<3>*> damping =
      gaussian_sums_.device_array();

  // Time derivative, computed in place in the history
  std::optional<Kokkos::Profiling::ScopedRegion> region{};
  region.emplace("GhValenciaDivClean::fd_patch::Patch::time_derivative");
  const auto& elements = this->patch_elements();
  const auto record = history->insert(exec, time_step_id, true);
  dt_variables_ = TimeSteppers::derivative_batch<subcell::dt_tags_on_device>(
      record, elements.subcell_mesh().number_of_grid_points());
  using MeshVelocity = gh::TimeDerivativeOnDevice::MeshVelocity;
  std::optional<typename ::Tags::BatchView<MeshVelocity>::type> mesh_velocity{};
  if (elements.mesh_is_moving()) {
    mesh_velocity = get<::Tags::BatchView<
        evolution::dg::subcell::fd_patch::Tags::CellMeshVelocity>>(
        elements.cell_mesh_velocity());
  }
  std::visit(
      [&]<typename Correction>(
          subcell::TimeDerivativeOnDevice<Correction>& time_derivative) {
        time_derivative.apply(
            exec, make_not_null(&dt_variables_), evolved_variables,
            fixing_.face_status(), elements.padded_ghost_variables(),
            *primitives,
            get<::Tags::BatchView<domain::Tags::InverseJacobian<
                3, Frame::ElementLogical, Frame::Inertial>>>(
                elements.cell_geometry()),
            elements.cell_grid_coordinates(), mesh_velocity,
            elements.face_geometry(), elements.cell_det_inverse_jacobian(),
            // Of each cell, as the host projects it from the DG grid, which
            // also supports maps that aren't affine
            ValenciaDivClean::subcell::DivergenceOfMeshVelocity{
                get<::Tags::BatchView<evolution::dg::subcell::fd_patch::Tags::
                                          CellDivMeshVelocity>>(
                    elements.cell_mesh_velocity())},
            damping, std::get<Correction>(boundary_correction_),
            fixing_.equation_of_state(),
            fix_reconstructed_states_ ? &fixing_.fix_to_atmosphere() : nullptr,
            constraint_damping_parameter_, elements.packaged_data_from_dg(),
            elements.received_mortar_data());
      },
      time_derivative_);

  // After the time derivative, which filters the evolved variables
  TimeSteppers::record_value_on_device(exec, record, *evolved_variables);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::update_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const gsl::not_null<PrimitiveVariables*> primitives,
    const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
    const TimeStepper& time_stepper, const TimeStepId& time_step_id,
    const TimeStepId& next_time_step_id, const TimeDelta& time_step) {
  std::optional<Kokkos::Profiling::ScopedRegion> region{};
  region.emplace("GhValenciaDivClean::fd_patch::Patch::update_u");
  TimeSteppers::update_u_on_device(
      exec, evolved_variables, make_not_null(&stepper_errors_), history,
      time_stepper, time_step_id, next_time_step_id, time_step, false,
      StepperErrorTolerances{});
  history->clean(time_stepper);
  region.reset();

  // With the metric of the updated spacetime, as
  // `FixConservativesAndComputePrims` and the compute tags of an element
  update_fixing_metric(exec, *evolved_variables);
  fixing_.fix_conservatives_and_recover_primitives(exec, evolved_variables,
                                                   primitives, fixing_metric_);
  fix_primitives_and_update_conservatives(exec, evolved_variables, primitives);
  fixing_.copy_statuses_to_host(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::fix_variables(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const gsl::not_null<PrimitiveVariables*> primitives) {
  // The metric is recomputed, since the evolved variables may have been
  // restored (in the self start) or the patch rebuilt (after a restart)
  update_fixing_metric(exec, *evolved_variables);
  fix_primitives_and_update_conservatives(exec, evolved_variables, primitives);
  fixing_.copy_statuses_to_host(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::
    fix_primitives_and_update_conservatives(
        const Kokkos::DefaultExecutionSpace& exec,
        const gsl::not_null<EvolvedVariables*> evolved_variables,
        const gsl::not_null<PrimitiveVariables*> primitives) {
  fixing_.fix_to_atmosphere(exec, primitives, fixing_metric_);
  {
    const Kokkos::Profiling::ScopedRegion region{
        "GhValenciaDivClean::fd_patch::Patch::limit_lorentz_factor"};
    VariableFixing::limit_lorentz_factor_on_device(exec, primitives,
                                                   limit_lorentz_factor_);
  }
  fixing_.compute_conservatives(exec, evolved_variables, *primitives,
                                fixing_metric_);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::update_fixing_metric(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables) {
  subcell::fixing_metric_on_device(
      exec, make_not_null(&fixing_metric_),
      get<::Tags::BatchView<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          evolved_variables));
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::check_last_step(
    const Kokkos::DefaultExecutionSpace& exec) const {
  fixing_.check(exec, patch_elements(), number_of_face_points());
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
size_t Patch<OrderedListOfPrimitiveRecoverySchemes>::number_of_face_points()
    const {
  return std::visit(
      [](const auto& time_derivative) {
        return time_derivative.number_of_face_points();
      },
      time_derivative_);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::save_initial_values(
    const Kokkos::DefaultExecutionSpace& exec,
    const EvolvedVariables& evolved_variables,
    const PrimitiveVariables& primitives) {
  initial_values_.save(exec, evolved_variables, primitives);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::restore_initial_values(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const gsl::not_null<PrimitiveVariables*> primitives) const {
  initial_values_.restore(exec, evolved_variables, primitives);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::discard_initial_values() {
  initial_values_.discard();
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void Patch<OrderedListOfPrimitiveRecoverySchemes>::pup(PUP::er& /*p*/) {}

template class Patch<
    tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
               ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
               ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>>;
}  // namespace grmhd::GhValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
