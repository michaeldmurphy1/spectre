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
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepperOnDeviceKokkos.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
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

  using ghost_tags = subcell::PrimitiveGhostVariables::ghost_variables_tag_list;
  // The cell geometry includes the inverse Jacobian of the FD derivatives of
  // the GH variables and the grid coordinates of the damping functions
  this->set_patch_elements(
      patch_ns::FdPatchElements{
          exec, domain, patch_table, patch, initial_refinement_levels, time,
          functions_of_time, hydro_reconstructor.ghost_zone_size(),
          Variables<ghost_tags>::number_of_independent_components,
          number_of_packaged_variables, number_of_dg_packaged_variables,
          dg_packaged_variables_offset, true},
      number_of_rdmp_values);

  const size_t number_of_cells =
      patch_elements().subcell_mesh().number_of_grid_points();
  ghost_variables_ =
      BatchedVariables<ghost_tags>{exec, number_of_cells, number_of_elements};
  dt_variables_ = BatchedVariables<subcell::dt_tags_on_device>{
      exec, number_of_cells, number_of_elements};
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
      exec, make_not_null(&ghost_variables_), primitives, evolved_variables);
  patch_elements().fill_ghost_zones(exec, flat(ghost_variables_));
  if (patch_elements().has_external_faces()) {
    patch_elements().enqueue_boundary_data(exec, flat(ghost_variables_));
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
void Patch<OrderedListOfPrimitiveRecoverySchemes>::take_step(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<EvolvedVariables*> evolved_variables,
    const gsl::not_null<PrimitiveVariables*> primitives,
    const gsl::not_null<TimeSteppers::DeviceHistory<>*> history,
    const TimeStepper& time_stepper, const TimeStepId& time_step_id,
    const TimeStepId& next_time_step_id, const TimeDelta& time_step,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  const double time = time_step_id.substep_time();
  patch_elements().update_geometry(exec, time, functions_of_time);
  if (not patch_elements().maps_are_affine()) {
    ERROR("The GhValenciaDivClean FD patch "
          << patch_elements().patch()
          << " only supports grid to inertial maps that are affine, but at "
             "time "
          << time
          << " they aren't: the GRMHD sources take the divergence of the mesh "
             "velocity of each element, not of each cell.");
  }
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

  // Time derivative
  std::optional<Kokkos::Profiling::ScopedRegion> region{};
  region.emplace("GhValenciaDivClean::fd_patch::Patch::time_derivative");
  const auto& elements = this->patch_elements();
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
            elements.div_mesh_velocity(), damping,
            std::get<Correction>(boundary_correction_),
            fixing_.equation_of_state(),
            fix_reconstructed_states_ ? &fixing_.fix_to_atmosphere() : nullptr,
            constraint_damping_parameter_, elements.packaged_data_from_dg(),
            elements.received_mortar_data());
      },
      time_derivative_);

  // Time stepping
  region.reset();
  region.emplace("GhValenciaDivClean::fd_patch::Patch::update_u");
  TimeSteppers::record_on_device(exec, history, time_step_id,
                                 *evolved_variables, dt_variables_);
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
