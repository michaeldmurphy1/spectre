// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/FdPatch/CheckOptions.hpp"
#include "Evolution/DgSubcell/FdPatch/PatchTable.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Patches/ActionsKokkos.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/PatchKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/LimitLorentzFactor.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/DeviceHistoryEvolvedVariablesKokkos.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::GhValenciaDivClean::fd_patch {
namespace Tags {
/// The GhValenciaDivClean patch of an `evolution::patches::PatchArray` member
template <typename OrderedListOfPrimitiveRecoverySchemes>
struct Patch : db::SimpleTag {
  using type = fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes>;
};
}  // namespace Tags

namespace detail {
// The patch of the `evolution::patches::PatchArray` member `array_index`, with
// the geometry at `::Tags::Time`
template <typename OrderedListOfPrimitiveRecoverySchemes, typename DbTagsList,
          typename Metavariables>
fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes> make_patch(
    const db::DataBox<DbTagsList>& box,
    const Parallel::GlobalCache<Metavariables>& cache, const int array_index) {
  evolution::dg::subcell::fd_patch::check_options(box, cache);
  // The device time derivative computes the GH time derivative in the
  // harmonic gauge
  const auto& gauge_condition = db::get<gh::gauges::Tags::GaugeCondition>(box);
  if (dynamic_cast<const gh::gauges::Harmonic*>(&gauge_condition) == nullptr) {
    ERROR("FD patches only support the Harmonic gauge, but got "
          << pretty_type::get_runtime_type_name(gauge_condition));
  }
  const auto patch = static_cast<size_t>(array_index);
  return fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes>{
      Parallel::device_execution_space(),
      db::get<domain::Tags::Domain<3>>(box),
      db::get<evolution::dg::subcell::fd_patch::Tags::PatchTable<3>>(box),
      patch,
      db::get<domain::Tags::InitialRefinementLevels<3>>(box),
      db::get<::Tags::Time>(box),
      Parallel::get<domain::Tags::FunctionsOfTime>(cache),
      db::get<fd::Tags::Reconstructor<System>>(box),
      db::get<evolution::Tags::BoundaryCorrection>(box),
      db::get<hydro::Tags::GrmhdEquationOfState>(box),
      db::get<::Tags::VariableFixer<VariableFixing::FixToAtmosphere<3>>>(box),
      db::get<::Tags::VariableFixer<VariableFixing::LimitLorentzFactor>>(box),
      db::get<::Tags::VariableFixer<ValenciaDivClean::FixConservatives>>(box),
      db::get<ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>(box),
      db::get<ValenciaDivClean::Tags::ConstraintDampingParameter>(box),
      {&db::get<gh::Tags::DampingFunctionGamma0<3, Frame::Grid>>(box),
       &db::get<gh::Tags::DampingFunctionGamma1<3, Frame::Grid>>(box),
       &db::get<gh::Tags::DampingFunctionGamma2<3, Frame::Grid>>(box)},
      db::get<fd::Tags::FilterOptions>(box).spacetime_dissipation};
}

// The GH and primitive variables of the element in `slot` of the `patch`, as
// an FD element on the host sets them from analytic initial data
// (`Actions::SetInitialData`) and the constraints
// (`SetPiAndPhiFromConstraints`)
template <typename Metavariables,
          typename OrderedListOfPrimitiveRecoverySchemes, typename DbTagsList>
Variables<initial_data_tags> initial_data_of_element(
    const db::DataBox<DbTagsList>& box,
    const Parallel::GlobalCache<Metavariables>& cache,
    const fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes>& patch,
    const size_t slot) {
  const auto& patch_elements = patch.patch_elements();
  const Mesh<3>& subcell_mesh = patch_elements.subcell_mesh();
  const double time = db::get<::Tags::Time>(box);
  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);
  // The cell geometry as the subcell compute tags of an element give it
  const auto cell_geometry = patch_elements.host_inertial_cell_geometry(slot);
  const auto& inertial_coords =
      get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
          cell_geometry);
  const auto& inverse_jacobian = get<
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>(
      cell_geometry);

  using initial_data_classes =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               evolution::initial_data::InitialData>;
  using adm_and_hydro_tags = tmpl::append<
      tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>,
      hydro::grmhd_tags<DataVector>>;
  Variables<adm_and_hydro_tags> adm_and_hydro{
      subcell_mesh.number_of_grid_points()};
  call_with_dynamic_type<void, initial_data_classes>(
      &db::get<evolution::initial_data::Tags::InitialData>(box),
      [&](const auto* const initial_data) {
        using DataType = std::decay_t<decltype(*initial_data)>;
        if constexpr (is_analytic_data_v<DataType> or
                      is_analytic_solution_v<DataType>) {
          adm_and_hydro.assign_subset(evolution::Initialization::initial_data(
              *initial_data, inertial_coords, time, adm_and_hydro_tags{}));
        } else {
          ERROR(
              "FD patches only support analytic initial data and analytic "
              "solutions, but got "
              << pretty_type::get_runtime_type_name(*initial_data));
        }
      });

  Variables<initial_data_tags> result{subcell_mesh.number_of_grid_points()};
  result.assign_subset(adm_and_hydro);
  // As `Actions::SetInitialData` for analytic data
  gh::initial_gh_variables_from_adm<3>(
      make_not_null(&get<gr::Tags::SpacetimeMetric<DataVector, 3>>(result)),
      make_not_null(&get<gh::Tags::Pi<DataVector, 3>>(result)),
      make_not_null(&get<gh::Tags::Phi<DataVector, 3>>(result)),
      get<gr::Tags::SpatialMetric<DataVector, 3>>(adm_and_hydro),
      get<gr::Tags::Lapse<DataVector>>(adm_and_hydro),
      get<gr::Tags::Shift<DataVector, 3>>(adm_and_hydro),
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(adm_and_hydro),
      subcell_mesh, inverse_jacobian, inertial_coords);
  // As `SetPiAndPhiFromConstraints` on the subcell grid
  gh::gauges::SetPiAndPhiFromConstraints<
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
      3>::impl(make_not_null(&get<gh::Tags::Pi<DataVector, 3>>(result)),
               make_not_null(&get<gh::Tags::Phi<DataVector, 3>>(result)), time,
               subcell_mesh, patch_elements.element_maps()[slot],
               patch_elements.grid_to_inertial_map(slot), functions_of_time,
               logical_coordinates(subcell_mesh),
               get<gr::Tags::SpacetimeMetric<DataVector, 3>>(result),
               db::get<gh::gauges::Tags::GaugeCondition>(box),
               Parallel::get<gh::Tags::SetPiAndPhiFromConstraints>(cache));
  return result;
}
}  // namespace detail

namespace Actions {
/*!
 * \brief Set up the GhValenciaDivClean patch of an
 * `evolution::patches::PatchArray` member and the initial state of its elements
 * (see `fd_patch::Patch`).
 *
 * The initial data must be analytic data or an analytic solution, evaluated
 * at `::Tags::Time` on the host as `EvolveGhValenciaDivClean` sets it on FD
 * elements (see `fd_patch::Patch::initial_state`). The time stepper history of
 * the elements is `::Tags::DeviceHistoryEvolvedVariables`. The time stepping
 * tags must be initialized before, e.g. by `Initialization::TimeStepping`.
 * The step size isn't controlled by error estimates, so
 * `::Tags::StepperErrorEstimatesEnabled` is `false`.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
struct InitializePatch {
  using patch_tag = Tags::Patch<OrderedListOfPrimitiveRecoverySchemes>;
  using history_tag =
      ::Tags::DeviceHistoryEvolvedVariables<typename System::variables_tag>;
  using state_tags = typename patch_tag::type::state_tags;
  using simple_tags = tmpl::append<
      tmpl::list<patch_tag>, state_tags,
      tmpl::list<history_tag, ::Tags::StepperErrorEstimatesEnabled>>;
  // The dense triggers, to check that there are none (see `check_options`)
  using simple_tags_from_options =
      tmpl::list<domain::Tags::InitialRefinementLevels<3>, ::Tags::Time,
                 ::Tags::EventsAndDenseTriggers>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>, fd::Tags::Reconstructor<System>,
                 fd::Tags::FilterOptions, evolution::Tags::BoundaryCorrection,
                 hydro::Tags::GrmhdEquationOfState,
                 ::Tags::VariableFixer<VariableFixing::FixToAtmosphere<3>>,
                 ::Tags::VariableFixer<VariableFixing::LimitLorentzFactor>,
                 ::Tags::VariableFixer<ValenciaDivClean::FixConservatives>,
                 ValenciaDivClean::Tags::PrimitiveFromConservativeOptions,
                 ValenciaDivClean::Tags::ConstraintDampingParameter,
                 gh::Tags::DampingFunctionGamma0<3, Frame::Grid>,
                 gh::Tags::DampingFunctionGamma1<3, Frame::Grid>,
                 gh::Tags::DampingFunctionGamma2<3, Frame::Grid>,
                 gh::gauges::Tags::GaugeCondition,
                 evolution::initial_data::Tags::InitialData,
                 evolution::dg::subcell::Tags::SubcellOptions<3>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    const auto& exec = Parallel::device_execution_space();
    auto new_patch = detail::make_patch<OrderedListOfPrimitiveRecoverySchemes>(
        box, cache, array_index);
    // The inertial geometry at the cell centers on the host
    new_patch.prepare_element_boxes(
        exec, db::get<::Tags::Time>(box),
        Parallel::get<domain::Tags::FunctionsOfTime>(cache));
    std::vector<Variables<initial_data_tags>> initial_data{};
    for (size_t slot = 0;
         slot < new_patch.patch_elements().number_of_elements(); ++slot) {
      initial_data.push_back(detail::initial_data_of_element<Metavariables>(
          box, cache, new_patch, slot));
    }
    auto state = new_patch.initial_state(exec, initial_data);
    auto history = new_patch.initial_history(
        db::get<::Tags::TimeStepper<TimeStepper>>(box));
    db::mutate<patch_tag, history_tag, ::Tags::StepperErrorEstimatesEnabled>(
        [&new_patch, &history](const auto patch_ptr, const auto history_ptr,
                               const auto error_estimates_enabled) {
          *patch_ptr = std::move(new_patch);
          *history_ptr = std::move(history);
          // Patches don't support step size control by error estimates
          *error_estimates_enabled = false;
        },
        make_not_null(&box));
    // One tag at a time: nvcc's front end mismatches a parameter pack in the
    // middle of the parameters of a single `db::mutate` lambda
    tmpl::for_each<state_tags>(
        [&box, &state]<typename StateTag>(tmpl::type_<StateTag> /*meta*/) {
          db::mutate<StateTag>(
              [&state](const auto state_ptr) {
                *state_ptr = std::move(get<StateTag>(state));
              },
              make_not_null(&box));
        });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Rebuild the GhValenciaDivClean patch of an
 * `evolution::patches::PatchArray` member after a restart from a checkpoint.
 *
 * \details Checkpoints hold the state of the elements of the patch, which is
 * in the DataBox (see `fd_patch::Patch`), but not the patch itself, which only
 * holds operations and scratch data. This builds the patch again from the
 * options, as `InitializePatch` does, with the geometry at `::Tags::Time`.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
struct RebuildPatch {
  using patch_tag = Tags::Patch<OrderedListOfPrimitiveRecoverySchemes>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    auto new_patch = detail::make_patch<OrderedListOfPrimitiveRecoverySchemes>(
        box, cache, array_index);
    db::mutate<patch_tag>(
        [&new_patch](const auto patch_ptr) {
          *patch_ptr = std::move(new_patch);
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Fix the primitive variables of the elements of the patch to the
 * atmosphere, limit their Lorentz factor and recompute the conserved
 * variables, as `EvolveGhValenciaDivClean` does at the start of each step of
 * the evolution (see `fd_patch::Patch::fix_variables`).
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
struct FixVariables {
  using patch_tag = Tags::Patch<OrderedListOfPrimitiveRecoverySchemes>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    evolution::patches::Actions::detail::mutate_with_state<patch_tag>(
        make_not_null(&box), [](const auto patch, const auto... state) {
          patch->fix_variables(Parallel::device_execution_space(), state...);
        });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace grmhd::GhValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
