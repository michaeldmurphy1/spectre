// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/FdPatch/PatchTable.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/PatchKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/LtsMode.hpp"
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
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd_patch {
namespace Tags {
/// The ValenciaDivClean patch of an `evolution::patches::PatchArray` member
template <typename OrderedListOfPrimitiveRecoverySchemes>
struct Patch : db::SimpleTag {
  using type = fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes>;
};
}  // namespace Tags

namespace detail {
// The patch of the `evolution::patches::PatchArray` member `array_index`, with
// the geometry at
// `::Tags::Time`
template <typename OrderedListOfPrimitiveRecoverySchemes, typename DbTagsList,
          typename Metavariables>
fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes> make_patch(
    const db::DataBox<DbTagsList>& box,
    const Parallel::GlobalCache<Metavariables>& cache, const int array_index) {
  // Patches assume that the elements that start on the subcell grid stay
  // there and the others stay on the DG grid (see
  // `evolution::dg::subcell::fd_patch::patch_table()`)
  const auto& subcell_options =
      db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box);
  if (not subcell_options.always_use_subcells()) {
    ERROR("FD patches need AlwaysUseSubcells.");
  }
  // The device time derivative only computes second-order FD derivatives of
  // the fluxes (no high-order flux corrections)
  if (subcell_options.finite_difference_derivative_order() !=
      ::fd::DerivativeOrder::Two) {
    ERROR("FD patches only support FiniteDifferenceDerivativeOrder: 2, but got "
          << subcell_options.finite_difference_derivative_order());
  }
  // Patches take global time steps: with local time stepping the DG neighbors
  // would send data at times the patches never reach
  if (db::get<::Tags::LtsMode>(box) != LtsMode::Off) {
    ERROR("FD patches don't support local time stepping, but LtsMode is "
          << db::get<::Tags::LtsMode>(box));
  }
  // Patches only run events and triggers at steps and slabs (see
  // `evolution::patches::Actions::RunEventsAndTriggers`), so the DG elements
  // would wait forever for the patches in the others
  std::vector<std::string> unsupported_events{};
  const auto add_event = [&unsupported_events](const Event& event) {
    unsupported_events.push_back(pretty_type::get_runtime_type_name(event));
  };
  db::get<::Tags::EventsAndDenseTriggers>(box).for_each_event(add_event);
  Parallel::get<
      ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtCheckpoints>>(cache)
      .for_each_event(add_event);
  for (const auto& event : Parallel::get<::Tags::EventsRunAtCleanup>(cache)) {
    add_event(*event);
  }
  if (not unsupported_events.empty()) {
    ERROR(
        "FD patches only run events in EventsAndTriggersAtSteps and "
        "EventsAndTriggersAtSlabs, but there are events in "
        "EventsAndDenseTriggers, EventsAndTriggersAtCheckpoints or "
        "EventsRunAtCleanup: "
        << unsupported_events);
  }
  const auto patch = static_cast<size_t>(array_index);
  const Domain<3>& domain = db::get<domain::Tags::Domain<3>>(box);
  const auto& patch_table =
      db::get<evolution::dg::subcell::fd_patch::Tags::PatchTable<3>>(box);
  const Block<3>& block = domain.blocks()[patch_table.block_id(patch)];
  const auto grid_to_inertial_map =
      block.is_time_dependent()
          ? block.moving_mesh_grid_to_inertial_map().get_clone()
          : domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                domain::CoordinateMaps::Identity<3>{});
  const double time = db::get<::Tags::Time>(box);

  const auto& exec = Parallel::device_execution_space();
  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);
  return fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes>{
      exec,
      domain,
      patch_table,
      patch,
      db::get<domain::Tags::InitialRefinementLevels<3>>(box),
      *grid_to_inertial_map,
      time,
      functions_of_time,
      db::get<fd::Tags::Reconstructor>(box),
      db::get<evolution::Tags::BoundaryCorrection>(box),
      db::get<hydro::Tags::GrmhdEquationOfState>(box),
      db::get<::Tags::VariableFixer<VariableFixing::FixToAtmosphere<3>>>(box),
      db::get<::Tags::VariableFixer<FixConservatives>>(box),
      db::get<ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>(box),
      db::get<ValenciaDivClean::Tags::ConstraintDampingParameter>(box)};
}
}  // namespace detail

namespace Actions {
/*!
 * \brief Set up the ValenciaDivClean patch of an
 * `evolution::patches::PatchArray` member and the initial state of its elements
 * (see `fd_patch::Patch`).
 *
 * The initial data must be analytic data or an analytic solution, evaluated
 * at `::Tags::Time`. The time stepper history of the elements is
 * `::Tags::DeviceHistoryEvolvedVariables`. The time stepping tags must be
 * initialized before, e.g. by `Initialization::TimeStepping`. The step size
 * isn't controlled by error estimates, so
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
  // The dense triggers, to check that there are none (see `make_patch`)
  using simple_tags_from_options =
      tmpl::list<domain::Tags::InitialRefinementLevels<3>, ::Tags::Time,
                 ::Tags::EventsAndDenseTriggers>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>, fd::Tags::Reconstructor,
                 evolution::Tags::BoundaryCorrection,
                 hydro::Tags::GrmhdEquationOfState,
                 ::Tags::VariableFixer<VariableFixing::FixToAtmosphere<3>>,
                 ::Tags::VariableFixer<FixConservatives>,
                 ValenciaDivClean::Tags::PrimitiveFromConservativeOptions,
                 ValenciaDivClean::Tags::ConstraintDampingParameter,
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
    const double time = db::get<::Tags::Time>(box);
    const auto& initial_data =
        db::get<evolution::initial_data::Tags::InitialData>(box);
    using initial_data_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;
    const InitialDataFunction evaluate_initial_data =
        [&initial_data,
         time](const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
          Variables<initial_data_tags> result{get<0>(coords).size()};
          call_with_dynamic_type<void, initial_data_classes>(
              &initial_data, [&result, &coords, time](const auto* const data) {
                using DataType = std::decay_t<decltype(*data)>;
                if constexpr (is_analytic_data_v<DataType> or
                              is_analytic_solution_v<DataType>) {
                  result.assign_subset(evolution::Initialization::initial_data(
                      *data, coords, time, initial_data_tags{}));
                } else {
                  ERROR(
                      "FD patches only support analytic initial data and "
                      "analytic solutions.");
                }
              });
          return result;
        };

    auto new_patch = detail::make_patch<OrderedListOfPrimitiveRecoverySchemes>(
        box, cache, array_index);
    auto state = new_patch.initial_state(
        exec, evaluate_initial_data, time,
        Parallel::get<domain::Tags::FunctionsOfTime>(cache));
    new_patch.set_face_spacetime_variables(
        exec,
        get<typename patch_tag::type::face_spacetime_variables_tag>(state));
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
 * \brief Rebuild the ValenciaDivClean patch of an
 * `evolution::patches::PatchArray` member after a restart from a checkpoint.
 *
 * \details Checkpoints hold the state of the elements of the patch, which is
 * in the DataBox (see `fd_patch::Patch`), but not the patch itself, which only
 * holds operations and scratch data. This builds the patch again from the
 * options, as `InitializePatch` does, with the geometry at `::Tags::Time`, and
 * passes it the restored spacetime on the faces.
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
    new_patch.set_face_spacetime_variables(
        Parallel::device_execution_space(),
        db::get<typename patch_tag::type::face_spacetime_variables_tag>(box));
    db::mutate<patch_tag>(
        [&new_patch](const auto patch_ptr) {
          *patch_ptr = std::move(new_patch);
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
