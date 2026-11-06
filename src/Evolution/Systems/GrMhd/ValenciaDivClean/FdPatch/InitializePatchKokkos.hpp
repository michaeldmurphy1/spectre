// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

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
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/Tags.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/PatchKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd_patch {
namespace Tags {
/// The ValenciaDivClean patch of an `FdPatchArray` member
template <typename OrderedListOfPrimitiveRecoverySchemes>
struct Patch : db::SimpleTag {
  using type = fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes>;
};
}  // namespace Tags

namespace Actions {
/*!
 * \brief Set up the ValenciaDivClean patch of an `FdPatchArray` member and its
 * initial data (see `fd_patch::Patch`).
 *
 * The initial data must be analytic data or an analytic solution, evaluated
 * at `::Tags::Time`. The time stepping tags must be initialized before, e.g.
 * by `Initialization::TimeStepping`. The step size isn't controlled by error
 * estimates, so `::Tags::StepperErrorEstimatesEnabled` is `false`.
 */
template <typename OrderedListOfPrimitiveRecoverySchemes>
struct InitializePatch {
  using patch_tag = Tags::Patch<OrderedListOfPrimitiveRecoverySchemes>;
  using simple_tags =
      tmpl::list<patch_tag, ::Tags::StepperErrorEstimatesEnabled>;
  using simple_tags_from_options =
      tmpl::list<domain::Tags::InitialRefinementLevels<3>, ::Tags::Time>;
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
    // Patches assume that the elements that start on the subcell grid stay
    // there and the others stay on the DG grid (see `FdPatches`)
    if (not db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box)
                .always_use_subcells()) {
      ERROR("FD patches need AlwaysUseSubcells.");
    }
    const auto patch = static_cast<size_t>(array_index);
    const Domain<3>& domain = db::get<domain::Tags::Domain<3>>(box);
    const auto& fd_patches =
        db::get<evolution::dg::subcell::fd_patch::Tags::FdPatches<3>>(box);
    const Block<3>& block = domain.blocks()[fd_patches.block_id(patch)];
    const auto grid_to_inertial_map =
        block.is_time_dependent()
            ? block.moving_mesh_grid_to_inertial_map().get_clone()
            : domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                  domain::CoordinateMaps::Identity<3>{});
    const double time = db::get<::Tags::Time>(box);

    using initial_data_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;
    const auto& initial_data =
        db::get<evolution::initial_data::Tags::InitialData>(box);
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

    fd_patch::Patch<OrderedListOfPrimitiveRecoverySchemes> new_patch{
        Parallel::device_execution_space(),
        domain,
        fd_patches,
        patch,
        db::get<domain::Tags::InitialRefinementLevels<3>>(box),
        evaluate_initial_data,
        *grid_to_inertial_map,
        time,
        Parallel::get<domain::Tags::FunctionsOfTime>(cache),
        db::get<fd::Tags::Reconstructor>(box),
        db::get<evolution::Tags::BoundaryCorrection>(box),
        db::get<hydro::Tags::GrmhdEquationOfState>(box),
        db::get<::Tags::VariableFixer<VariableFixing::FixToAtmosphere<3>>>(box),
        db::get<::Tags::VariableFixer<FixConservatives>>(box),
        db::get<ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>(box),
        db::get<ValenciaDivClean::Tags::ConstraintDampingParameter>(box),
        db::get<::Tags::TimeStepper<TimeStepper>>(box)};
    db::mutate<patch_tag, ::Tags::StepperErrorEstimatesEnabled>(
        [&new_patch](const auto patch_ptr, const auto error_estimates_enabled) {
          *patch_ptr = std::move(new_patch);
          // Patches don't support step size control by error estimates
          *error_estimates_enabled = false;
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
