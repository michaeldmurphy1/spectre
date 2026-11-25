// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/Patches/PatchState.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace evolution::patches {
/// Where `for_each_element_box` gets the variables of the elements from
enum class ElementVariables {
  /// The right size but no meaningful values
  WithoutValues,
  /// The state of the elements at the current substep
  State,
  /// The last dense output of the patch (see `Patch::dense_output`)
  DenseOutput
};

/*!
 * \brief Calls `function(element_id, box)` with the DataBox of each element of
 * the patch in `PatchTag` (see `Patch::make_element_box`) at the time in the
 * DataBox of the patch.
 *
 * \details The `variables_source` of the elements is copied from the device,
 * once for all elements. `ElementVariables::WithoutValues` is enough for
 * events whose `needs_evolved_variables()` is false (e.g. `ObserveTimeStep`
 * only counts the grid points) and saves most of the cost of the DataBoxes.
 */
template <typename PatchTag, typename DbTagsList, typename Metavariables,
          typename Function>
void for_each_element_box(const gsl::not_null<db::DataBox<DbTagsList>*> box_ptr,
                          Parallel::GlobalCache<Metavariables>& cache,
                          const ElementVariables variables_source,
                          const Function& function) {
  constexpr size_t dim = Metavariables::volume_dim;
  const Kokkos::Profiling::ScopedRegion region{"patches::element_boxes"};
  const auto& box = *box_ptr;
  const auto& patch = db::get<PatchTag>(box);
  // Report failures of the last substep before anything sees its state
  patch.check_last_step(Parallel::device_execution_space());
  patch.prepare_element_boxes(
      Parallel::device_execution_space(), db::get<::Tags::Time>(box),
      Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
  const auto& domain = db::get<::domain::Tags::Domain<dim>>(box);
  std::vector<typename std::decay_t<decltype(patch)>::ElementVariables>
      variables{};
  if (variables_source == ElementVariables::WithoutValues) {
    variables = patch.element_variables_without_values();
  } else {
    const Kokkos::Profiling::ScopedRegion copy_region{
        "patches::element_boxes::copy_variables"};
    const double time = db::get<::Tags::Time>(box);
    // The patch copies the variables into its host memory
    variables = with_state<PatchTag>(box, [&](const auto&... state) {
      return db::mutate<PatchTag>(
          [&](const auto mutable_patch) {
            return variables_source == ElementVariables::State
                       ? mutable_patch->element_variables(
                             Parallel::device_execution_space(), state...)
                       : mutable_patch->dense_element_variables(
                             Parallel::device_execution_space(), state...,
                             time);
          },
          box_ptr);
    });
  }
  const auto& element_ids = patch.patch_elements().element_ids();
  for (size_t slot = 0; slot < element_ids.size(); ++slot) {
    std::optional<Kokkos::Profiling::ScopedRegion> make_box_region{
        "patches::element_boxes::make_box"};
    auto element_box = patch.make_element_box(
        cache, domain, slot, db::get<::Tags::Time>(box),
        db::get<::Tags::TimeStepId>(box), db::get<::Tags::TimeStep>(box),
        std::move(variables[slot]));
    make_box_region.reset();
    function(element_ids[slot], element_box);
  }
}
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
