// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/FdPatch/ElementBox.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatchElementsKokkos.hpp"
#include "Evolution/Patches/HostCopyKokkos.hpp"
#include "Evolution/Patches/InboxTags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
class TimeStepper;
/// \endcond

namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief The parts of the FD patches of the evolved `System` that don't depend
 * on its physics, from which the patch of the system (e.g.
 * `grmhd::ValenciaDivClean::fd_patch::Patch`) derives.
 *
 * \details This holds the elements of the patch (`patch_elements()`) and
 * provides the members of a patch that the generic actions call (see the
 * patch section of the GPU support guide) and that are the same for all
 * systems: the exchange of boundary data with the neighbors outside the patch
 * (`received_all()` and `receive()`), the time stepper history and the
 * DataBoxes of the elements for events (`make_element_box()`), which hold the
 * `ElementVariables` of an element and the `ElementBoxComputeTags` of the
 * system (see `make_element_box`). The derived patch adds the batched
 * physics.
 *
 * The elements of FD patches never switch to DG, so no TCI runs. They send
 * the TCI status `tci_status_sent` to their neighbors, which is what FD
 * elements send on the host with `AlwaysUseSubcells` (the only option FD
 * patches support), since their TCI decision is reset to 0 at initialization
 * and never recomputed.
 */
template <typename System, typename ElementVariablesTags,
          typename ElementBoxComputeTags = tmpl::list<>>
class PatchBase {
  static_assert(std::is_same_v<tmpl::front<ElementVariablesTags>,
                               typename System::variables_tag>,
                "The evolved variables come first in the variables of the "
                "element boxes.");

 public:
  /// The TCI status that the elements of the patch send to their neighbors
  static constexpr int tci_status_sent = 0;

  /// Host copies of the variables of an element (`ElementVariablesTags`, e.g.
  /// the evolved, primitive and spacetime variables), as the DataBox of an FD
  /// element of the system holds them (see `make_element_box`)
  using ElementVariables =
      tuples::tagged_tuple_from_typelist<ElementVariablesTags>;

  /// @{
  /// The elements of the patch, their geometry and the exchange of boundary
  /// data
  const FdPatchElements& patch_elements() const { return patch_elements_; }
  FdPatchElements& patch_elements() { return patch_elements_; }
  /// @}

  /// Host variables of each element with the right size but no meaningful
  /// values, for events that don't read them (see the `element_variables()`
  /// of the derived patch)
  std::vector<ElementVariables> element_variables_without_values() const {
    const size_t number_of_cells =
        patch_elements_.subcell_mesh().number_of_grid_points();
    std::vector<ElementVariables> result(patch_elements_.number_of_elements());
    for (auto& element_variables : result) {
      tmpl::for_each<ElementVariablesTags>(
          [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
            get<Tag>(element_variables).initialize(number_of_cells);
          });
    }
    return result;
  }

  /*!
   * \brief The DataBox of the element in `slot` for events, as that of an FD
   * element of the `DgElementArray` (see
   * `evolution::dg::subcell::fd_patch::make_element_box`), at the time
   * `time` of the substep `time_step_id` with the step `time_step`, holding
   * the element's `variables`.
   */
  template <typename Metavariables>
  auto make_element_box(Parallel::GlobalCache<Metavariables>& cache,
                        const Domain<3>& domain, const size_t slot,
                        const double time, const TimeStepId& time_step_id,
                        const TimeDelta& time_step,
                        ElementVariables variables) const {
    const Element<3>& element = patch_elements_.elements()[slot];
    return evolution::dg::subcell::fd_patch::make_element_box<
        Metavariables, ElementVariablesTags, ElementBoxComputeTags>(
        cache, element, patch_elements_.dg_mesh(),
        ElementMap<3, Frame::Grid>{element.id(),
                                   domain.blocks()[element.id().block_id()]},
        patch_elements_.grid_to_inertial_map(slot).get_clone(), time,
        time_step_id, time_step, tci_status_sent,
        patch_elements_.host_cell_geometry(slot),
        patch_elements_.host_inertial_cell_geometry(slot),
        std::move(variables));
  }

  /// Prepare what the DataBoxes of `make_element_box()` at `time` share, on
  /// `exec`, which is fenced (see
  /// `FdPatchElements::prepare_host_inertial_cell_geometry()`)
  void prepare_element_boxes(
      const Kokkos::DefaultExecutionSpace& exec, const double time,
      const ::domain::FunctionsOfTimeMap& functions_of_time) const {
    patch_elements_.prepare_host_inertial_cell_geometry(exec, time,
                                                        functions_of_time);
  }

  /// Whether `messages` hold all the boundary data that the patch receives in
  /// a substep (see `FdPatchElements::received_all()`)
  bool received_all(
      const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
          messages) const {
    return patch_elements_.received_all(messages);
  }

  /// Receive the boundary data of a substep (see `FdPatchElements::receive()`)
  void receive(
      const Kokkos::DefaultExecutionSpace& exec,
      const evolution::patches::Tags::BoundaryDataInbox<3>::messages_type&
          messages) {
    patch_elements_.receive(exec, messages, number_of_rdmp_values_);
  }

  /// The number of evolved values of each element, i.e., the number of
  /// cells times the number of independent components of the evolved
  /// variables
  size_t number_of_evolved_values_per_element() const {
    return patch_elements_.subcell_mesh().number_of_grid_points() *
           System::variables_tag::type::number_of_independent_components;
  }

  /// The time stepper history of the elements, empty and at the starting
  /// order of the `time_stepper`, as `Initialization::TimeStepperHistory`
  /// sets it up for an element
  TimeSteppers::DeviceHistory<> initial_history(
      const TimeStepper& time_stepper) const {
    return patch_elements_.initial_history(
        time_stepper, number_of_evolved_values_per_element());
  }

 protected:
  PatchBase() = default;
  PatchBase(const PatchBase&) = delete;
  PatchBase& operator=(const PatchBase&) = delete;
  PatchBase(PatchBase&&) = default;
  PatchBase& operator=(PatchBase&&) = default;
  ~PatchBase() = default;

  /// Set the elements of the patch, whose neighbors send
  /// `number_of_rdmp_values` RDMP values with their ghost data
  void set_patch_elements(FdPatchElements patch_elements,
                          const size_t number_of_rdmp_values) {
    patch_elements_ = std::move(patch_elements);
    number_of_rdmp_values_ = number_of_rdmp_values;
  }

  /*!
   * \brief The variables of each element on the subcell grid, copied from the
   * device on `exec`, which is fenced.
   *
   * \details The `batches` hold the `ElementVariablesTags` in their order, and
   * may hold more variables than these. The variables are copied into pinned
   * host memory that the patch allocates at the first call (see
   * `evolution::patches::HostCopy`), and the returned `Variables` refer to it
   * without owning the data. So they are valid until the next call (and while
   * the patch exists), which is enough for the element boxes of one
   * observation.
   */
  template <typename... BatchTags>
  std::vector<ElementVariables> copy_element_variables(
      const Kokkos::DefaultExecutionSpace& exec,
      const BatchedVariables<BatchTags>&... batches) {
    static_assert(sizeof...(BatchTags) ==
                  tmpl::size<ElementVariablesTags>::value);
    auto host_batches = std::apply(
        [&](auto&... host_copies) {
          return std::forward_as_tuple(
              host_copies.enqueue_copy(exec, batches)...);
        },
        host_copies_);
    exec.fence(
        "evolution::dg::subcell::fd_patch::PatchBase::element_variables");

    std::vector<ElementVariables> result(patch_elements_.number_of_elements());
    for (size_t slot = 0; slot < result.size(); ++slot) {
      [&]<size_t... Is>(std::index_sequence<Is...> /*meta*/) {
        (set_data_ref_to_element(
             make_not_null(
                 &get<tmpl::at_c<ElementVariablesTags, Is>>(result[slot])),
             std::get<Is>(host_batches), slot),
         ...);
      }(std::make_index_sequence<sizeof...(BatchTags)>{});
    }
    return result;
  }

 private:
  FdPatchElements patch_elements_{};
  size_t number_of_rdmp_values_ = 0;
  // The variables of the elements copied to the host for the element boxes
  template <typename Tag>
  struct host_copy {
    using type = evolution::patches::HostCopy<typename Tag::type::tags_list>;
  };
  tmpl::wrap<tmpl::transform<ElementVariablesTags, host_copy<tmpl::_1>>,
             std::tuple>
      host_copies_{};
};
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
