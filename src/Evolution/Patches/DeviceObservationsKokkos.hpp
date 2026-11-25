// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Patches/DeviceComputeTagsKokkos.hpp"
#include "Evolution/Patches/EventsOnElements.hpp"
#include "Evolution/Patches/PatchState.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/ExecutionSpaceKokkos.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace evolution::patches {
/// Whether patches of type `Patch` can evaluate observables on the device (see
/// `evolution::patches::device_compute`)
template <typename Patch>
concept ComputesDeviceObservables =
    requires { typename Patch::device_observable_sources; };

/// The tensors of the `ObserveNorms` `event` that patches of type `Patch`
/// can't compute on the device, or `std::nullopt` if `event` isn't an
/// `ObserveNorms` that patches can observe on the device at all
template <typename Patch, typename Metavariables>
std::optional<std::vector<std::string>> tensors_not_observable_on_device(
    const Event& event) {
  if constexpr (not ComputesDeviceObservables<Patch>) {
    (void)event;
    return std::nullopt;
  } else {
    std::optional<std::vector<std::string>> result{};
    tmpl::for_each<observe_norms_events<Metavariables>>(
        [&]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
          const auto* const derived_event =
              dynamic_cast<const EventType*>(&event);
          // Sections need the DataBox of the elements
          if (derived_event != nullptr and
              std::is_same_v<typename EventType::array_section_id_tag, void>) {
            result = std::vector<std::string>{};
            for (const std::string& name : derived_event->tensor_names()) {
              if (not device_compute::is_device_observable<
                      Metavariables, typename Patch::device_observable_sources>(
                      name)) {
                result->push_back(name);
              }
            }
          }
        });
    return result;
  }
}

/// Whether the patch computes the observations of `event` on the device:
/// `ObserveNorms` of tensors the patch can all compute on the device
template <typename Patch, typename Metavariables>
bool observes_on_device(const Event& event) {
  const auto missing =
      tensors_not_observable_on_device<Patch, Metavariables>(event);
  return missing.has_value() and missing->empty();
}

/// The slots of the elements that contribute to each subfile of `event`, in
/// the order of the slots. Elements that `event` doesn't observe are left out.
template <typename ObserveNormsEvent, size_t Dim>
std::vector<std::vector<size_t>> slots_by_subfile(
    const ObserveNormsEvent& event,
    const std::optional<std::string>& section_observation_key,
    const Domain<Dim>& domain, const std::vector<ElementId<Dim>>& element_ids) {
  std::vector<std::string> subfiles{};
  std::vector<std::vector<size_t>> result{};
  for (size_t slot = 0; slot < element_ids.size(); ++slot) {
    const std::optional<std::string> subfile = event.subfile_path_for_element(
        section_observation_key, domain, element_ids[slot]);
    if (not subfile.has_value()) {
      continue;
    }
    const auto it = alg::find(subfiles, *subfile);
    if (it == subfiles.end()) {
      subfiles.push_back(*subfile);
      result.push_back({slot});
    } else {
      result[static_cast<size_t>(it - subfiles.begin())].push_back(slot);
    }
  }
  return result;
}

/// The norm statistics of the tensors of the elements in the `slots`, combined
/// (see `Events::combine`), and the volume of these elements, from the
/// `observables` of all elements of a patch
std::pair<::Events::NormStatisticsOfTensors, double> combine_slot_statistics(
    const device_compute::DeviceNormStatistics& observables,
    const std::vector<size_t>& slots);

/*!
 * \brief Runs the `ObserveNorms` `events` (see `observes_on_device`) on the
 * device.
 *
 * \details The patch computes the observed tensors of all its elements on the
 * device, they are reduced there (see `device_compute::device_norm_statistics`)
 * and only the results of each element are copied to the host. The patch
 * combines those of its elements that contribute to the same subfile (see
 * `combine_slot_statistics`) and contributes them to the observer once, which
 * gives the observation of the events on the DataBoxes of the elements up to
 * the order of the sums. With `dense_output`, the tensors are computed from the
 * last dense output of the patch.
 */
template <typename PatchTag, typename ParallelComponent, typename DbTagsList,
          typename Metavariables>
void observe_norms_on_device(const std::vector<const Event*>& events,
                             const gsl::not_null<db::DataBox<DbTagsList>*> box,
                             Parallel::GlobalCache<Metavariables>& cache,
                             const Event::ObservationValue& observation_value,
                             const bool dense_output) {
  constexpr size_t dim = Metavariables::volume_dim;
  const Kokkos::Profiling::ScopedRegion region{
      "patches::observe_norms_on_device"};
  const auto& exec = Parallel::device_execution_space();
  // Report failures of the last substep before anything sees its state
  db::get<PatchTag>(*box).check_last_step(exec);

  // The tensors of all events, each computed once
  std::vector<std::string> tensor_names{};
  const auto for_each_event = [&events](const auto& function) {
    for (const Event* const event : events) {
      call_with_dynamic_type<void, observe_norms_events<Metavariables>>(
          event, function);
    }
  };
  for_each_event([&tensor_names](const auto* const derived_event) {
    for (const std::string& name : derived_event->tensor_names()) {
      if (not alg::found(tensor_names, name)) {
        tensor_names.push_back(name);
      }
    }
  });

  const double time = db::get<::Tags::Time>(*box);
  using Sources = typename PatchTag::type::device_observable_sources;
  const Sources sources = with_state<PatchTag>(*box, [&](const auto&... state) {
    return db::mutate<PatchTag>(
        [&](const auto mutable_patch) {
          return mutable_patch->observable_sources(
              exec, state..., dense_output, time,
              Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
        },
        box);
  });
  const device_compute::DeviceNormStatistics observables =
      device_compute::device_norm_statistics<Metavariables>(exec, sources,
                                                            tensor_names);

  const Kokkos::Profiling::ScopedRegion contribute_region{
      "patches::observe_norms_on_device::contribute"};
  const auto& patch = db::get<PatchTag>(*box);
  const auto& element_ids = patch.patch_elements().element_ids();
  const auto& domain = db::get<::domain::Tags::Domain<dim>>(*box);
  const size_t number_of_points =
      patch.patch_elements().subcell_mesh().number_of_grid_points();
  const std::optional<std::string> section_observation_key =
      observers::get_section_observation_key<void>(*box);
  for_each_event([&](const auto* const derived_event) {
    // The elements contribute once per subfile, combined, from the first of
    // them, which is the only one registered (see `RegisterWithObservers`)
    for (const auto& slots : slots_by_subfile(
             *derived_event, section_observation_key, domain, element_ids)) {
      const auto [tensors_statistics, volume] =
          combine_slot_statistics(observables, slots);
      derived_event->contribute_statistics(
          cache, element_ids[slots.front()],
          static_cast<const ParallelComponent*>(nullptr), observation_value,
          section_observation_key, domain, number_of_points * slots.size(),
          volume, tensors_statistics);
    }
  });
}

/// `sum += term` for the integrals of an event that integrates over the
/// volume (see `IntegratesOverVolume`)
template <typename... Arrays>
void add_integrals(const gsl::not_null<std::tuple<Arrays...>*> sum,
                   const std::tuple<Arrays...>& term) {
  [&]<size_t... Is>(std::index_sequence<Is...> /*meta*/) {
    const auto add = [](auto& sum_array, const auto& term_array) {
      for (size_t i = 0; i < sum_array.size(); ++i) {
        gsl::at(sum_array, i) += gsl::at(term_array, i);
      }
    };
    (add(std::get<Is>(*sum), std::get<Is>(term)), ...);
  }(std::index_sequence_for<Arrays...>{});
}

/// The integrals of the `EventType` (see `IntegratesOverVolume`) over all
/// elements of the patch, evaluated on the device from the compute tags of the
/// integrands and reduced there (the `integral` of
/// `device_compute::device_norm_statistics`). With `dense_output`, from the
/// last dense output of the patch.
template <typename PatchTag, typename EventType, typename DbTagsList,
          typename Metavariables>
typename EventType::integrals_type volume_integrals_on_device(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    Parallel::GlobalCache<Metavariables>& cache, const bool dense_output) {
  const Kokkos::Profiling::ScopedRegion region{
      "patches::volume_integrals_on_device"};
  const auto& exec = Parallel::device_execution_space();
  // Report failures of the last substep before anything sees its state
  db::get<PatchTag>(*box).check_last_step(exec);
  const double time = db::get<::Tags::Time>(*box);
  using Sources = typename PatchTag::type::device_observable_sources;
  const Sources sources = with_state<PatchTag>(*box, [&](const auto&... state) {
    return db::mutate<PatchTag>(
        [&](const auto mutable_patch) {
          return mutable_patch->observable_sources(
              exec, state..., dense_output, time,
              Parallel::get<::domain::Tags::FunctionsOfTime>(cache));
        },
        box);
  });
  using integrand_tags = typename EventType::volume_integrand_tags;
  std::vector<std::string> tensor_names{};
  tmpl::for_each<integrand_tags>([&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    tensor_names.push_back(db::tag_name<Tag>());
  });
  const device_compute::DeviceNormStatistics observables =
      device_compute::device_norm_statistics<Metavariables>(exec, sources,
                                                            tensor_names);
  typename EventType::integrals_type result{};
  [&]<size_t... Is>(std::index_sequence<Is...> /*meta*/) {
    const auto sum_over_elements = [&](auto& integrals, const size_t tensor) {
      const size_t first_column = observables.tensors[tensor].first_component;
      for (size_t c = 0; c < integrals.size(); ++c) {
        gsl::at(integrals, c) = 0.0;
        for (size_t e = 0; e < observables.statistics.extent(0); ++e) {
          gsl::at(integrals, c) +=
              observables.statistics(e, first_column + c).integral;
        }
      }
    };
    (sum_over_elements(std::get<Is>(result), Is), ...);
  }(std::make_index_sequence<tmpl::size<integrand_tags>::value>{});
  return result;
}

/// Prints which `ObserveNorms` events patches observe on the device and, for
/// the others, which tensors they can't compute there (see
/// `observes_on_device`), so that a slow observation isn't a surprise
template <typename Patch, typename Metavariables, typename DbTagsList>
void report_device_observations(const db::DataBox<DbTagsList>& box) {
  const auto report = [](const Event& event) {
    tmpl::for_each<observe_norms_events<Metavariables>>(
        [&event]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
          const auto* const derived_event =
              dynamic_cast<const EventType*>(&event);
          if (derived_event == nullptr) {
            return;
          }
          std::string tensors{};
          for (const std::string& name : derived_event->tensor_names()) {
            tensors += (tensors.empty() ? "" : ", ") + name;
          }
          const auto missing =
              tensors_not_observable_on_device<Patch, Metavariables>(event);
          if (missing.has_value() and missing->empty()) {
            Parallel::printf("Patches observe the norms of %s on the device.\n",
                             tensors);
          } else {
            std::string reason{};
            if (missing.has_value()) {
              for (const std::string& name : *missing) {
                reason += (reason.empty() ? "" : ", ") + name;
              }
              reason = "they can't compute " + reason + " on the device";
            } else {
              reason = ComputesDeviceObservables<Patch>
                           ? "the event uses array sections"
                           : "they can't compute observables on the device";
            }
            Parallel::printf(
                "Patches observe the norms of %s on the host, because %s.\n",
                tensors, reason);
          }
        });
  };
  db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>(box)
      .for_each_event(report);
  db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(box)
      .for_each_event(report);
  db::get<::Tags::EventsAndDenseTriggers>(box).for_each_event(report);
}
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
