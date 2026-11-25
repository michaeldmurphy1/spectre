// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Subitems.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/ErrorIfDataTooBig.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/Triggers/OnSubsteps.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
/// The events that the elements of a patch run as the elements of the
/// `DgElementArray` do, on their DataBox (see `Patch::make_element_box`):
/// events that each element contributes data to
template <typename T>
struct RunsOnElements : std::false_type {};
/// \cond
template <size_t Dim, typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag>
struct RunsOnElements<::dg::Events::ObserveFields<
    Dim, Fields, NonTensorComputeTags, ArraySectionIdTag>> : std::true_type {};
template <typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct RunsOnElements<::Events::ObserveNorms<Fields, NonTensorComputeTags,
                                             ArraySectionIdTag, OptionName>>
    : std::true_type {};
template <typename System, typename VariablesTags>
struct RunsOnElements<::Events::ObserveTimeStep<System, VariablesTags>>
    : std::true_type {};
template <size_t Dim, typename Fields, typename NonTensorComputeTags>
struct RunsOnElements<
    ::Events::ErrorIfDataTooBig<Dim, Fields, NonTensorComputeTags>>
    : std::true_type {};
/// \endcond

/// The events that the elements of a patch run at the times of dense
/// triggers, on their DataBox with the dense output (a subset of
/// `RunsOnElements`)
template <typename T>
struct RunsDenseOnElements : std::false_type {};
/// \cond
template <size_t Dim, typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag>
struct RunsDenseOnElements<::dg::Events::ObserveFields<
    Dim, Fields, NonTensorComputeTags, ArraySectionIdTag>> : std::true_type {};
template <typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct RunsDenseOnElements<::Events::ObserveNorms<
    Fields, NonTensorComputeTags, ArraySectionIdTag, OptionName>>
    : std::true_type {};
/// \endcond

/// The events of the executable that the elements of a patch run (see
/// `RunsOnElements`)
template <typename Metavariables>
using events_on_elements = tmpl::filter<
    tmpl::at<typename Metavariables::factory_creation::factory_classes, Event>,
    RunsOnElements<tmpl::_1>>;

/// Whether `T` is an `Events::ObserveNorms`
template <typename T>
struct IsObserveNorms : std::false_type {};
/// \cond
template <typename Fields, typename NonTensorComputeTags,
          typename ArraySectionIdTag, typename OptionName>
struct IsObserveNorms<::Events::ObserveNorms<Fields, NonTensorComputeTags,
                                             ArraySectionIdTag, OptionName>>
    : std::true_type {};
/// \endcond

/// The `ObserveNorms` events of the executable
template <typename Metavariables>
using observe_norms_events = tmpl::filter<
    tmpl::at<typename Metavariables::factory_creation::factory_classes, Event>,
    IsObserveNorms<tmpl::_1>>;

/// Whether `EventType` integrates tensors over the volume of the elements
/// (e.g. `control_system::measurements::VolumeIntegralsEvent`), which it
/// defines by the compute tags `volume_integrand_tags`
template <typename EventType>
concept IntegratesOverVolume =
    requires { typename EventType::volume_integrand_tags; };

namespace EventsOnElements_detail {
template <typename EventType>
struct IsVolumeIntegralsEvent
    : std::bool_constant<IntegratesOverVolume<EventType>> {};

template <typename EventType>
struct get_observation_compute_tags {
  using type = typename EventType::compute_tags_for_observation_box;
};

// Whether the DataBox of the elements (see `Patch::make_element_box`) has the
// item of the `ComputeTag` or one of its subitems, e.g. the geometry for
// observations, which the events then don't compute again
//
// (A partial specialization rather than a lambda in the template argument,
// since nvcc's front end crashes on the lambda.)
template <typename ElementBox, typename Subitems>
struct any_in_element_box;

template <typename ElementBox, typename... Subitems>
struct any_in_element_box<ElementBox, tmpl::list<Subitems...>>
    : std::bool_constant<(false or ... or
                          db::tag_is_retrievable_v<Subitems, ElementBox>)> {};

template <typename ComputeTag, typename ElementBox>
struct in_element_box
    : std::bool_constant<
          db::tag_is_retrievable_v<typename ComputeTag::base, ElementBox> or
          any_in_element_box<ElementBox,
                             typename db::Subitems<ComputeTag>::type>::value> {
};

// Whether `event` is of one of the `EventTypes`
template <typename EventTypes>
bool is_one_of(const Event& event) {
  bool result = false;
  tmpl::for_each<EventTypes>(
      [&]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
        result = result or dynamic_cast<const EventType*>(&event) != nullptr;
      });
  return result;
}
}  // namespace EventsOnElements_detail

/// The events of the executable that integrate tensors over the volume of the
/// elements (see `IntegratesOverVolume`)
template <typename Metavariables>
using volume_integrals_events = tmpl::filter<
    tmpl::at<typename Metavariables::factory_creation::factory_classes, Event>,
    EventsOnElements_detail::IsVolumeIntegralsEvent<tmpl::_1>>;

/// The compute tags of the `ObservationBox` of the events that run on the
/// elements, as `EventsAndTriggers::run_events`
template <typename Metavariables>
using observation_compute_tags = tmpl::remove_duplicates<tmpl::filter<
    tmpl::flatten<tmpl::transform<
        events_on_elements<Metavariables>,
        EventsOnElements_detail::get_observation_compute_tags<tmpl::_1>>>,
    db::is_compute_tag<tmpl::_1>>>;

/// The `observation_compute_tags` of the items that the DataBox
/// `ElementBox` of the elements doesn't have (see `Patch::make_element_box`)
template <typename Metavariables, typename ElementBox>
using element_observation_compute_tags = tmpl::remove_if<
    observation_compute_tags<Metavariables>,
    EventsOnElements_detail::in_element_box<tmpl::_1, tmpl::pin<ElementBox>>>;

/// Whether the elements of a patch run the `event` (see `RunsOnElements`)
template <typename Metavariables>
bool runs_on_elements(const Event& event) {
  return EventsOnElements_detail::is_one_of<events_on_elements<Metavariables>>(
      event);
}

/// Whether the elements of a patch run the `event` at the times of dense
/// triggers (see `RunsDenseOnElements`)
template <typename Metavariables>
bool runs_dense_on_elements(const Event& event) {
  return EventsOnElements_detail::is_one_of<tmpl::filter<
      events_on_elements<Metavariables>, RunsDenseOnElements<tmpl::_1>>>(event);
}

/// Whether the `event` integrates tensors over the volume of the elements,
/// e.g. a measurement of a control system (see `IntegratesOverVolume`)
template <typename Metavariables>
bool integrates_over_volume(const Event& event) {
  return EventsOnElements_detail::is_one_of<
      volume_integrals_events<Metavariables>>(event);
}

/// The names of the `EventTypes` in the input file, without template
/// arguments and each once
template <typename EventTypes>
std::string event_names() {
  std::vector<std::string> names{};
  tmpl::for_each<EventTypes>(
      [&names]<typename EventType>(tmpl::type_<EventType> /*meta*/) {
        std::string name = pretty_type::name<EventType>();
        name = name.substr(0, name.find('<'));
        if (not alg::found(names, name)) {
          names.push_back(std::move(name));
        }
      });
  std::string result{};
  for (const std::string& name : names) {
    result += (result.empty() ? "" : ", ") + name;
  }
  return result;
}

/// Errors because patches don't run the `event`, so the DG elements would
/// wait for them forever (e.g. in a reduction or an interpolation)
template <typename Metavariables>
[[noreturn]] void error_about_unsupported_event(const Event& event) {
  ERROR("FD patches only run the events Completion, ChangeSlabSize, "
        << event_names<events_on_elements<Metavariables>>() << ", but got "
        << pretty_type::get_runtime_type_name(event));
}

/// Errors because patches don't run the `event` at the times of dense
/// triggers (see `error_about_unsupported_event`)
template <typename Metavariables>
[[noreturn]] void error_about_unsupported_dense_event(const Event& event) {
  const std::string dense_events =
      event_names<tmpl::filter<events_on_elements<Metavariables>,
                               RunsDenseOnElements<tmpl::_1>>>();
  const std::string measurements =
      event_names<volume_integrals_events<Metavariables>>();
  ERROR(
      "FD patches only run the events Completion, "
      << dense_events
      << (measurements.empty()
              ? std::string{}
              : " and the volume integrals of the measurements " + measurements)
      << " in EventsAndDenseTriggers, but got "
      << pretty_type::get_runtime_type_name(event));
}

/// The events whose triggers fire, sorted by how patches run them (see
/// `triggered_events`)
struct TriggeredEvents {
  /// Whether an `Events::Completion` fired
  bool completed = false;
  /// The events that run on the elements (see `RunsOnElements`)
  std::vector<const Event*> on_elements{};
  /// The `Events::ChangeSlabSize` events
  std::vector<const ::Events::ChangeSlabSize*> slab_size_changes{};
};

/*!
 * \brief The events of `events_and_triggers` whose triggers fire on the
 * `box`, sorted by how patches run them.
 *
 * \details All events and triggers are checked whether or not they fire, so
 * that a run fails early: patches only run `Events::Completion`,
 * `Events::ChangeSlabSize` and the events that run on the elements (see
 * `RunsOnElements`), and they run the events at the first substep of a step,
 * so `Triggers::OnSubsteps` is an error too.
 */
template <typename Metavariables, typename DbTagsList>
TriggeredEvents triggered_events(const EventsAndTriggers& events_and_triggers,
                                 const db::DataBox<DbTagsList>& box) {
  TriggeredEvents result{};
  events_and_triggers.for_each_trigger_and_events([&](const Trigger& trigger,
                                                      const auto& events) {
    if (dynamic_cast<const ::Triggers::OnSubsteps*>(&trigger) != nullptr) {
      ERROR("FD patches don't support OnSubsteps triggers.");
    }
    for (const auto& event : events) {
      if (dynamic_cast<const ::Events::Completion*>(event.get()) == nullptr and
          dynamic_cast<const ::Events::ChangeSlabSize*>(event.get()) ==
              nullptr and
          not runs_on_elements<Metavariables>(*event)) {
        error_about_unsupported_event<Metavariables>(*event);
      }
    }
    if (not trigger.is_triggered(box)) {
      return;
    }
    for (const auto& event : events) {
      if (dynamic_cast<const ::Events::Completion*>(event.get()) != nullptr) {
        result.completed = true;
      } else if (const auto* const change_slab_size =
                     dynamic_cast<const ::Events::ChangeSlabSize*>(event.get());
                 change_slab_size != nullptr) {
        result.slab_size_changes.push_back(change_slab_size);
      } else {
        result.on_elements.push_back(event.get());
      }
    }
  });
  return result;
}
}  // namespace evolution::patches
