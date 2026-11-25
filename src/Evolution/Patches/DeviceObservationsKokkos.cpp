// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Patches/DeviceObservationsKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <cstddef>
#include <utility>
#include <vector>

#include "Evolution/Patches/DeviceComputeTagsKokkos.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"

namespace evolution::patches {
std::pair<::Events::NormStatisticsOfTensors, double> combine_slot_statistics(
    const device_compute::DeviceNormStatistics& observables,
    const std::vector<size_t>& slots) {
  const auto& statistics = observables.statistics;
  ::Events::NormStatisticsOfTensors tensors_statistics{};
  for (const auto& tensor : observables.tensors) {
    std::vector<::Events::NormStatistics> component_statistics(
        tensor.component_names.size());
    for (size_t c = 0; c < component_statistics.size(); ++c) {
      component_statistics[c] =
          statistics(slots.front(), tensor.first_component + c);
      for (size_t i = 1; i < slots.size(); ++i) {
        component_statistics[c] =
            ::Events::combine(component_statistics[c],
                              statistics(slots[i], tensor.first_component + c));
      }
    }
    tensors_statistics.emplace(
        tensor.name,
        std::pair{tensor.component_names, std::move(component_statistics)});
  }
  // The last component is the volume
  const size_t volume_component = statistics.extent(1) - 1;
  double volume = 0.0;
  for (const size_t slot : slots) {
    volume += statistics(slot, volume_component).integral;
  }
  return {std::move(tensors_statistics), volume};
}
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
