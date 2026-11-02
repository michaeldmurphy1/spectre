// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/SliceDataKokkos.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace evolution::dg::subcell {
template <size_t Dim>
DirectionMap<Dim, DataVector> slice_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const DataVector& volume_subcell_vars, const Index<Dim>& subcell_extents,
    const size_t number_of_ghost_points,
    const std::unordered_set<Direction<Dim>>& directions_to_slice,
    const size_t additional_buffer,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        fd_to_neighbor_fd_interpolants) {
  const size_t number_of_points = subcell_extents.product();
  const size_t number_of_components =
      volume_subcell_vars.size() / number_of_points;
  ASSERT(volume_subcell_vars.size() == number_of_components * number_of_points,
         "The volume data has " << volume_subcell_vars.size()
                                << " values, which is not a multiple of the "
                                << number_of_points << " points");

  // As in `slice_data`: interpolate where there is an interpolant, leave
  // the directions with `std::nullopt` unfilled, and copy otherwise.
  DirectionMap<Dim, const intrp::Irregular<Dim>*> interpolants{};
  std::unordered_set<Direction<Dim>> unfilled_directions{};
  for (const auto& [directional_id, interpolant] :
       fd_to_neighbor_fd_interpolants) {
    if (not interpolant.has_value()) {
      unfilled_directions.insert(directional_id.direction());
    } else if (directions_to_slice.contains(directional_id.direction())) {
      interpolants[directional_id.direction()] = &interpolant.value();
    }
  }

  DirectionMap<Dim, DataVector> result{};
  // Offset of each direction in the buffer of all sliced data
  DirectionMap<Dim, size_t> offsets{};
  size_t buffer_size = 0;
  fd::device::Transfers transfers{};
  for (const auto& direction : directions_to_slice) {
    const size_t dimension = direction.dimension();
    const size_t sliced_points =
        number_of_ghost_points *
        subcell_extents.slice_away(dimension).product();
    result[direction] =
        DataVector{sliced_points * number_of_components + additional_buffer};
    if (unfilled_directions.contains(direction) and
        not interpolants.contains(direction)) {
      continue;
    }
    offsets[direction] = buffer_size;
    std::vector<size_t> destination_points(sliced_points);
    std::iota(destination_points.begin(), destination_points.end(), size_t{0});
    if (interpolants.contains(direction)) {
      transfers.add_matrix(0, number_of_points, buffer_size, sliced_points,
                           interpolants.at(direction)->matrix(),
                           destination_points);
    } else {
      ASSERT(number_of_ghost_points <= subcell_extents[dimension],
             "Number of ghost points (" << number_of_ghost_points
                                        << ") is larger than the subcell mesh "
                                           "extent in the slicing direction ("
                                        << subcell_extents[dimension] << ")");
      const size_t first =
          direction.side() == Side::Lower
              ? 0
              : subcell_extents[dimension] - number_of_ghost_points;
      transfers.add_copy(
          0, number_of_points, buffer_size, sliced_points,
          fd::device::slab_points(subcell_extents, dimension, first,
                                  first + number_of_ghost_points),
          destination_points);
    }
    buffer_size += sliced_points * number_of_components;
  }
  if (buffer_size == 0) {
    return result;
  }

  // Slice on the device
  transfers.copy_to_device(exec);
  const Kokkos::View<double*> volume(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "volume data"),
      volume_subcell_vars.size());
  Kokkos::deep_copy(
      exec, volume,
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          volume_subcell_vars.data(), volume_subcell_vars.size()));
  const Kokkos::View<double*> sliced(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sliced data"),
      buffer_size);
  transfers.apply(exec, sliced, volume, number_of_components);
  const auto host_sliced =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, sliced);
  Kokkos::deep_copy(exec, host_sliced, sliced);
  exec.fence();
  for (const auto& [direction, offset] : offsets) {
    auto& sliced_in_direction = result.at(direction);
    const size_t size = sliced_in_direction.size() - additional_buffer;
    std::copy(
        std::next(host_sliced.data(), static_cast<std::ptrdiff_t>(offset)),
        std::next(host_sliced.data(),
                  static_cast<std::ptrdiff_t>(offset + size)),
        sliced_in_direction.data());
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template DirectionMap<DIM(data), DataVector> slice_data_on_device(          \
      const Kokkos::DefaultExecutionSpace& exec,                              \
      const DataVector& volume_subcell_vars,                                  \
      const Index<DIM(data)>& subcell_extents, size_t number_of_ghost_points, \
      const std::unordered_set<Direction<DIM(data)>>& directions_to_slice,    \
      size_t additional_buffer,                                               \
      const DirectionalIdMap<DIM(data),                                       \
                             std::optional<intrp::Irregular<DIM(data)>>>&     \
          fd_to_neighbor_fd_interpolants);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell
#endif  // SPECTRE_KOKKOS
