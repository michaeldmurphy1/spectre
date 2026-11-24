// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeSignalingNan.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device {
namespace {
// Extents in (x, y, z) order, with 1 for unused dimensions
template <size_t Dim>
std::array<size_t, 3> extents_3d(const Index<Dim>& volume_extents) {
  std::array<size_t, 3> result{{1, 1, 1}};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = volume_extents[d];
  }
  return result;
}
}  // namespace

template <size_t Dim>
VolumeView make_volume_view(const Kokkos::DefaultExecutionSpace& exec,
                            const std::string& label,
                            const size_t number_of_elements,
                            const size_t number_of_variables,
                            const Index<Dim>& volume_extents,
                            const size_t ghost_width) {
  auto extents = extents_3d(volume_extents);
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(extents, d) += 2 * ghost_width;
  }
  VolumeView result{
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label),
      number_of_elements,
      number_of_variables,
      extents[2],
      extents[1],
      extents[0]};
#ifdef SPECTRE_NAN_INIT
  // Like `Variables`, so that reading uninitialized data, e.g. the unused
  // edges and corners of a padded view, is caught. Kokkos compares the fill
  // value to zero to decide whether it can memset, which raises an FPE for a
  // signaling NaN.
  const ScopedFpeState disable_fpes(false);
  Kokkos::deep_copy(exec, result, make_signaling_NaN<double>());
#endif  // SPECTRE_NAN_INIT
  return result;
}

template <size_t Dim>
void pack_with_ghost_zones(
    const VolumeView::host_mirror_type& padded_vars, const size_t element,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents) {
  const size_t number_of_variables = padded_vars.extent(1);
  ASSERT(number_of_variables > 0 and volume_extents.product() > 0,
         "The volume data must not be empty");
  const auto extents = extents_3d(volume_extents);
  const size_t ghost_width = detail::ghost_width(padded_vars, volume_extents);
  ASSERT(element < padded_vars.extent(0),
         "Element " << element << " is out of bounds for a view of "
                    << padded_vars.extent(0) << " elements");
  ASSERT(volume_vars.size() == volume_extents.product() * number_of_variables,
         "The volume data has size " << volume_vars.size() << " but expected "
                                     << volume_extents.product() << " points "
                                     << "times " << number_of_variables
                                     << " variables");
  std::array<size_t, 3> offsets{{0, 0, 0}};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(offsets, d) = ghost_width;
  }

  // Copy the volume data into the interior of the padded view
  for (size_t var = 0, index = 0; var < number_of_variables; ++var) {
    for (size_t z = 0; z < extents[2]; ++z) {
      for (size_t y = 0; y < extents[1]; ++y) {
        for (size_t x = 0; x < extents[0]; ++x, ++index) {
          padded_vars(element, var, z + offsets[2], y + offsets[1],
                      x + offsets[0]) = volume_vars[index];
        }
      }
    }
  }

  // Copy the ghost cells closest to the element into the ghost zones
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const size_t dim = direction.dimension();
    const auto& ghost_data = ghost_cell_vars.at(direction);
    auto ghost_extents = extents;
    const size_t points_per_slice =
        number_of_variables * volume_extents.slice_away(dim).product();
    gsl::at(ghost_extents, dim) = ghost_data.size() / points_per_slice;
    const size_t neighbor_width = gsl::at(ghost_extents, dim);
    ASSERT(ghost_data.size() == neighbor_width * points_per_slice and
               neighbor_width >= ghost_width,
           "The ghost data in direction "
               << direction << " has size " << ghost_data.size()
               << ", which is not a multiple of the " << points_per_slice
               << " points on a slice times at least " << ghost_width
               << " ghost cells");
    // Range of neighbor cells to copy perpendicular to the face, and the
    // padded index of the first one
    const size_t first_ghost_cell =
        direction.side() == Side::Lower ? neighbor_width - ghost_width : 0;
    const size_t first_padded_cell = direction.side() == Side::Lower
                                         ? 0
                                         : ghost_width + gsl::at(extents, dim);

    for (size_t var = 0, index = 0; var < number_of_variables; ++var) {
      for (size_t z = 0; z < ghost_extents[2]; ++z) {
        for (size_t y = 0; y < ghost_extents[1]; ++y) {
          for (size_t x = 0; x < ghost_extents[0]; ++x, ++index) {
            const std::array<size_t, 3> ghost_index{{x, y, z}};
            const size_t ghost_cell = gsl::at(ghost_index, dim);
            if (ghost_cell < first_ghost_cell or
                ghost_cell >= first_ghost_cell + ghost_width) {
              continue;
            }
            std::array<size_t, 3> padded_index{
                {x + offsets[0], y + offsets[1], z + offsets[2]}};
            gsl::at(padded_index, dim) =
                first_padded_cell + ghost_cell - first_ghost_cell;
            padded_vars(element, var, padded_index[2], padded_index[1],
                        padded_index[0]) = ghost_data[index];
          }
        }
      }
    }
  }
}

template <size_t Dim>
SplitVolumeView split_volume_view(const ConstStridedVolumeView& padded_vars,
                                  const Index<Dim>& volume_extents) {
  const size_t ghost_width = detail::ghost_width(padded_vars, volume_extents);
  std::array<std::pair<size_t, size_t>, 3> interior{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(interior, d) =
        d < Dim ? std::pair{ghost_width, ghost_width + volume_extents[d]}
                : std::pair{0_st, 1_st};
  }
  return {.interior = Kokkos::subview(padded_vars, Kokkos::ALL(), Kokkos::ALL(),
                                      interior[2], interior[1], interior[0]),
          .padded = padded_vars};
}

SplitVolumeView variable_range(const SplitVolumeView& volume_vars,
                               const size_t begin, const size_t end) {
  ASSERT(begin <= end and end <= volume_vars.interior.extent(1) and
             end <= volume_vars.padded.extent(1),
         "The variables [" << begin << ", " << end
                           << ") aren't in the volume data of "
                           << volume_vars.interior.extent(1) << " variables");
  const auto variables = Kokkos::make_pair(begin, end);
  return {
      .interior =
          Kokkos::subview(volume_vars.interior, Kokkos::ALL(), variables,
                          Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
      .padded = Kokkos::subview(volume_vars.padded, Kokkos::ALL(), variables,
                                Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL())};
}

template <size_t Dim>
StridedVolumeView batch_volume_view(
    const Kokkos::View<double***, Kokkos::LayoutLeft>& storage,
    const size_t first, const size_t count, const Index<Dim>& volume_extents) {
  const size_t number_of_points = storage.extent(0);
  const size_t number_of_components = storage.extent(1);
  ASSERT(number_of_points == volume_extents.product(),
         "The batch has " << number_of_points
                          << " grid points but the extents are "
                          << volume_extents);
  ASSERT(first + count <= number_of_components,
         "The components [" << first << ", " << first + count
                            << ") aren't in the batch of "
                            << number_of_components << " components");
  const auto extents = extents_3d(volume_extents);
  return StridedVolumeView(
      // The first point of component `first` of element 0
      Kokkos::subview(storage, Kokkos::ALL(), first, Kokkos::ALL()).data(),
      Kokkos::LayoutStride(
          storage.extent(2), number_of_components * number_of_points, count,
          number_of_points, extents[2], extents[1] * extents[0], extents[1],
          extents[0], extents[0], 1));
}

namespace detail {
namespace {
int checked_size(const size_t number_of_elements,
                 const size_t number_of_variables,
                 const Kokkos::Array<int, 3>& extents) {
  const size_t size = number_of_elements * number_of_variables *
                      static_cast<size_t>(extents[0] * extents[1] * extents[2]);
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    ERROR("The batch has " << size << " cells (" << number_of_elements
                           << " elements, " << number_of_variables
                           << " variables), which is too many to enumerate "
                              "with an int. Split it into smaller batches.");
  }
  return static_cast<int>(size);
}
}  // namespace

BatchCells::BatchCells(const size_t number_of_elements,
                       const size_t number_of_variables,
                       const Kokkos::Array<int, 3>& extents)
    : number_of_variables_(static_cast<int>(number_of_variables)),
      extents_(extents),
      size_(checked_size(number_of_elements, number_of_variables, extents)) {}

BatchPoints::BatchPoints(const size_t number_of_elements,
                         const Kokkos::Array<int, 3>& extents)
    : extents_(extents), size_(checked_size(number_of_elements, 1, extents)) {}
}  // namespace detail

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template VolumeView make_volume_view(                                        \
      const Kokkos::DefaultExecutionSpace& exec, const std::string& label,     \
      size_t number_of_elements, size_t number_of_variables,                   \
      const Index<DIM(data)>& volume_extents, size_t ghost_width);             \
  template void pack_with_ghost_zones(                                         \
      const VolumeView::host_mirror_type& padded_vars, size_t element,         \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Index<DIM(data)>& volume_extents);                                 \
  template SplitVolumeView split_volume_view(                                  \
      const ConstStridedVolumeView& padded_vars,                               \
      const Index<DIM(data)>& volume_extents);                                 \
  template StridedVolumeView batch_volume_view(                                \
      const Kokkos::View<double***, Kokkos::LayoutLeft>& storage,              \
      size_t first, size_t count, const Index<DIM(data)>& volume_extents);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
