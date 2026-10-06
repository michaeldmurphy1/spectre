// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/Device/VolumeView.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

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
VolumeView make_volume_view(const std::string& label,
                            const size_t number_of_elements,
                            const size_t number_of_variables,
                            const Index<Dim>& volume_extents,
                            const size_t ghost_width) {
  auto extents = extents_3d(volume_extents);
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(extents, d) += 2 * ghost_width;
  }
  return VolumeView{label,      number_of_elements, number_of_variables,
                    extents[2], extents[1],         extents[0]};
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
  const size_t ghost_width = (padded_vars.extent(4) - extents[0]) / 2;
  ASSERT(element < padded_vars.extent(0),
         "Element " << element << " is out of bounds for a view of "
                    << padded_vars.extent(0) << " elements");
  ASSERT(volume_vars.size() == volume_extents.product() * number_of_variables,
         "The volume data has size " << volume_vars.size() << " but expected "
                                     << volume_extents.product() << " points "
                                     << "times " << number_of_variables
                                     << " variables");
#ifdef SPECTRE_DEBUG
  for (size_t d = 0; d < 3; ++d) {
    const size_t expected_extent =
        gsl::at(extents, d) + (d < Dim ? 2 * ghost_width : 0);
    ASSERT(padded_vars.extent(4 - d) == expected_extent,
           "The padded view has extent "
               << padded_vars.extent(4 - d) << " in dimension " << d
               << " but expected " << expected_extent << " for "
               << gsl::at(extents, d) << " cells and ghost width "
               << ghost_width);
  }
#endif  // SPECTRE_DEBUG
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

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template VolumeView make_volume_view(                                        \
      const std::string& label, size_t number_of_elements,                     \
      size_t number_of_variables, const Index<DIM(data)>& volume_extents,      \
      size_t ghost_width);                                                     \
  template void pack_with_ghost_zones(                                         \
      const VolumeView::host_mirror_type& padded_vars, size_t element,         \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Index<DIM(data)>& volume_extents);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
