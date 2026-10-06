// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/Device/VolumeView.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace fd::device::detail {
/*!
 * \brief Reconstruct a batch of elements to the faces with a
 * `Reconstructor` from `fd::reconstruction::detail`, launching one Kokkos
 * kernel per direction.
 *
 * Each thread reconstructs one cell, including the cells of the neighbors
 * that are adjacent to the element, so each face value is written once. The
 * padded volume data must have ghost zones of width
 * `(Reconstructor::stencil_width() + 1) / 2`: the faces on the element
 * boundary need the reconstruction in the neighbor's cell, whose stencil
 * extends one cell further than the stencils in the element.
 */
template <typename Reconstructor, size_t Dim>
void reconstruct(const std::array<VolumeView, Dim>& upper_side_of_face_vars,
                 const std::array<VolumeView, Dim>& lower_side_of_face_vars,
                 const ConstVolumeView& padded_volume_vars,
                 const Index<Dim>& volume_extents) {
  constexpr size_t stencil_width = Reconstructor::stencil_width();
  static_assert(stencil_width % 2 == 1, "The stencil width must be odd");
  constexpr int ghost_width = static_cast<int>(stencil_width + 1) / 2;

  // Extents and ghost widths in (x, y, z) order
  Kokkos::Array<int, 3> extents{1, 1, 1};
  Kokkos::Array<int, 3> offsets{0, 0, 0};
  for (size_t d = 0; d < Dim; ++d) {
    extents[d] = static_cast<int>(volume_extents[d]);
    offsets[d] = ghost_width;
  }
  const auto number_of_elements =
      static_cast<int>(padded_volume_vars.extent(0));
  const auto number_of_variables =
      static_cast<int>(padded_volume_vars.extent(1));
#ifdef SPECTRE_DEBUG
  for (size_t d = 0; d < 3; ++d) {
    ASSERT(padded_volume_vars.extent(4 - d) ==
               static_cast<size_t>(extents[d] + 2 * offsets[d]),
           "The padded volume data has extent "
               << padded_volume_vars.extent(4 - d) << " in dimension " << d
               << " but expected " << extents[d] + 2 * offsets[d]
               << " for ghost zones of width " << ghost_width);
  }
#endif  // SPECTRE_DEBUG

  for (size_t dim = 0; dim < Dim; ++dim) {
    const VolumeView upper_side = gsl::at(upper_side_of_face_vars, dim);
    const VolumeView lower_side = gsl::at(lower_side_of_face_vars, dim);
#ifdef SPECTRE_DEBUG
    for (const auto& face_vars : {upper_side, lower_side}) {
      ASSERT(face_vars.extent(0) == padded_volume_vars.extent(0) and
                 face_vars.extent(1) == padded_volume_vars.extent(1),
             "The face data must have the same number of elements and "
             "variables as the volume data");
      for (size_t d = 0; d < 3; ++d) {
        const auto expected_extent =
            static_cast<size_t>(extents[d] + (d == dim ? 1 : 0));
        ASSERT(face_vars.extent(4 - d) == expected_extent,
               "The face data in direction "
                   << dim << " has extent " << face_vars.extent(4 - d)
                   << " in dimension " << d << " but expected "
                   << expected_extent);
      }
    }
#endif  // SPECTRE_DEBUG
    const auto stride = static_cast<int>(padded_volume_vars.stride(4 - dim));
    const int number_of_cells = extents[dim];
    // Also reconstruct in the neighbors' cells -1 and `number_of_cells`
    Kokkos::Array<int, 3> end = extents;
    end[dim] += 2;
    Kokkos::parallel_for(
        "fd::device::reconstruct",
        Kokkos::MDRangePolicy<Kokkos::Rank<5>, Kokkos::IndexType<int>>(
            {0, 0, 0, 0, 0},
            {number_of_elements, number_of_variables, end[2], end[1], end[0]}),
        KOKKOS_LAMBDA(const int element, const int var, const int z,
                      const int y, const int x) {
          Kokkos::Array<int, 3> cell{x, y, z};
          cell[dim] -= 1;
          const double* const q =
              &padded_volume_vars(element, var, cell[2] + offsets[2],
                                  cell[1] + offsets[1], cell[0] + offsets[0]);
          const auto lower_and_upper_values =
              Reconstructor::pointwise(q, stride);
          // The value on the lower face of the cell is on the upper side of
          // face `cell`, and the value on the upper face of the cell is on the
          // lower side of face `cell + 1`
          if (cell[dim] >= 0) {
            upper_side(element, var, cell[2], cell[1], cell[0]) =
                lower_and_upper_values[0];
          }
          if (cell[dim] < number_of_cells) {
            cell[dim] += 1;
            lower_side(element, var, cell[2], cell[1], cell[0]) =
                lower_and_upper_values[1];
          }
        });
  }
}
}  // namespace fd::device::detail
