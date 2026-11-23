// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device::detail {
// `Reconstructor::pointwise(q, stride, args...)` with the `args` in a tuple,
// since device lambdas can't capture parameter packs
template <typename Reconstructor, typename... Args, size_t... Is>
KOKKOS_FUNCTION auto reconstruct_pointwise(
    const double* const q, const int stride, const std::tuple<Args...>& args,
    std::index_sequence<Is...> /*meta*/) {
  return Reconstructor::pointwise(q, stride, std::get<Is>(args)...);
}

/*!
 * \brief Reconstruct a batch of elements to the faces with a
 * `Reconstructor` from `fd::reconstruction::detail`, launching one Kokkos
 * kernel per direction on `exec`.
 *
 * Each thread reconstructs one cell, including the cells of the neighbors
 * that are adjacent to the element, so each face value is written once. The
 * padded volume data must have ghost zones of width at least
 * `(Reconstructor::stencil_width() + 1) / 2`: the faces on the element
 * boundary need the reconstruction in the neighbor's cell, whose stencil
 * extends one cell further than the stencils in the element.
 *
 * `Reconstructor::pointwise(q, stride, args...)` returns the values on the
 * lower and upper face of the cell as its first two elements (e.g. a
 * `std::array` or a `std::tuple`). The padded volume data can be a
 * `ConstVolumeView` or a `ConstStridedVolumeView`.
 */
template <typename Reconstructor, size_t Dim, typename PaddedView,
          typename... Args>
void reconstruct(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<StridedVolumeView, Dim>& upper_side_of_face_vars,
    const std::array<StridedVolumeView, Dim>& lower_side_of_face_vars,
    const PaddedView& padded_volume_vars, const Index<Dim>& volume_extents,
    const Args&... args) {
  const std::tuple<Args...> arguments{args...};
  constexpr size_t stencil_width = Reconstructor::stencil_width();
  static_assert(stencil_width % 2 == 1, "The stencil width must be odd");
  const size_t ghost_width =
      detail::ghost_width(padded_volume_vars, volume_extents);
  ASSERT(ghost_width >= (stencil_width + 1) / 2,
         "The ghost zones of width " << ghost_width
                                     << " are too small for a stencil of width "
                                     << stencil_width);

  // Extents and ghost widths in (x, y, z) order
  Kokkos::Array<int, 3> extents{1, 1, 1};
  Kokkos::Array<int, 3> offsets{0, 0, 0};
  for (size_t d = 0; d < Dim; ++d) {
    extents[d] = static_cast<int>(volume_extents[d]);
    offsets[d] = static_cast<int>(ghost_width);
  }
  const size_t number_of_elements = padded_volume_vars.extent(0);
  const size_t number_of_variables = padded_volume_vars.extent(1);

  for (size_t dim = 0; dim < Dim; ++dim) {
    const StridedVolumeView upper_side = gsl::at(upper_side_of_face_vars, dim);
    const StridedVolumeView lower_side = gsl::at(lower_side_of_face_vars, dim);
#ifdef SPECTRE_DEBUG
    for (const auto& face_vars : {upper_side, lower_side}) {
      ASSERT(face_vars.extent(0) == number_of_elements and
                 face_vars.extent(1) == number_of_variables,
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
    // One thread per cell, looping over the variables
    const BatchPoints points{number_of_elements, end};
    const size_t variable_stride = padded_volume_vars.stride(1);
    const size_t upper_stride = upper_side.stride(1);
    const size_t lower_stride = lower_side.stride(1);
    Kokkos::parallel_for(
        "fd::device::reconstruct",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                            Kokkos::IndexType<int>>(exec, 0, points.size()),
        KOKKOS_LAMBDA(const int index) {
          const auto [element, face] = points(index);
          // The cell below face `face` in direction `dim`, from -1 to
          // `number_of_cells`. Only constant indices into `face`: indexing it
          // with `dim` would keep it in local memory.
          const int x = face[0] - (dim == 0 ? 1 : 0);
          const int y = face[1] - (dim == 1 ? 1 : 0);
          const int z = face[2] - (dim == 2 ? 1 : 0);
          const int cell_in_dim = dim == 0 ? x : (dim == 1 ? y : z);
          const double* q = &padded_volume_vars(element, 0, z + offsets[2],
                                                y + offsets[1], x + offsets[0]);
          // The value on the lower face of the cell is on the upper side of
          // face `cell`, and the value on the upper face of the cell is on the
          // lower side of face `cell + 1`
          const bool has_lower_face = cell_in_dim >= 0;
          const bool has_upper_face = cell_in_dim < number_of_cells;
          double* upper_side_value =
              has_lower_face ? &upper_side(element, 0, z, y, x) : nullptr;
          double* lower_side_value =
              has_upper_face
                  ? &lower_side(element, 0, face[2], face[1], face[0])
                  : nullptr;
          for (size_t var = 0; var < number_of_variables; ++var) {
            const auto lower_and_upper_values =
                reconstruct_pointwise<Reconstructor>(
                    q, stride, arguments, std::index_sequence_for<Args...>{});
            if (has_lower_face) {
              upper_side_value[var * upper_stride] =
                  std::get<0>(lower_and_upper_values);
            }
            if (has_upper_face) {
              lower_side_value[var * lower_stride] =
                  std::get<1>(lower_and_upper_values);
            }
            q += variable_stride;
          }
        });
  }
}
}  // namespace fd::device::detail
#endif  // SPECTRE_KOKKOS
