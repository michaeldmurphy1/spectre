// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace Kokkos_detail {
template <typename View, typename... Extents>
bool has_extents(const View& view, const Extents... extents) {
  size_t i = 0;
  return view.is_allocated() and
         ((view.extent(i++) == static_cast<size_t>(extents)) and ...);
}
}  // namespace Kokkos_detail

/*!
 * \brief A buffer `View` with the `extents` for a function that is called
 * repeatedly, e.g. in every time step.
 *
 * \details If `scratch` is not `nullptr`, it is (re)allocated only if its
 * extents differ and then returned, so a caller that keeps it between calls
 * doesn't allocate (and free, which synchronizes the device) every time. If
 * it is `nullptr`, a new view is allocated. The values are uninitialized.
 */
template <typename View, typename ExecutionSpace, typename... Extents>
View scratch_view(const ExecutionSpace& exec, View* const scratch,
                  const std::string& label, const Extents... extents) {
  const auto allocate = [&]() {
    return View(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label),
                extents...);
  };
  if (scratch == nullptr) {
    return allocate();
  }
  if (not Kokkos_detail::has_extents(*scratch, extents...)) {
    *scratch = allocate();
  }
  return *scratch;
}
#endif  // SPECTRE_KOKKOS
