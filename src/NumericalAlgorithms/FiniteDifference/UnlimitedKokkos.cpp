// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/UnlimitedKokkos.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/ReconstructKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device {
template <size_t Degree, size_t Dim>
void unlimited(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<StridedVolumeView, Dim>& upper_side_of_face_vars,
    const std::array<StridedVolumeView, Dim>& lower_side_of_face_vars,
    const ConstStridedVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents) {
  detail::reconstruct<
      ::fd::reconstruction::detail::UnlimitedReconstructor<Degree>>(
      exec, upper_side_of_face_vars, lower_side_of_face_vars,
      padded_volume_vars, volume_extents);
}

#define DEGREE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                                 \
  template void unlimited<DEGREE(data), DIM(data)>(                            \
      const Kokkos::DefaultExecutionSpace& exec,                               \
      const std::array<StridedVolumeView, DIM(data)>& upper_side_of_face_vars, \
      const std::array<StridedVolumeView, DIM(data)>& lower_side_of_face_vars, \
      const ConstStridedVolumeView& padded_volume_vars,                        \
      const Index<DIM(data)>& volume_extents);

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 4, 6, 8), (1, 2, 3))

#undef INSTANTIATION
#undef DIM
#undef DEGREE
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
