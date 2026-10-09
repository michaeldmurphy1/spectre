// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/Device/MonotonisedCentral.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/Device/Reconstruct.hpp"
#include "NumericalAlgorithms/FiniteDifference/Device/VolumeView.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace fd::device {
template <size_t Dim>
void monotonised_central(
    const std::array<VolumeView, Dim>& upper_side_of_face_vars,
    const std::array<VolumeView, Dim>& lower_side_of_face_vars,
    const ConstVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents) {
  detail::reconstruct<
      ::fd::reconstruction::detail::MonotonisedCentralReconstructor>(
      upper_side_of_face_vars, lower_side_of_face_vars, padded_volume_vars,
      volume_extents);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                          \
  template void monotonised_central(                                    \
      const std::array<VolumeView, DIM(data)>& upper_side_of_face_vars, \
      const std::array<VolumeView, DIM(data)>& lower_side_of_face_vars, \
      const ConstVolumeView& padded_volume_vars,                        \
      const Index<DIM(data)>& volume_extents);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
