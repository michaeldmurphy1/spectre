// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrderKokkos.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/Minmod.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/ReconstructKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device {
namespace {
template <typename LowOrderReconstructor, bool PositivityPreserving,
          bool Use9thOrder, bool Use7thOrder, size_t Dim>
void reconstruct(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<StridedVolumeView, Dim>& upper_side_of_face_vars,
    const std::array<StridedVolumeView, Dim>& lower_side_of_face_vars,
    const ConstStridedVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents, const double four_to_the_alpha_5,
    const double six_to_the_alpha_7, const double eight_to_the_alpha_9) {
  detail::reconstruct<::fd::reconstruction::detail::
                          PositivityPreservingAdaptiveOrderReconstructor<
                              LowOrderReconstructor, PositivityPreserving,
                              Use9thOrder, Use7thOrder>>(
      exec, upper_side_of_face_vars, lower_side_of_face_vars,
      padded_volume_vars, volume_extents, four_to_the_alpha_5,
      six_to_the_alpha_7, eight_to_the_alpha_9);
}

template <bool PositivityPreserving, bool Use9thOrder, bool Use7thOrder,
          size_t Dim, typename... Args>
void select_low_order(const ::fd::reconstruction::FallbackReconstructorType
                          low_order_reconstructor,
                      const Args&... args) {
  switch (low_order_reconstructor) {
    case ::fd::reconstruction::FallbackReconstructorType::Minmod:
      reconstruct<::fd::reconstruction::detail::MinmodReconstructor,
                  PositivityPreserving, Use9thOrder, Use7thOrder, Dim>(args...);
      return;
    case ::fd::reconstruction::FallbackReconstructorType::MonotonisedCentral:
      reconstruct<::fd::reconstruction::detail::MonotonisedCentralReconstructor,
                  PositivityPreserving, Use9thOrder, Use7thOrder, Dim>(args...);
      return;
    default:
      ERROR(
          "The low-order reconstructor of the positivity-preserving "
          "adaptive-order reconstruction must be Minmod or "
          "MonotonisedCentral, not "
          << low_order_reconstructor);
  }
}

template <bool Use9thOrder, bool Use7thOrder, size_t Dim, typename... Args>
void select_positivity(const bool positivity_preserving,
                       const ::fd::reconstruction::FallbackReconstructorType
                           low_order_reconstructor,
                       const Args&... args) {
  if (positivity_preserving) {
    select_low_order<true, Use9thOrder, Use7thOrder, Dim>(
        low_order_reconstructor, args...);
  } else {
    select_low_order<false, Use9thOrder, Use7thOrder, Dim>(
        low_order_reconstructor, args...);
  }
}
}  // namespace

template <size_t Dim>
void positivity_preserving_adaptive_order(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<StridedVolumeView, Dim>& upper_side_of_face_vars,
    const std::array<StridedVolumeView, Dim>& lower_side_of_face_vars,
    const ConstStridedVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents, const bool positivity_preserving,
    const ::fd::reconstruction::FallbackReconstructorType
        low_order_reconstructor,
    const double four_to_the_alpha_5,
    const std::optional<double>& six_to_the_alpha_7,
    const std::optional<double>& eight_to_the_alpha_9) {
  // As `fd::PositivityPreservingAdaptiveOrderPrim` passes them
  const double alpha_7_value =
      six_to_the_alpha_7.value_or(std::numeric_limits<double>::signaling_NaN());
  const double alpha_9_value = eight_to_the_alpha_9.value_or(
      std::numeric_limits<double>::signaling_NaN());
  const auto select = [&]<bool Use9thOrder, bool Use7thOrder>() {
    select_positivity<Use9thOrder, Use7thOrder, Dim>(
        positivity_preserving, low_order_reconstructor, exec,
        upper_side_of_face_vars, lower_side_of_face_vars, padded_volume_vars,
        volume_extents, four_to_the_alpha_5, alpha_7_value, alpha_9_value);
  };
  if (six_to_the_alpha_7.has_value()) {
    if (eight_to_the_alpha_9.has_value()) {
      select.template operator()<true, true>();
    } else {
      select.template operator()<false, true>();
    }
  } else {
    if (eight_to_the_alpha_9.has_value()) {
      select.template operator()<true, false>();
    } else {
      select.template operator()<false, false>();
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template void positivity_preserving_adaptive_order(                          \
      const Kokkos::DefaultExecutionSpace& exec,                               \
      const std::array<StridedVolumeView, DIM(data)>& upper_side_of_face_vars, \
      const std::array<StridedVolumeView, DIM(data)>& lower_side_of_face_vars, \
      const ConstStridedVolumeView& padded_volume_vars,                        \
      const Index<DIM(data)>& volume_extents, bool positivity_preserving,      \
      ::fd::reconstruction::FallbackReconstructorType low_order_reconstructor, \
      double four_to_the_alpha_5,                                              \
      const std::optional<double>& six_to_the_alpha_7,                         \
      const std::optional<double>& eight_to_the_alpha_9);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
