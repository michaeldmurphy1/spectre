// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructorOnDeviceKokkos.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Index.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentralKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrderKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd {
namespace {
// The variables that `PositivityPreservingAdaptiveOrderPrim` reconstructs
// with the positivity-preserving variant, which come first in the
// `reconstructed_face_tags`
using positivity_preserving_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::Temperature<DataVector>>;
static_assert(
    std::is_same_v<
        tmpl::front<tmpl::split_at<reconstructed_face_tags,
                                   tmpl::size<positivity_preserving_tags>>>,
        positivity_preserving_tags>,
    "The positivity-preserving variables must be the first reconstructed "
    "variables.");
constexpr size_t number_of_positivity_preserving_variables =
    Variables<positivity_preserving_tags>::number_of_independent_components;

// The variables in [`begin`, `end`) of the `views`
template <typename View>
std::array<View, 3> variable_range(const std::array<View, 3>& views,
                                   const size_t begin, const size_t end) {
  std::array<View, 3> result{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(result, d) = Kokkos::subview(
        gsl::at(views, d), Kokkos::ALL(), Kokkos::make_pair(begin, end),
        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }
  return result;
}
}  // namespace

ReconstructorOnDevice::ReconstructorOnDevice(
    const Reconstructor& reconstructor) {
  if (reconstructor.reconstruct_rho_times_temperature()) {
    ERROR(
        "FD patches don't support reconstructing the rest mass density times "
        "the temperature (ReconstructRhoTimesTemperature).");
  }
  ghost_zone_size_ = reconstructor.ghost_zone_size();
  if (dynamic_cast<const MonotonisedCentralPrim*>(&reconstructor) != nullptr) {
    adaptive_order_ = false;
  } else if (const auto* const adaptive_order =
                 dynamic_cast<const PositivityPreservingAdaptiveOrderPrim*>(
                     &reconstructor);
             adaptive_order != nullptr) {
    adaptive_order_ = true;
    low_order_reconstructor_ = adaptive_order->low_order_reconstructor();
    four_to_the_alpha_5_ = adaptive_order->four_to_the_alpha_5();
    six_to_the_alpha_7_ = adaptive_order->six_to_the_alpha_7();
    eight_to_the_alpha_9_ = adaptive_order->eight_to_the_alpha_9();
  } else {
    ERROR(
        "FD patches only support the MonotonisedCentralPrim and "
        "PositivityPreservingAdaptiveOrderPrim reconstructors.");
  }
}

void ReconstructorOnDevice::reconstruct(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<::fd::device::StridedVolumeView, 3>&
        upper_side_of_face_vars,
    const std::array<::fd::device::StridedVolumeView, 3>&
        lower_side_of_face_vars,
    const ::fd::device::ConstVolumeView& padded_ghost_variables,
    const Index<3>& volume_extents) const {
  if (not adaptive_order_) {
    ::fd::device::monotonised_central(exec, upper_side_of_face_vars,
                                      lower_side_of_face_vars,
                                      padded_ghost_variables, volume_extents);
    return;
  }
  const size_t number_of_variables = padded_ghost_variables.extent(1);
  for (const auto& [begin, end, positivity_preserving] :
       {std::tuple{size_t{0}, number_of_positivity_preserving_variables, true},
        std::tuple{number_of_positivity_preserving_variables,
                   number_of_variables, false}}) {
    const ::fd::device::ConstStridedVolumeView padded = Kokkos::subview(
        padded_ghost_variables, Kokkos::ALL(), Kokkos::make_pair(begin, end),
        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    ::fd::device::positivity_preserving_adaptive_order(
        exec, variable_range(upper_side_of_face_vars, begin, end),
        variable_range(lower_side_of_face_vars, begin, end), padded,
        volume_extents, positivity_preserving, low_order_reconstructor_,
        four_to_the_alpha_5_, six_to_the_alpha_7_, eight_to_the_alpha_9_);
  }
}
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
