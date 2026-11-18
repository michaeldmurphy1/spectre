// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructorOnDeviceKokkos.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

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
#include "Utilities/ErrorHandling/Assert.hpp"
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

constexpr size_t rest_mass_density_index =
    tmpl::index_of<reconstructed_face_tags,
                   hydro::Tags::RestMassDensity<DataVector>>::value;
constexpr size_t temperature_index =
    tmpl::index_of<reconstructed_face_tags,
                   hydro::Tags::Temperature<DataVector>>::value;
static_assert(rest_mass_density_index == 0 and temperature_index == 2,
              "Update the variable indices of the reconstruction");
constexpr size_t number_of_reconstructed_variables =
    Variables<reconstructed_face_tags>::number_of_independent_components;

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

ReconstructorOnDevice::ReconstructorOnDevice(const Reconstructor& reconstructor)
    : ghost_zone_size_(reconstructor.ghost_zone_size()),
      reconstruct_rho_times_temperature_(
          reconstructor.reconstruct_rho_times_temperature()) {
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

ReconstructorOnDevice::ReconstructorOnDevice(
    const size_t ghost_zone_size, const bool adaptive_order,
    const ::fd::reconstruction::FallbackReconstructorType
        low_order_reconstructor,
    const double four_to_the_alpha_5,
    const std::optional<double>& six_to_the_alpha_7,
    const std::optional<double>& eight_to_the_alpha_9,
    const bool reconstruct_rho_times_temperature)
    : ghost_zone_size_(ghost_zone_size),
      adaptive_order_(adaptive_order),
      low_order_reconstructor_(low_order_reconstructor),
      four_to_the_alpha_5_(four_to_the_alpha_5),
      six_to_the_alpha_7_(six_to_the_alpha_7),
      eight_to_the_alpha_9_(eight_to_the_alpha_9),
      reconstruct_rho_times_temperature_(reconstruct_rho_times_temperature) {}

void ReconstructorOnDevice::reconstruct(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<::fd::device::StridedVolumeView, 3>&
        upper_side_of_face_vars,
    const std::array<::fd::device::StridedVolumeView, 3>&
        lower_side_of_face_vars,
    const ::fd::device::ConstVolumeView& padded_ghost_variables,
    const Index<3>& volume_extents) {
  ASSERT(padded_ghost_variables.extent(1) >= number_of_reconstructed_variables,
         "The padded ghost variables have "
             << padded_ghost_variables.extent(1)
             << " variables, but the reconstruction needs "
             << number_of_reconstructed_variables);
  const auto padded_range = [&padded_ghost_variables](const size_t begin,
                                                      const size_t end) {
    return ::fd::device::ConstStridedVolumeView{Kokkos::subview(
        padded_ghost_variables, Kokkos::ALL(), Kokkos::make_pair(begin, end),
        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL())};
  };
  // The ranges of variables that are reconstructed from the same padded data
  // with the same method: [begin, end), the padded data, and whether to
  // preserve positivity (with adaptive order)
  std::vector<
      std::tuple<size_t, size_t, ::fd::device::ConstStridedVolumeView, bool>>
      ranges{};
  if (reconstruct_rho_times_temperature_) {
    // As the host, which multiplies the temperature by the rest mass density
    // in the volume and the ghost zones
    if (rho_times_temperature_.extent(0) != padded_ghost_variables.extent(0) or
        rho_times_temperature_.extent(2) != padded_ghost_variables.extent(2) or
        rho_times_temperature_.extent(3) != padded_ghost_variables.extent(3) or
        rho_times_temperature_.extent(4) != padded_ghost_variables.extent(4)) {
      rho_times_temperature_ = ::fd::device::VolumeView{
          Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                             "rho_times_temperature"),
          padded_ghost_variables.extent(0),
          1,
          padded_ghost_variables.extent(2),
          padded_ghost_variables.extent(3),
          padded_ghost_variables.extent(4)};
    }
    const auto& padded = padded_ghost_variables;
    const auto& rho_times_temperature = rho_times_temperature_;
    // The edges and corners of the padded data aren't set (they are signaling
    // NaNs with `SPECTRE_NAN_INIT`) and aren't used, so skip them
    const size_t ghost_width =
        ::fd::device::detail::ghost_width(padded, volume_extents);
    const Kokkos::Array<size_t, 3> interior_end{
        ghost_width + volume_extents[2], ghost_width + volume_extents[1],
        ghost_width + volume_extents[0]};
    Kokkos::parallel_for(
        "grmhd::ValenciaDivClean::fd::ReconstructorOnDevice::"
        "rho_times_temperature",
        // `Iterate::Right`, so consecutive threads access consecutive `i` (the
        // default on GPUs iterates fastest over the elements)
        Kokkos::MDRangePolicy<
            Kokkos::DefaultExecutionSpace,
            Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
            Kokkos::IndexType<size_t>>(exec, {0, 0, 0, 0},
                                       {padded.extent(0), padded.extent(2),
                                        padded.extent(3), padded.extent(4)}),
        KOKKOS_LAMBDA(const size_t e, const size_t k, const size_t j,
                      const size_t i) {
          const size_t outside = (k < ghost_width or k >= interior_end[0]) +
                                 (j < ghost_width or j >= interior_end[1]) +
                                 (i < ghost_width or i >= interior_end[2]);
          if (outside <= 1) {
            rho_times_temperature(e, 0, k, j, i) =
                padded(e, temperature_index, k, j, i) *
                padded(e, rest_mass_density_index, k, j, i);
          }
        });
    ranges = {
        {0, temperature_index, padded_range(0, temperature_index), true},
        {temperature_index, temperature_index + 1,
         ::fd::device::ConstStridedVolumeView{rho_times_temperature_}, true},
        {temperature_index + 1, number_of_positivity_preserving_variables,
         padded_range(temperature_index + 1,
                      number_of_positivity_preserving_variables),
         true}};
  } else {
    ranges = {{0, number_of_positivity_preserving_variables,
               padded_range(0, number_of_positivity_preserving_variables),
               true}};
  }
  ranges.emplace_back(number_of_positivity_preserving_variables,
                      number_of_reconstructed_variables,
                      padded_range(number_of_positivity_preserving_variables,
                                   number_of_reconstructed_variables),
                      false);

  for (const auto& [begin, end, padded, positivity_preserving] : ranges) {
    if (begin == end) {
      continue;
    }
    if (adaptive_order_) {
      ::fd::device::positivity_preserving_adaptive_order(
          exec, variable_range(upper_side_of_face_vars, begin, end),
          variable_range(lower_side_of_face_vars, begin, end), padded,
          volume_extents, positivity_preserving, low_order_reconstructor_,
          four_to_the_alpha_5_, six_to_the_alpha_7_, eight_to_the_alpha_9_);
    } else {
      ::fd::device::monotonised_central(
          exec, variable_range(upper_side_of_face_vars, begin, end),
          variable_range(lower_side_of_face_vars, begin, end), padded,
          volume_extents);
    }
  }

  if (reconstruct_rho_times_temperature_) {
    // Divide by the reconstructed rest mass density, as the host does
    for (size_t d = 0; d < 3; ++d) {
      for (const auto& face_vars : {gsl::at(upper_side_of_face_vars, d),
                                    gsl::at(lower_side_of_face_vars, d)}) {
        Kokkos::parallel_for(
            "grmhd::ValenciaDivClean::fd::ReconstructorOnDevice::"
            "divide_by_rest_mass_density",
            Kokkos::MDRangePolicy<
                Kokkos::DefaultExecutionSpace,
                Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
                Kokkos::IndexType<size_t>>(
                exec, {0, 0, 0, 0},
                {face_vars.extent(0), face_vars.extent(2), face_vars.extent(3),
                 face_vars.extent(4)}),
            KOKKOS_LAMBDA(const size_t e, const size_t k, const size_t j,
                          const size_t i) {
              face_vars(e, temperature_index, k, j, i) /=
                  face_vars(e, rest_mass_density_index, k, j, i);
            });
      }
    }
  }
}
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
