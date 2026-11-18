// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
template <size_t Dim>
class Index;
namespace grmhd::ValenciaDivClean::fd {
class Reconstructor;
}  // namespace grmhd::ValenciaDivClean::fd
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief The reconstruction of the primitive variables of a batch of elements
 * on the device, with the same method and options as a `Reconstructor` on the
 * host.
 *
 * \details Supported are `MonotonisedCentralPrim` and
 * `PositivityPreservingAdaptiveOrderPrim`, also with
 * `ReconstructRhoTimesTemperature`; other reconstructors are an error. The
 * reconstructors of other systems with the same methods (e.g.
 * GhValenciaDivClean) use the constructor that takes the options. As
 * `PositivityPreservingAdaptiveOrderPrim` does, the rest mass density, the
 * electron fraction and the temperature are reconstructed with the
 * positivity-preserving variant and the other variables without it. With
 * `ReconstructRhoTimesTemperature`, \f$\rho T\f$ is reconstructed instead of
 * \f$T\f$ and divided by the reconstructed \f$\rho\f$ on the faces, as on
 * the host.
 */
class ReconstructorOnDevice {
 public:
  ReconstructorOnDevice() = default;
  explicit ReconstructorOnDevice(const Reconstructor& reconstructor);

  /*!
   * \brief The reconstruction with the given options.
   *
   * \param ghost_zone_size the width of the ghost zones that the
   * reconstruction needs.
   * \param adaptive_order whether to use positivity-preserving adaptive-order
   * reconstruction, otherwise monotonised central.
   * \param low_order_reconstructor the fallback of the adaptive-order
   * reconstruction.
   * \param four_to_the_alpha_5 the value of \f$4^{\alpha_5}\f$ of the
   * adaptive-order reconstruction.
   * \param six_to_the_alpha_7 the value of \f$6^{\alpha_7}\f$ if the 7th
   * order is used.
   * \param eight_to_the_alpha_9 the value of \f$8^{\alpha_9}\f$ if the 9th
   * order is used.
   * \param reconstruct_rho_times_temperature whether to reconstruct
   * \f$\rho T\f$ instead of \f$T\f$.
   */
  ReconstructorOnDevice(
      size_t ghost_zone_size, bool adaptive_order,
      ::fd::reconstruction::FallbackReconstructorType low_order_reconstructor,
      double four_to_the_alpha_5,
      const std::optional<double>& six_to_the_alpha_7,
      const std::optional<double>& eight_to_the_alpha_9,
      bool reconstruct_rho_times_temperature);

  /// The width of the ghost zones that the reconstruction needs
  size_t ghost_zone_size() const { return ghost_zone_size_; }

  /*!
   * \brief Reconstruct the ghost variables (the `reconstructed_face_tags`, in
   * that order) of a batch of elements on `exec`.
   *
   * \details With `ReconstructRhoTimesTemperature`, \f$\rho T\f$ is stored
   * in a buffer that is allocated on the first call, which is why this isn't
   * `const`.
   *
   * \param exec the execution space instance that runs the kernels.
   * \param upper_side_of_face_vars the values on the upper side of the faces
   * in each direction, e.g. from `reconstructed_face_views`.
   * \param lower_side_of_face_vars the values on the lower side of the faces.
   * \param padded_ghost_variables the ghost variables padded with ghost zones
   * of width at least `ghost_zone_size()`. Variables after the
   * `reconstructed_face_tags` (e.g. the spacetime variables of
   * GhValenciaDivClean) are ignored.
   * \param volume_extents the number of cells in each element.
   */
  void reconstruct(const Kokkos::DefaultExecutionSpace& exec,
                   const std::array<::fd::device::StridedVolumeView, 3>&
                       upper_side_of_face_vars,
                   const std::array<::fd::device::StridedVolumeView, 3>&
                       lower_side_of_face_vars,
                   const ::fd::device::ConstVolumeView& padded_ghost_variables,
                   const Index<3>& volume_extents);

 private:
  size_t ghost_zone_size_ = 0;
  // Monotonised central reconstruction if false
  bool adaptive_order_ = false;
  ::fd::reconstruction::FallbackReconstructorType low_order_reconstructor_ =
      ::fd::reconstruction::FallbackReconstructorType::None;
  double four_to_the_alpha_5_ = std::numeric_limits<double>::signaling_NaN();
  std::optional<double> six_to_the_alpha_7_{};
  std::optional<double> eight_to_the_alpha_9_{};
  bool reconstruct_rho_times_temperature_ = false;
  // The padded rho T, if it is reconstructed
  ::fd::device::VolumeView rho_times_temperature_{};
};
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
