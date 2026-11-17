// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/DeviceHistoryKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
/// \cond
template <size_t Dim>
class Domain;
class TimeStepper;
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
namespace evolution::patches {
template <size_t Dim>
class PatchTable;
}  // namespace evolution::patches
/// \endcond

namespace evolution::patches {
/*!
 * \brief The elements of a patch and their maps: the parts of a patch that
 * don't depend on the method or the system.
 *
 * \details Method layers derive from this class and add what their method
 * needs, e.g. `evolution::dg::subcell::fd_patch::FdPatchElements`. The
 * generic actions use these members through the system's patch (see
 * `PatchArray`).
 *
 * The grid to inertial map of an element is the moving mesh map of its block,
 * or the identity. The patch holds one map per block of its elements.
 * `update_device_maps()` evaluates them at a time for device kernels (see
 * `device_grid_to_inertial`), which read the map of each element from
 * `device_maps()`.
 */
class PatchElements {
 public:
  PatchElements() = default;
  PatchElements(const PatchElements&) = delete;
  PatchElements& operator=(const PatchElements&) = delete;
  PatchElements(PatchElements&&) = default;
  PatchElements& operator=(PatchElements&&) = default;
  ~PatchElements() = default;

  /// Set up the elements of the patch `patch` of `patch_table` and their
  /// maps. Allocates the device maps on `exec` (see `update_device_maps()`).
  PatchElements(
      const Kokkos::DefaultExecutionSpace& exec, const Domain<3>& domain,
      const PatchTable<3>& patch_table, size_t patch,
      const std::vector<std::array<size_t, 3>>& initial_refinement_levels);

  /// The index of the patch in the `PatchTable`
  size_t patch() const { return patch_; }

  /// @{
  /// The elements of the patch, in slot order, and their maps
  size_t number_of_elements() const { return element_ids_.size(); }
  const std::vector<ElementId<3>>& element_ids() const { return element_ids_; }
  const std::vector<Element<3>>& elements() const { return elements_; }
  const std::vector<ElementMap<3, Frame::Grid>>& element_maps() const {
    return element_maps_;
  }
  /// @}

  /// The DG mesh of the elements
  const Mesh<3>& dg_mesh() const { return dg_mesh_; }

  /// The map from the grid to the inertial frame of the element in `slot`
  const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
  grid_to_inertial_map(size_t slot) const;

  /// Whether the mesh of any element of the patch moves
  bool mesh_is_moving() const { return mesh_is_moving_; }

  /// @{
  /// The grid to inertial maps of the blocks of the elements, the map of the
  /// element in `slot`, and the grid coordinates of the corners of the
  /// elements with the `map` (where it must be affine if it has no device
  /// version, see `device_grid_to_inertial`)
  size_t number_of_maps() const { return grid_to_inertial_maps_.size(); }
  const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>& map(
      size_t map) const;
  size_t map_of_slot(size_t slot) const;
  const tnsr::I<DataVector, 3, Frame::Grid>& map_corners(size_t map) const;
  /// @}

  /*!
   * \brief Evaluate the maps at `time` for device kernels, and return whether
   * all of them are affine.
   *
   * \details The host evaluates the maps of the blocks into a pinned buffer,
   * and enqueues copying them to the device and gathering the map of each
   * element into `device_maps()` on `exec`, so the arguments of the kernels
   * don't change between substeps. Since the copy is only enqueued, the
   * previous call's copy must be done before the next call, e.g. because the
   * patch waited for the device in between (which it does once per substep).
   */
  bool update_device_maps(
      const Kokkos::DefaultExecutionSpace& exec, double time,
      const ::domain::FunctionsOfTimeMap& functions_of_time);

  /// The map of each element at the time of the last `update_device_maps()`
  Kokkos::View<const ::domain::CoordinateMaps::DeviceMap<3>*> device_maps()
      const {
    return device_maps_.of_elements;
  }

  /*!
   * \brief The map of each element at `time` for device kernels, in buffers
   * separate from `device_maps()`, e.g. for observations between substeps.
   *
   * \details As `update_device_maps()`, so the copy of the previous call must
   * be done before the next call.
   */
  Kokkos::View<const ::domain::CoordinateMaps::DeviceMap<3>*> device_maps_at(
      const Kokkos::DefaultExecutionSpace& exec, double time,
      const ::domain::FunctionsOfTimeMap& functions_of_time) const;

  /// A description of the element in `slot` for error messages
  std::string element_description(size_t slot) const;

  /// The time stepper history of the elements for `number_of_values` evolved
  /// values per element, empty and at the starting order of the
  /// `time_stepper`, as `Initialization::TimeStepperHistory` sets it up for an
  /// element
  TimeSteppers::DeviceHistory<> initial_history(const TimeStepper& time_stepper,
                                                size_t number_of_values) const;

 private:
  size_t patch_ = 0;
  Mesh<3> dg_mesh_{};
  std::vector<ElementId<3>> element_ids_{};
  std::vector<Element<3>> elements_{};
  std::vector<ElementMap<3, Frame::Grid>> element_maps_{};
  std::vector<std::unique_ptr<
      ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
      grid_to_inertial_maps_{};
  std::vector<size_t> map_of_slot_{};
  std::vector<tnsr::I<DataVector, 3, Frame::Grid>> map_corners_{};
  bool mesh_is_moving_ = false;
  // The maps for device kernels at a time: one per block (with its host
  // buffer), and the map of each element, gathered on the device
  struct DeviceMaps {
    DeviceMaps() = default;
    DeviceMaps(const Kokkos::DefaultExecutionSpace& exec, size_t number_of_maps,
               size_t number_of_elements);

    Kokkos::View<::domain::CoordinateMaps::DeviceMap<3>*> table{};
    Kokkos::View<::domain::CoordinateMaps::DeviceMap<3>*,
                 Kokkos::SharedHostPinnedSpace>
        host_table{};
    Kokkos::View<::domain::CoordinateMaps::DeviceMap<3>*> of_elements{};
  };
  // Evaluate the maps at `time` into `device_maps` (whose views can be written
  // through const handles, so it can be called on const members), and return
  // whether all of them are affine
  bool fill_device_maps(
      const Kokkos::DefaultExecutionSpace& exec, const DeviceMaps& device_maps,
      double time, const ::domain::FunctionsOfTimeMap& functions_of_time) const;

  Kokkos::View<size_t*> device_map_of_slot_{};
  // For the kernels of the substeps, and for observations between them
  DeviceMaps device_maps_{};
  DeviceMaps observation_device_maps_{};
};
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
