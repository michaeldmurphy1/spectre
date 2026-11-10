// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

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
 * All elements of a patch are in one block and share its grid to inertial
 * map.
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
  /// maps, with the `grid_to_inertial_map` of their block.
  PatchElements(
      const Domain<3>& domain, const PatchTable<3>& patch_table, size_t patch,
      const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
      const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map);

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

  /// The map from the grid to the inertial frame of the block of the patch
  const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
  grid_to_inertial_map() const {
    return *grid_to_inertial_map_;
  }

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
  std::unique_ptr<::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
      grid_to_inertial_map_{};
};
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
