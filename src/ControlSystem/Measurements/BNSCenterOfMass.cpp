// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Measurements/BNSCenterOfMass.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace control_system::measurements {
std::array<std::array<double, 3>, 3> inertial_centers(
    const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, const double time,
    const std::array<std::array<double, 3>, 3>& grid_centers) {
  tnsr::I<DataVector, 3, Frame::Grid> grid_tnsr_center{};
  for (size_t i = 0; i < 3; i++) {
    grid_tnsr_center.get(i) =
        DataVector{gsl::at(grid_centers[0], i), gsl::at(grid_centers[1], i),
                   gsl::at(grid_centers[2], i)};
  }

  const auto block_logical_coords = block_logical_coordinates(
      domain, grid_tnsr_center, time, functions_of_time);

  ASSERT(alg::all_of(block_logical_coords,
                     [](const auto& logical_coord_holder) {
                       return logical_coord_holder.has_value();
                     }),
         "Grid centers of BNS (" << grid_centers[0] << ", " << grid_centers[1]
                                 << ", " << grid_centers[2]
                                 << ") could not be mapped to the logical "
                                    "frame.");

  const auto& blocks = domain.blocks();

  ASSERT(block_logical_coords.size() == 3,
         "There should be exactly 3 block logical coordinates for the two "
         "centers of the BNS plus the system COM, but instead there are "
             << block_logical_coords.size());

  std::array<std::array<double, 3>, 3> result{};
  for (size_t n = 0; n < block_logical_coords.size(); n++) {
    const auto& logical_coord_holder = block_logical_coords[n];
    const auto& block_id = logical_coord_holder.value().id;

    const auto& block = blocks[block_id.get_index()];
    const auto& grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map();

    const auto inertial_center = grid_to_inertial_map(
        tnsr::I<double, 3, Frame::Grid>{gsl::at(grid_centers, n)}, time,
        functions_of_time);

    for (size_t i = 0; i < 3; i++) {
      gsl::at(gsl::at(result, n), i) = inertial_center.get(i);
    }
  }
  return result;
}
}  // namespace control_system::measurements
