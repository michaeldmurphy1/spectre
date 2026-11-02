// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
class DataVector;
template <typename T>
class Variables;
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief `PrimitiveGhostVariables` that can also compute the ghost variables
 * on the device.
 *
 * `evolution::dg::subcell::Actions::SendDataForReconstruction` uses
 * `apply_on_device` with Kokkos, and `apply` otherwise.
 */
class PrimitiveGhostVariablesOnDevice : public PrimitiveGhostVariables {
 public:
  /// Write the ghost variables into `ghost_variables` on the device, in the
  /// same component-major layout as `apply`. The primitive variables are
  /// copied from the host, since they are still stored there. This waits for
  /// `exec` to finish.
  static void apply_on_device(
      const Kokkos::DefaultExecutionSpace& exec,
      const Kokkos::View<double*>& ghost_variables,
      const Variables<hydro::grmhd_tags<DataVector>>& prims);
};
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
