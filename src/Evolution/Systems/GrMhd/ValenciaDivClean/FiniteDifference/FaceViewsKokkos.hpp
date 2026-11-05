// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

/// \cond
template <size_t Dim>
class Index;
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief Views of the `reconstructed_face_tags` on one side of the faces in
 * each direction, pointing into the batch `faces`, so the device reconstruction
 * (e.g. `::fd::device::monotonised_central`) writes them without a copy.
 *
 * The `faces` hold one face set for each direction of each of
 * \f$N\f$ elements with `volume_extents` cells, ordered by direction:
 * entry \f$d N + e\f$ holds the faces in direction \f$d\f$ of element \f$e\f$.
 * Each face set has `volume_extents` with one more face in its direction, so
 * the extents must be isotropic for all face sets to have the same number of
 * points.
 */
std::array<::fd::device::StridedVolumeView, 3> reconstructed_face_views(
    const BatchedVariables<face_tags>& faces, const Index<3>& volume_extents);
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
