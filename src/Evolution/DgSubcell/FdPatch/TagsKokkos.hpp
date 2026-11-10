// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace evolution::dg::subcell::fd_patch::Tags {
/*!
 * \brief The variables of `VariablesTag` (a `::Tags::Variables`) at the cell
 * centers of all elements of an FD patch, batched on the device.
 *
 * \details Patches keep the state of their elements, e.g. the evolved
 * variables, in their DataBox, so that it is written to checkpoints (see
 * `BatchedVariables::pup`).
 */
template <typename VariablesTag>
struct Batched : db::SimpleTag {
  using type = BatchedVariables<typename VariablesTag::tags_list>;
};

/*!
 * \brief The variables of `VariablesTag` (a `::Tags::Variables`) on the faces
 * of all elements of an FD patch, batched on the device.
 *
 * \details The batch holds \f$3N\f$ face sets for \f$N\f$ elements, ordered by
 * direction and then element: face set \f$dN + e\f$ holds the faces normal
 * to direction \f$d\f$ of the element \f$e\f$.
 */
template <typename VariablesTag>
struct BatchedOnFaces : db::SimpleTag {
  using type = BatchedVariables<typename VariablesTag::tags_list>;
};
}  // namespace evolution::dg::subcell::fd_patch::Tags
#endif  // SPECTRE_KOKKOS
