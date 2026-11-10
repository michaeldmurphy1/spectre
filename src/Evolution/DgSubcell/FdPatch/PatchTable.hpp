// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "Evolution/DiscontinuousGalerkin/OptionTags.hpp"
#include "Evolution/Patches/PatchTable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Block;
namespace Spectral {
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

/// Running FD elements in patches on the device (see `evolution::patches`)
namespace evolution::dg::subcell::fd_patch {
/*!
 * \brief The `evolution::patches::PatchTable` of the FD elements: the
 * elements that use the subcell (FD) grid for the whole evolution, i.e. for
 * which `evolution::dg::subcell::starts_on_subcell_grid()` is true (this
 * assumes `AlwaysUseSubcells`).
 *
 * \details The DG meshes of the elements are Legendre meshes with
 * `i1_quadrature`, as `domain::create_initial_mesh()` creates them. The
 * elements of the `only_dg_block_ids` are never FD elements.
 */
template <size_t Dim>
evolution::patches::PatchTable<Dim> patch_table(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    Spectral::Quadrature i1_quadrature,
    const std::vector<size_t>& only_dg_block_ids,
    bool subcell_enabled_at_external_boundary);

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Which FD elements belong to which patch. See `patch_table()`.
///
/// Uses `Metavariables::SubcellOptions::subcell_enabled_at_external_boundary`.
template <size_t Dim>
struct PatchTable : db::SimpleTag {
  using type = evolution::patches::PatchTable<Dim>;
  template <typename Metavariables>
  using option_tags =
      tmpl::list<::domain::OptionTags::DomainCreator<Dim>,
                 evolution::dg::OptionTags::OnlyDgBlocksAndGroups,
                 evolution::dg::OptionTags::Quadrature>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
      const std::optional<std::vector<std::string>>&
          only_dg_block_and_group_names,
      const Spectral::Quadrature i1_quadrature) {
    const auto domain = domain_creator->create_domain();
    return patch_table(
        domain.blocks(), domain_creator->initial_refinement_levels(),
        domain_creator->initial_extents(), i1_quadrature,
        evolution::dg::compute_only_dg_block_ids(
            only_dg_block_and_group_names, domain_creator->block_names(),
            domain_creator->block_groups(), domain.blocks()),
        Metavariables::SubcellOptions::subcell_enabled_at_external_boundary);
  }
};
}  // namespace Tags
}  // namespace evolution::dg::subcell::fd_patch
