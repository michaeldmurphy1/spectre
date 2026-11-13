// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "Evolution/DiscontinuousGalerkin/OptionTags.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd_patch::OptionTags {
/// \ingroup OptionGroupsGroup
/// Options for running the FD elements in patches on the device
struct FdPatchesGroup {
  static std::string name() { return "FdPatches"; }
  static constexpr Options::String help{
      "Options for running the FD elements in patches on the device"};
};

/// \ingroup OptionTagsGroup
/// The largest number of FD elements in a patch (see
/// `evolution::dg::subcell::fd_patch::FdPatches`)
struct MaxElementsPerPatch {
  using type = Options::Auto<size_t>;
  static constexpr Options::String help{
      "The largest number of FD elements in a patch. The FD elements with the "
      "same DG mesh, ordered by block and along the Z-curve of each block, "
      "are split into the fewest patches of about equal size. 'Auto' puts all "
      "FD elements with the same DG mesh in one patch."};
  using group = FdPatchesGroup;
};

/// \ingroup OptionTagsGroup
/// Whether the DG elements avoid the processors of the FD patches (see
/// `evolution::dg::subcell::fd_patch::FdPatches::dedicated_procs`)
struct DedicatedProcs {
  using type = bool;
  static constexpr Options::String help{
      "Whether the DG elements avoid the processors (PEs) of the FD patches, "
      "so that a patch never waits for DG elements on its processor."};
  using group = FdPatchesGroup;
};
}  // namespace evolution::dg::subcell::fd_patch::OptionTags

namespace evolution::dg::subcell::fd_patch::Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Which FD elements belong to which patch. See
/// `evolution::dg::subcell::fd_patch::FdPatches`.
///
/// Uses `Metavariables::SubcellOptions::subcell_enabled_at_external_boundary`.
template <size_t Dim>
struct FdPatches : db::SimpleTag {
  using type = fd_patch::FdPatches<Dim>;
  template <typename Metavariables>
  using option_tags =
      tmpl::list<::domain::OptionTags::DomainCreator<Dim>,
                 evolution::dg::OptionTags::OnlyDgBlocksAndGroups,
                 evolution::dg::OptionTags::Quadrature,
                 OptionTags::MaxElementsPerPatch, OptionTags::DedicatedProcs>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
      const std::optional<std::vector<std::string>>&
          only_dg_block_and_group_names,
      const Spectral::Quadrature i1_quadrature,
      const std::optional<size_t>& max_elements_per_patch,
      const bool dedicated_procs) {
    const auto domain = domain_creator->create_domain();
    return type{
        domain.blocks(),
        domain_creator->initial_refinement_levels(),
        domain_creator->initial_extents(),
        i1_quadrature,
        evolution::dg::compute_only_dg_block_ids(
            only_dg_block_and_group_names, domain_creator->block_names(),
            domain_creator->block_groups(), domain.blocks()),
        Metavariables::SubcellOptions::subcell_enabled_at_external_boundary,
        max_elements_per_patch,
        dedicated_procs};
  }
};
}  // namespace evolution::dg::subcell::fd_patch::Tags
