// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches::OptionTags {
/// \ingroup OptionGroupsGroup
/// Options for running elements in patches on the device
struct PatchesGroup {
  static std::string name() { return "Patches"; }
  static constexpr Options::String help{
      "Options for running elements in patches on the device"};
};

/// \ingroup OptionTagsGroup
/// The largest number of elements in a patch (see
/// `evolution::patches::PatchTable`)
struct MaxElementsPerPatch {
  using type = Options::Auto<size_t>;
  static constexpr Options::String help{
      "The largest number of elements in a patch. The elements in patches "
      "with the same DG mesh, ordered by block and along the Z-curve of each "
      "block, are split into the fewest patches of about equal size. 'Auto' "
      "puts all of them in one patch."};
  using group = PatchesGroup;
};

/// \ingroup OptionTagsGroup
/// Whether patches compute the observations they support on the device (see
/// `evolution::patches::Tags::ObserveOnDevice`)
struct ObserveOnDevice {
  using type = bool;
  static constexpr Options::String help{
      "Whether patches compute the observations they support (e.g. the norms "
      "of ObserveNorms for the tensors the patch can compute on the device) "
      "on the device, copying only their results to the host. Otherwise all "
      "observations copy the variables of the elements to the host and "
      "evaluate the events on DataBoxes of the elements there."};
  using group = PatchesGroup;
};

/// \ingroup OptionTagsGroup
/// Whether the elements outside patches avoid the processors of the patches
/// (see `evolution::patches::PatchTable::dedicated_procs`)
struct DedicatedProcs {
  using type = bool;
  static constexpr Options::String help{
      "Whether the elements outside patches avoid the processors (PEs) of the "
      "patches, so that a patch never waits for other elements on its "
      "processor."};
  using group = PatchesGroup;
};
}  // namespace evolution::patches::OptionTags

namespace evolution::patches::Tags {
/// Whether patches compute the observations they support on the device (see
/// `OptionTags::ObserveOnDevice`)
struct ObserveOnDevice : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::ObserveOnDevice>;
  static constexpr bool pass_metavariables = false;
  static bool create_from_options(const bool observe_on_device) {
    return observe_on_device;
  }
};
}  // namespace evolution::patches::Tags
