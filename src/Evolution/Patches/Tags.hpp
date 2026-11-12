// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "Options/Auto.hpp"
#include "Options/String.hpp"

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
      "The largest number of elements in a patch. The elements in patches of "
      "a block are split along a Z-curve into the fewest patches of about "
      "equal size. 'Auto' puts all of them in one patch."};
  using group = PatchesGroup;
};
}  // namespace evolution::patches::OptionTags
