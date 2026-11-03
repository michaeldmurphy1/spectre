// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "Framework/TestHelpers.hpp"
#include "Parallel/ArrayCollection/ReceiveDataForElement.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
// Inbox tags have no `pup` function.
struct InboxTag {};

SPECTRE_TEST_CASE("Unit.Parallel.ArrayCollection.ReceiveDataForElement",
                  "[Unit][Parallel]") {
  using Wrapper = Parallel::Actions::ReceiveTagWrapper<InboxTag>;
  static_assert(std::is_same_v<typename Wrapper::type, InboxTag>);
  // Messages to elements on other nodes carry the wrapper, so it must be
  // serializable. It carries no data.
  CHECK(serialize(Wrapper{}).empty());
  [[maybe_unused]] const Wrapper deserialized =
      serialize_and_deserialize(Wrapper{});
}
}  // namespace
