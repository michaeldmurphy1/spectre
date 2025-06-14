// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.OptionTags",
                  "[Unit][Evolution]") {
  CHECK(TestHelpers::test_option_tag<ScalarWave::OptionTags::MassSq>("1.5") ==
        1.5);
}
