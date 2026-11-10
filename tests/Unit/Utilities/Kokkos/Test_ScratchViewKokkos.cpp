// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Kokkos/ScratchViewKokkos.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.ScratchView", "[Unit][Utilities]") {
  const Kokkos::DefaultExecutionSpace exec{};
  // Without a scratch view a new view is allocated every time
  const auto first =
      scratch_view<Kokkos::View<double**>>(exec, nullptr, "Scratch", 3, 4);
  const auto second =
      scratch_view<Kokkos::View<double**>>(exec, nullptr, "Scratch", 3, 4);
  CHECK(first.extent(0) == 3);
  CHECK(first.extent(1) == 4);
  CHECK(first.data() != second.data());

  // A scratch view is allocated once and reused for the same extents
  Kokkos::View<double**> scratch{};
  CHECK(not scratch.is_allocated());
  const auto allocated = scratch_view(exec, &scratch, "Scratch", 3, 4);
  CHECK(scratch.is_allocated());
  CHECK(allocated.data() == scratch.data());
  CHECK(scratch_view(exec, &scratch, "Scratch", 3, 4).data() ==
        allocated.data());
  // and reallocated for other extents
  const auto reallocated = scratch_view(exec, &scratch, "Scratch", 5, 4);
  CHECK(reallocated.extent(0) == 5);
  CHECK(reallocated.extent(1) == 4);
  CHECK(scratch.data() == reallocated.data());
  CHECK(scratch_view(exec, &scratch, "Scratch", 5, 2).extent(1) == 2);
}
