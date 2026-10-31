// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
namespace helpers = TestHelpers::fd::device;

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  constexpr size_t number_of_elements = 2;
  constexpr size_t number_of_variables = 2;
  constexpr size_t ghost_width = 2;
  // Anisotropic to check the order of the dimensions
  Index<Dim> volume_extents{};
  for (size_t d = 0; d < Dim; ++d) {
    volume_extents[d] = 3 + d;
  }

  const Kokkos::DefaultExecutionSpace exec{};
  const auto unpadded =
      ::fd::device::make_volume_view(exec, "unpadded", number_of_elements,
                                     number_of_variables, volume_extents);
  const auto padded = ::fd::device::make_volume_view(
      exec, "padded", number_of_elements, number_of_variables, volume_extents,
      ghost_width);
  CHECK(padded.extent(0) == number_of_elements);
  CHECK(padded.extent(1) == number_of_variables);
  for (size_t d = 0; d < 3; ++d) {
    const size_t expected_extent = d < Dim ? volume_extents[d] : 1;
    CHECK(unpadded.extent(4 - d) == expected_extent);
    CHECK(padded.extent(4 - d) ==
          expected_extent + (d < Dim ? 2 * ghost_width : 0));
  }

  // A unique value at each cell. The neighbors send one more ghost cell than
  // needed, so the ghost zones must hold the ones closest to the element.
  const auto f = [](const size_t element, const size_t var,
                    const std::array<int, Dim>& cell) {
    constexpr std::array<double, 3> scales{{0.1, 0.01, 0.001}};
    auto result = static_cast<double>(1 + 100 * element + 10 * var);
    for (size_t d = 0; d < Dim; ++d) {
      result += (gsl::at(cell, d) + 5) * gsl::at(scales, d);
    }
    return result;
  };
  const auto data =
      helpers::make_element_data(number_of_elements, number_of_variables,
                                 volume_extents, ghost_width + 1, f);
  CHECK(::fd::device::detail::ghost_width(padded, volume_extents) ==
        ghost_width);
  CHECK(::fd::device::detail::ghost_width(unpadded, volume_extents) == 0);
  const auto host_padded =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, padded);
  for (size_t element = 0; element < number_of_elements; ++element) {
    const auto& volume_vars = data[element].volume_vars;
    ::fd::device::pack_with_ghost_zones(
        host_padded, element,
        gsl::make_span(volume_vars.data(), volume_vars.size()),
        helpers::make_spans(data[element].ghost_data), volume_extents);
  }

  for (size_t element = 0; element < number_of_elements; ++element) {
    for (size_t var = 0; var < number_of_variables; ++var) {
      for (size_t z = 0; z < padded.extent(2); ++z) {
        for (size_t y = 0; y < padded.extent(3); ++y) {
          for (size_t x = 0; x < padded.extent(4); ++x) {
            const std::array<size_t, 3> padded_index{{x, y, z}};
            std::array<int, Dim> cell{};
            size_t dims_outside_element = 0;
            for (size_t d = 0; d < Dim; ++d) {
              gsl::at(cell, d) = static_cast<int>(gsl::at(padded_index, d)) -
                                 static_cast<int>(ghost_width);
              if (gsl::at(cell, d) < 0 or
                  gsl::at(cell, d) >= static_cast<int>(volume_extents[d])) {
                ++dims_outside_element;
              }
            }
            CAPTURE(element);
            CAPTURE(var);
            CAPTURE(padded_index);
            // The interior holds the volume data and the ghost zones hold the
            // ghost data. The edges and corners aren't filled, so they keep
            // the signaling NaNs the view was allocated with.
            if (dims_outside_element <= 1) {
              CHECK(host_padded(element, var, z, y, x) ==
                    f(element, var, cell));
            } else {
#ifdef SPECTRE_NAN_INIT
              CHECK(std::isnan(host_padded(element, var, z, y, x)));
#endif  // SPECTRE_NAN_INIT
            }
          }
        }
      }
    }
  }
}

// The flat index enumerates the cells with `x` varying fastest
void test_batch_cells() {
  const ::fd::device::detail::BatchCells cells{2, 3, {4, 5, 6}};
  CHECK(cells.size() == 2 * 3 * 4 * 5 * 6);
  int index = 0;
  for (int element = 0; element < 2; ++element) {
    for (int var = 0; var < 3; ++var) {
      for (int z = 0; z < 6; ++z) {
        for (int y = 0; y < 5; ++y) {
          for (int x = 0; x < 4; ++x, ++index) {
            const auto cell = cells(index);
            CHECK(cell.element == element);
            CHECK(cell.variable == var);
            CHECK(cell.index[0] == x);
            CHECK(cell.index[1] == y);
            CHECK(cell.index[2] == z);
          }
        }
      }
    }
  }
  CHECK_THROWS_WITH(
      (::fd::device::detail::BatchCells{100000, 100, {10, 10, 10}}),
      Catch::Matchers::ContainsSubstring("too many to enumerate with an int"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.VolumeView.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  test<1>();
  test<2>();
  test<3>();
  test_batch_cells();
}
