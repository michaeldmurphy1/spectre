// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/Device.hpp"
#include "NumericalAlgorithms/FiniteDifference/Device/VolumeView.hpp"
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

  const auto unpadded = ::fd::device::make_volume_view(
      "unpadded", number_of_elements, number_of_variables, volume_extents);
  const auto padded = ::fd::device::make_volume_view(
      "padded", number_of_elements, number_of_variables, volume_extents,
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
  const auto host_padded = Kokkos::create_mirror_view(padded);
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
            // the zeros the view was initialized with.
            CHECK(host_padded(element, var, z, y, x) ==
                  (dims_outside_element <= 1 ? f(element, var, cell) : 0.0));
          }
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.Device.VolumeView",
                  "[Unit][NumericalAlgorithms]") {
  test<1>();
  test<2>();
  test<3>();
}
