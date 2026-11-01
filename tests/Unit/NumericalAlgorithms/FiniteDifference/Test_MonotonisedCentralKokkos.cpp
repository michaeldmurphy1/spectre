// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentralKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
using ::fd::device::StridedVolumeView;
using ::fd::device::VolumeView;
namespace helpers = TestHelpers::fd::device;

constexpr size_t number_of_elements = 3;
constexpr size_t number_of_variables = 2;
// Monotonised central reconstruction needs 2 ghost cells
constexpr size_t ghost_width = 2;

// Views for the reconstructed data in each direction
template <size_t Dim>
std::array<VolumeView, Dim> make_face_views(const std::string& label,
                                            const Index<Dim>& volume_extents) {
  std::array<VolumeView, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    Index<Dim> face_extents = volume_extents;
    ++face_extents[d];
    gsl::at(result, d) = ::fd::device::make_volume_view(
        Kokkos::DefaultExecutionSpace{}, label, number_of_elements,
        number_of_variables, face_extents);
  }
  return result;
}

template <size_t Dim>
std::array<StridedVolumeView, Dim> strided(
    const std::array<VolumeView, Dim>& views) {
  std::array<StridedVolumeView, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = gsl::at(views, d);
  }
  return result;
}

// Reconstruct the data on the device, padded with ghost zones of width
// `padded_ghost_width`. Returns the host mirrors of the upper and lower side
// of the faces in each direction.
template <size_t Dim>
std::array<std::array<VolumeView::host_mirror_type, Dim>, 2>
reconstruct_on_device(const std::vector<helpers::ElementData<Dim>>& data,
                      const Index<Dim>& volume_extents,
                      const size_t padded_ghost_width = ghost_width) {
  const Kokkos::DefaultExecutionSpace exec{};
  const auto padded = helpers::pack(exec, data, number_of_variables,
                                    volume_extents, padded_ghost_width);
  const auto upper_side = make_face_views("upper side", volume_extents);
  const auto lower_side = make_face_views("lower side", volume_extents);
  ::fd::device::monotonised_central(
      exec, strided(upper_side), strided(lower_side), padded, volume_extents);
  return {{helpers::to_host(upper_side), helpers::to_host(lower_side)}};
}

// Compare to `fd::reconstruction::monotonised_central` with random data
template <size_t Dim>
void test_compare_to_host(const gsl::not_null<std::mt19937*> generator,
                          const Index<Dim>& volume_extents) {
  const auto data = helpers::make_random_element_data(
      generator, number_of_elements, number_of_variables, volume_extents,
      ghost_width);
  const auto [upper_side, lower_side] =
      reconstruct_on_device(data, volume_extents);

  for (size_t element = 0; element < number_of_elements; ++element) {
    const auto& volume_vars = data[element].volume_vars;
    std::array<DataVector, Dim> expected_upper{};
    std::array<DataVector, Dim> expected_lower{};
    std::array<gsl::span<double>, Dim> expected_upper_spans{};
    std::array<gsl::span<double>, Dim> expected_lower_spans{};
    for (size_t d = 0; d < Dim; ++d) {
      const size_t face_points = volume_extents.slice_away(d).product() *
                                 (volume_extents[d] + 1) * number_of_variables;
      gsl::at(expected_upper, d) = DataVector(face_points);
      gsl::at(expected_lower, d) = DataVector(face_points);
      gsl::at(expected_upper_spans, d) =
          gsl::make_span(gsl::at(expected_upper, d).data(), face_points);
      gsl::at(expected_lower_spans, d) =
          gsl::make_span(gsl::at(expected_lower, d).data(), face_points);
    }
    ::fd::reconstruction::monotonised_central(
        make_not_null(&expected_upper_spans),
        make_not_null(&expected_lower_spans),
        gsl::make_span(volume_vars.data(), volume_vars.size()),
        helpers::make_spans(data[element].ghost_data), volume_extents,
        number_of_variables);
    for (size_t d = 0; d < Dim; ++d) {
      CAPTURE(element);
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(
          helpers::element_data(gsl::at(upper_side, d), element),
          gsl::at(expected_upper, d));
      CHECK_ITERABLE_APPROX(
          helpers::element_data(gsl::at(lower_side, d), element),
          gsl::at(expected_lower, d));
    }
  }
}

// Monotonised central reconstruction is exact for linear data. The neighbors
// send more ghost cells than needed, which checks that the ghost zones are
// filled with the cells adjacent to the element. The padded view may also
// have wider ghost zones than needed, e.g. when it is shared with a
// higher-order derivative.
// The face views can be strided views into storage that holds more
// variables, e.g., a batch of face variables. The results are the same as
// with contiguous views, and the other variables are left unchanged.
template <size_t Dim>
void test_strided_face_views(const gsl::not_null<std::mt19937*> generator,
                             const Index<Dim>& volume_extents) {
  const auto data = helpers::make_random_element_data(
      generator, number_of_elements, number_of_variables, volume_extents,
      ghost_width);
  const auto [expected_upper_side, expected_lower_side] =
      reconstruct_on_device(data, volume_extents);

  const Kokkos::DefaultExecutionSpace exec{};
  const auto padded = helpers::pack(exec, data, number_of_variables,
                                    volume_extents, ghost_width);
  // One more variable than is reconstructed, before the reconstructed ones
  std::array<VolumeView, Dim> upper_storage{};
  std::array<VolumeView, Dim> lower_storage{};
  std::array<StridedVolumeView, Dim> upper_side{};
  std::array<StridedVolumeView, Dim> lower_side{};
  for (size_t d = 0; d < Dim; ++d) {
    Index<Dim> face_extents = volume_extents;
    ++face_extents[d];
    for (auto* storage :
         {&gsl::at(upper_storage, d), &gsl::at(lower_storage, d)}) {
      *storage =
          ::fd::device::make_volume_view(exec, "storage", number_of_elements,
                                         number_of_variables + 1, face_extents);
      Kokkos::deep_copy(exec, *storage, -1.0);
    }
    const auto variables =
        Kokkos::make_pair(size_t{1}, number_of_variables + 1);
    gsl::at(upper_side, d) =
        Kokkos::subview(gsl::at(upper_storage, d), Kokkos::ALL(), variables,
                        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    gsl::at(lower_side, d) =
        Kokkos::subview(gsl::at(lower_storage, d), Kokkos::ALL(), variables,
                        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }
  ::fd::device::monotonised_central(exec, upper_side, lower_side, padded,
                                    volume_extents);
  const auto upper = helpers::to_host(upper_storage);
  const auto lower = helpers::to_host(lower_storage);
  for (size_t d = 0; d < Dim; ++d) {
    for (const auto& [storage, expected] :
         {std::pair{gsl::at(upper, d), gsl::at(expected_upper_side, d)},
          std::pair{gsl::at(lower, d), gsl::at(expected_lower_side, d)}}) {
      for (size_t e = 0; e < storage.extent(0); ++e) {
        for (size_t k = 0; k < storage.extent(2); ++k) {
          for (size_t j = 0; j < storage.extent(3); ++j) {
            for (size_t i = 0; i < storage.extent(4); ++i) {
              CHECK(storage(e, 0, k, j, i) == -1.0);
              for (size_t v = 0; v < number_of_variables; ++v) {
                CHECK(storage(e, v + 1, k, j, i) == expected(e, v, k, j, i));
              }
            }
          }
        }
      }
    }
  }
}

template <size_t Dim>
void test_linear(const Index<Dim>& volume_extents,
                 const size_t padded_ghost_width) {
  CAPTURE(padded_ghost_width);
  const auto f = [](const size_t element, const size_t var, const auto& cell) {
    double result =
        static_cast<double>(element) - 2.0 * static_cast<double>(var);
    for (size_t d = 0; d < Dim; ++d) {
      result += static_cast<double>(1 + d + element) * gsl::at(cell, d);
    }
    return result;
  };
  const auto data =
      helpers::make_element_data(number_of_elements, number_of_variables,
                                 volume_extents, padded_ghost_width + 1, f);
  const auto [upper_side, lower_side] =
      reconstruct_on_device(data, volume_extents, padded_ghost_width);

  for (size_t d = 0; d < Dim; ++d) {
    Index<Dim> face_extents = volume_extents;
    ++face_extents[d];
    for (size_t element = 0; element < number_of_elements; ++element) {
      // Face `i` is at cell index `i - 1/2`
      const DataVector expected = helpers::make_data(
          face_extents, number_of_variables, std::array<int, Dim>{},
          [&f, element, d](const size_t var, const std::array<int, Dim>& face) {
            std::array<double, Dim> position{};
            for (size_t i = 0; i < Dim; ++i) {
              gsl::at(position, i) = gsl::at(face, i);
            }
            gsl::at(position, d) -= 0.5;
            return f(element, var, position);
          });
      CAPTURE(element);
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(
          helpers::element_data(gsl::at(upper_side, d), element), expected);
      CHECK_ITERABLE_APPROX(
          helpers::element_data(gsl::at(lower_side, d), element), expected);
    }
  }
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator) {
  const Index<Dim> volume_extents(6);
  test_compare_to_host(generator, volume_extents);
  test_strided_face_views(generator, volume_extents);
  test_linear(volume_extents, ghost_width);
  test_linear(volume_extents, ghost_width + 1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.MonotonisedCentral.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
