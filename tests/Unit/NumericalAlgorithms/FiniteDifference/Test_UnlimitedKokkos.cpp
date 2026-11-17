// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "NumericalAlgorithms/FiniteDifference/UnlimitedKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

namespace {
using ::fd::device::StridedVolumeView;
using ::fd::device::VolumeView;
namespace helpers = TestHelpers::fd::device;

constexpr size_t number_of_elements = 3;
constexpr size_t number_of_variables = 2;
// At least the stencil width of all degrees, as the host requires
constexpr size_t number_of_cells = 10;

// Reconstruct the last `number_of_variables` of the `number_of_variables + 1`
// variables of the data on the device, into views that hold one more
// variable before the reconstructed ones, so that neither the volume nor the
// face views are contiguous. Returns the host mirrors of the face storage
// (upper and lower side in each direction), whose first variable must be left
// at -1.
template <size_t Degree, size_t Dim>
std::array<std::array<VolumeView::host_mirror_type, Dim>, 2>
reconstruct_on_device(const std::vector<helpers::ElementData<Dim>>& data,
                      const Index<Dim>& volume_extents) {
  const Kokkos::DefaultExecutionSpace exec{};
  const auto padded = helpers::pack(exec, data, number_of_variables + 1,
                                    volume_extents, Degree / 2 + 1);
  const auto variables = Kokkos::make_pair(1_st, number_of_variables + 1);
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
    gsl::at(upper_side, d) =
        Kokkos::subview(gsl::at(upper_storage, d), Kokkos::ALL(), variables,
                        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    gsl::at(lower_side, d) =
        Kokkos::subview(gsl::at(lower_storage, d), Kokkos::ALL(), variables,
                        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }
  ::fd::device::unlimited<Degree>(
      exec, upper_side, lower_side,
      Kokkos::subview(padded, Kokkos::ALL(), variables, Kokkos::ALL(),
                      Kokkos::ALL(), Kokkos::ALL()),
      volume_extents);
  return {{helpers::to_host(upper_storage), helpers::to_host(lower_storage)}};
}

// The reconstructed variables of an element in the face storage, in the
// layout of a `Variables`. Checks that the first variable wasn't changed.
DataVector reconstructed_data(const VolumeView::host_mirror_type& storage,
                              const size_t element) {
  for (size_t k = 0; k < storage.extent(2); ++k) {
    for (size_t j = 0; j < storage.extent(3); ++j) {
      for (size_t i = 0; i < storage.extent(4); ++i) {
        CHECK(storage(element, 0, k, j, i) == -1.0);
      }
    }
  }
  DataVector result(number_of_variables * storage.extent(2) *
                    storage.extent(3) * storage.extent(4));
  size_t index = 0;
  for (size_t v = 0; v < number_of_variables; ++v) {
    for (size_t k = 0; k < storage.extent(2); ++k) {
      for (size_t j = 0; j < storage.extent(3); ++j) {
        for (size_t i = 0; i < storage.extent(4); ++i) {
          result[index++] = storage(element, v + 1, k, j, i);
        }
      }
    }
  }
  return result;
}

// The last `number_of_variables` variables of the data
DataVector last_variables(const DataVector& data) {
  const size_t size =
      data.size() / (number_of_variables + 1) * number_of_variables;
  DataVector result(size);
  std::copy(data.end() - static_cast<std::ptrdiff_t>(size), data.end(),
            result.begin());
  return result;
}

// Compare to `fd::reconstruction::unlimited` with random data
template <size_t Degree, size_t Dim>
void test_compare_to_host(const gsl::not_null<std::mt19937*> generator) {
  const Index<Dim> volume_extents(number_of_cells);
  const auto data = helpers::make_random_element_data(
      generator, number_of_elements, number_of_variables + 1, volume_extents,
      Degree / 2 + 1);
  const auto [upper_side, lower_side] =
      reconstruct_on_device<Degree>(data, volume_extents);

  for (size_t element = 0; element < number_of_elements; ++element) {
    const DataVector volume_vars = last_variables(data[element].volume_vars);
    DirectionMap<Dim, DataVector> ghost_data{};
    for (const auto& [direction, ghost_vars] : data[element].ghost_data) {
      ghost_data[direction] = last_variables(ghost_vars);
    }
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
    ::fd::reconstruction::unlimited<Degree>(
        make_not_null(&expected_upper_spans),
        make_not_null(&expected_lower_spans),
        gsl::make_span(volume_vars.data(), volume_vars.size()),
        helpers::make_spans(ghost_data), volume_extents, number_of_variables);
    for (size_t d = 0; d < Dim; ++d) {
      CAPTURE(element);
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(reconstructed_data(gsl::at(upper_side, d), element),
                            gsl::at(expected_upper, d));
      CHECK_ITERABLE_APPROX(reconstructed_data(gsl::at(lower_side, d), element),
                            gsl::at(expected_lower, d));
    }
  }
}

// Unlimited reconstruction of degree `Degree` is exact for polynomials of that
// degree, on both sides of the faces
template <size_t Degree, size_t Dim>
void test_polynomial() {
  const Index<Dim> volume_extents(number_of_cells);
  const auto f = [](const size_t element, const size_t var,
                    const auto& position) {
    double result =
        static_cast<double>(element) - 2.0 * static_cast<double>(var);
    for (size_t d = 0; d < Dim; ++d) {
      result += static_cast<double>(1 + d + element) *
                pow(0.2 * gsl::at(position, d), Degree);
    }
    return result;
  };
  const auto data =
      helpers::make_element_data(number_of_elements, number_of_variables + 1,
                                 volume_extents, Degree / 2 + 1, f);
  const auto [upper_side, lower_side] =
      reconstruct_on_device<Degree>(data, volume_extents);

  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
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
            return f(element, var + 1, position);
          });
      CAPTURE(element);
      CAPTURE(d);
      CHECK_ITERABLE_CUSTOM_APPROX(
          reconstructed_data(gsl::at(upper_side, d), element), expected,
          custom_approx);
      CHECK_ITERABLE_CUSTOM_APPROX(
          reconstructed_data(gsl::at(lower_side, d), element), expected,
          custom_approx);
    }
  }
}

template <size_t Degree>
void test(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Degree);
  test_compare_to_host<Degree, 1>(generator);
  test_compare_to_host<Degree, 2>(generator);
  test_compare_to_host<Degree, 3>(generator);
  test_polynomial<Degree, 1>();
  test_polynomial<Degree, 2>();
  test_polynomial<Degree, 3>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.Unlimited.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test<2>(make_not_null(&generator));
  test<4>(make_not_null(&generator));
  test<6>(make_not_null(&generator));
  test<8>(make_not_null(&generator));
}
