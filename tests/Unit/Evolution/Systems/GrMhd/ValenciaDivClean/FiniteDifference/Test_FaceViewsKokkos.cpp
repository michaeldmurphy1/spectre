// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <utility>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceViewsKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentralKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
using grmhd::ValenciaDivClean::fd::face_tags;
using grmhd::ValenciaDivClean::fd::reconstructed_face_tags;

// Reconstructing into the views of a batch of face variables gives the same
// values as reconstructing into separate views, and leaves the other face
// variables unchanged
void test_reconstruct_into_face_batch(
    const gsl::not_null<std::mt19937*> generator) {
  constexpr size_t number_of_elements = 2;
  constexpr size_t ghost_width = 2;
  constexpr size_t number_of_variables =
      Variables<reconstructed_face_tags>::number_of_independent_components;
  constexpr size_t number_of_components =
      Variables<face_tags>::number_of_independent_components;
  const Index<3> volume_extents{4, 4, 4};
  const Kokkos::DefaultExecutionSpace exec{};

  const auto padded = ::fd::device::make_volume_view(
      exec, "padded", number_of_elements, number_of_variables, volume_extents,
      ghost_width);
  const auto host_padded = Kokkos::create_mirror_view(padded);
  std::uniform_real_distribution<> dist(0.0, 1.0);
  for (size_t i = 0; i < host_padded.size(); ++i) {
    host_padded.data()[i] = dist(*generator);
  }
  Kokkos::deep_copy(exec, padded, host_padded);

  // Reconstruction into separate views
  std::array<::fd::device::VolumeView, 3> expected_upper{};
  std::array<::fd::device::VolumeView, 3> expected_lower{};
  for (size_t d = 0; d < 3; ++d) {
    Index<3> face_extents = volume_extents;
    ++face_extents[d];
    for (auto* view :
         {&gsl::at(expected_upper, d), &gsl::at(expected_lower, d)}) {
      *view = ::fd::device::make_volume_view(exec, "faces", number_of_elements,
                                             number_of_variables, face_extents);
    }
  }
  ::fd::device::monotonised_central(
      exec,
      std::array<::fd::device::StridedVolumeView, 3>{
          expected_upper[0], expected_upper[1], expected_upper[2]},
      std::array<::fd::device::StridedVolumeView, 3>{
          expected_lower[0], expected_lower[1], expected_lower[2]},
      padded, volume_extents);

  // Reconstruction into batches of face variables
  const size_t points_per_face_set = 5 * 4 * 4;
  BatchedVariables<face_tags> upper{exec, points_per_face_set,
                                    3 * number_of_elements};
  BatchedVariables<face_tags> lower{exec, points_per_face_set,
                                    3 * number_of_elements};
  Kokkos::deep_copy(exec, upper.view(), -1.0);
  Kokkos::deep_copy(exec, lower.view(), -1.0);
  ::fd::device::monotonised_central(
      exec,
      grmhd::ValenciaDivClean::fd::reconstructed_face_views(upper,
                                                            volume_extents),
      grmhd::ValenciaDivClean::fd::reconstructed_face_views(lower,
                                                            volume_extents),
      padded, volume_extents);

  for (const auto& [batch, expected] : {std::pair{&upper, &expected_upper},
                                        std::pair{&lower, &expected_lower}}) {
    const auto host_batch =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, batch->view());
    for (size_t d = 0; d < 3; ++d) {
      CAPTURE(d);
      const auto host_expected = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, gsl::at(*expected, d));
      for (size_t e = 0; e < number_of_elements; ++e) {
        const size_t entry = d * number_of_elements + e;
        for (size_t k = 0; k < host_expected.extent(2); ++k) {
          for (size_t j = 0; j < host_expected.extent(3); ++j) {
            for (size_t i = 0; i < host_expected.extent(4); ++i) {
              const size_t point = i + host_expected.extent(4) *
                                           (j + host_expected.extent(3) * k);
              for (size_t v = 0; v < number_of_variables; ++v) {
                CHECK(host_batch(point, v, entry) ==
                      host_expected(e, v, k, j, i));
              }
              for (size_t c = number_of_variables; c < number_of_components;
                   ++c) {
                CHECK(host_batch(point, c, entry) == -1.0);
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Fd.FaceViews",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test_reconstruct_into_face_batch(make_not_null(&generator));
}
