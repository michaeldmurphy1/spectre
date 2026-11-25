// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/Patches/DeviceComputeTagsKokkos.hpp"
#include "Evolution/Patches/DeviceObservationsKokkos.hpp"
#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarA : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// Four disjoint unit cubes named Shell0, Shell1, Shell2, and Cube, with the
// three shells in the group Shells
Domain<3> make_domain() {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      maps{};
  for (size_t block_id = 0; block_id < 4; ++block_id) {
    const double lower_x = 2.0 * static_cast<double>(block_id);
    maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3D{Affine{-1.0, 1.0, lower_x, lower_x + 1.0},
                     Affine{-1.0, 1.0, 0.0, 1.0},
                     Affine{-1.0, 1.0, 0.0, 1.0}}));
  }
  return Domain<3>{std::move(maps),
                   {},
                   {"Shell0", "Shell1", "Shell2", "Cube"},
                   {{"Shells", {"Shell0", "Shell1", "Shell2"}}}};
}

// The elements of the patch contribute to the subfile of their block or of
// all blocks, from the first element of each subfile
void test_slots_by_subfile() {
  using ObserveNorms = Events::ObserveNorms<tmpl::list<ScalarA>>;
  const auto domain = make_domain();
  const auto element = [](const size_t block, const size_t index) {
    return ElementId<3>{block, std::array{SegmentId{1, index}, SegmentId{0, 0},
                                          SegmentId{0, 0}}};
  };
  // The slots of the elements of the patch
  const std::vector<ElementId<3>> element_ids{element(0, 0), element(3, 0),
                                              element(0, 1), element(1, 0),
                                              element(2, 1)};
  const ObserveNorms whole_domain{"Norms", {}};
  CHECK(evolution::patches::slots_by_subfile(whole_domain, "", domain,
                                             element_ids) ==
        std::vector<std::vector<size_t>>{{0, 1, 2, 3, 4}});
  // The cube isn't observed
  const ObserveNorms shells_by_block{"Norms",
                                     {},
                                     std::vector<std::string>{"Shells"},
                                     Events::ReductionScope::ByBlock};
  CHECK(evolution::patches::slots_by_subfile(shells_by_block, "", domain,
                                             element_ids) ==
        std::vector<std::vector<size_t>>{{0, 2}, {3}, {4}});
  // Elements that aren't in the array section (without a section key) don't
  // contribute
  CHECK(evolution::patches::slots_by_subfile(whole_domain, std::nullopt, domain,
                                             element_ids)
            .empty());
  const ObserveNorms nothing{"Norms", {}, std::vector<std::string>{"Cube"}};
  CHECK(evolution::patches::slots_by_subfile(nothing, "", domain,
                                             {element(0, 0), element(1, 0)})
            .empty());
}

Events::NormStatistics statistics(const double value) {
  return {.max = value,
          .min = -value,
          .sum_abs = 2.0 * value,
          .sum_square = 3.0 * value,
          .integral_abs = 4.0 * value,
          .integral_square = 5.0 * value,
          .integral = 6.0 * value};
}

void check_equal(const Events::NormStatistics& a,
                 const Events::NormStatistics& b) {
  CHECK(a.max == b.max);
  CHECK(a.min == b.min);
  CHECK(a.sum_abs == b.sum_abs);
  CHECK(a.sum_square == b.sum_square);
  CHECK(a.integral_abs == b.integral_abs);
  CHECK(a.integral_square == b.integral_square);
  CHECK(a.integral == b.integral);
}

// The statistics of the elements that contribute to a subfile are combined,
// and their volumes summed
void test_combine_slot_statistics() {
  evolution::patches::device_compute::DeviceNormStatistics observables{};
  observables.tensors = {{"A", 0, {"A"}}, {"B", 1, {"B_x", "B_y"}}};
  // 3 elements, 3 components and the volume
  observables.statistics =
      Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight,
                   Kokkos::HostSpace>{"statistics", 3, 4};
  for (size_t element = 0; element < 3; ++element) {
    for (size_t column = 0; column < 4; ++column) {
      observables.statistics(element, column) =
          statistics(static_cast<double>(10 * element + column + 1));
    }
  }
  const auto [tensors, volume] =
      evolution::patches::combine_slot_statistics(observables, {0, 2});
  // The `integral` of the last column, 6 * (4 + 24)
  CHECK(volume == 168.0);
  REQUIRE(tensors.size() == 2);
  CHECK(tensors.at("A").first == std::vector<std::string>{"A"});
  CHECK(tensors.at("B").first == std::vector<std::string>{"B_x", "B_y"});
  REQUIRE(tensors.at("A").second.size() == 1);
  REQUIRE(tensors.at("B").second.size() == 2);
  check_equal(tensors.at("A").second[0],
              Events::combine(statistics(1.0), statistics(21.0)));
  check_equal(tensors.at("B").second[0],
              Events::combine(statistics(2.0), statistics(22.0)));
  check_equal(tensors.at("B").second[1],
              Events::combine(statistics(3.0), statistics(23.0)));

  const auto [single, single_volume] =
      evolution::patches::combine_slot_statistics(observables, {1});
  CHECK(single_volume == 6.0 * 14.0);
  check_equal(single.at("B").second[1], statistics(13.0));
}

void test_add_integrals() {
  std::tuple<std::array<double, 1>, std::array<double, 2>> sum{{1.0},
                                                               {2.0, 3.0}};
  evolution::patches::add_integrals(
      make_not_null(&sum),
      std::tuple<std::array<double, 1>, std::array<double, 2>>{{10.0},
                                                               {20.0, 30.0}});
  CHECK(std::get<0>(sum) == std::array{11.0});
  CHECK(std::get<1>(sum) == std::array{22.0, 33.0});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Patches.DeviceObservationsKokkos",
                  "[Unit][Evolution]") {
  test_slots_by_subfile();
  test_combine_slot_statistics();
  test_add_integrals();
}
