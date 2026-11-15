// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotScaleTrans.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using RotScaleTrans3D = domain::CoordinateMaps::TimeDependent::RotScaleTrans<3>;
using FunctionsOfTimeMap = std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
constexpr double inner_radius = 2.0;
constexpr double outer_radius = 20.0;

template <size_t Dim>
FunctionsOfTimeMap make_functions_of_time() {
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<3>;
  const double initial_time = 0.0;
  const double expiration_time = 10.0;
  FunctionsOfTimeMap functions_of_time{};
  // Time-dependent expansion factors and translation, so the frame velocity
  // has all terms
  functions_of_time["ExpansionA"] = std::make_unique<Polynomial>(
      initial_time, std::array<DataVector, 4>{{{0.9}, {-0.02}, {0.003}, {0.0}}},
      expiration_time);
  functions_of_time["ExpansionB"] = std::make_unique<Polynomial>(
      initial_time, std::array<DataVector, 4>{{{1.0}, {-0.001}, {0.0}, {0.0}}},
      expiration_time);
  functions_of_time["Translation"] = std::make_unique<Polynomial>(
      initial_time,
      std::array<DataVector, 4>{
          {Dim == 3 ? DataVector{0.3, -0.4, 0.2} : DataVector{0.3, -0.4},
           Dim == 3 ? DataVector{-0.2, 0.1, 0.05} : DataVector{-0.2, 0.1},
           DataVector(Dim, 0.01), DataVector(Dim, 0.0)}},
      expiration_time);
  if constexpr (Dim == 2) {
    functions_of_time["Rotation"] = std::make_unique<Polynomial>(
        initial_time, std::array<DataVector, 4>{{{0.7}, {0.3}, {-0.05}, {0.0}}},
        expiration_time);
  } else {
    DataVector axis{{1.0, -2.0, 0.5}};
    axis /= sqrt(5.25);
    const double angle = 0.7;
    functions_of_time["Rotation"] =
        std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
            initial_time,
            std::array<DataVector, 1>{DataVector{
                {cos(0.5 * angle), axis[0] * sin(0.5 * angle),
                 axis[1] * sin(0.5 * angle), axis[2] * sin(0.5 * angle)}}},
            std::array<DataVector, 4>{axis * angle, axis * 0.3, axis * -0.05,
                                      axis * 0.0},
            expiration_time);
  }
  return functions_of_time;
}

// A point with a radius between `min_radius` and `max_radius`
template <size_t Dim>
tnsr::I<double, Dim, Frame::Grid> random_point(
    const gsl::not_null<std::mt19937*> generator, const double min_radius,
    const double max_radius) {
  std::uniform_real_distribution<double> component_distribution{-1.0, 1.0};
  std::uniform_real_distribution<double> radius_distribution{min_radius,
                                                             max_radius};
  tnsr::I<double, Dim, Frame::Grid> point{};
  double norm = 0.0;
  while (norm < 0.1) {
    norm = 0.0;
    for (size_t i = 0; i < Dim; ++i) {
      point.get(i) = component_distribution(*generator);
      norm += square(point.get(i));
    }
    norm = sqrt(norm);
  }
  const double radius = radius_distribution(*generator);
  for (size_t i = 0; i < Dim; ++i) {
    point.get(i) *= radius / norm;
  }
  return point;
}

// Compares the device map of `map` at `time` with the host map at random
// points with radii in [min_radius, max_radius]
template <size_t Dim, typename... Maps>
void check_against_host(
    const gsl::not_null<std::mt19937*> generator,
    const domain::CoordinateMap<Frame::Grid, Frame::Inertial, Maps...>& map,
    const double time, const FunctionsOfTimeMap& functions_of_time,
    const double min_radius, const double max_radius,
    const bool expected_affine) {
  const std::optional<domain::CoordinateMaps::DeviceMap<Dim>> device_map =
      map.device_map(time, functions_of_time);
  REQUIRE(device_map.has_value());
  CHECK(device_map->is_affine() == expected_affine);
  CHECK(not device_map->is_identity());
  for (size_t p = 0; p < 20; ++p) {
    const auto point = random_point<Dim>(generator, min_radius, max_radius);
    CAPTURE(point);
    const auto [coords, inv_jacobian, jacobian, frame_velocity] =
        map.coords_frame_velocity_jacobians(point, time, functions_of_time);
    tnsr::I<double, Dim, Frame::NoFrame> source{};
    for (size_t i = 0; i < Dim; ++i) {
      source.get(i) = point.get(i);
    }
    const auto device_result = (*device_map)(source);
    for (size_t i = 0; i < Dim; ++i) {
      CHECK(device_result.coords.get(i) == approx(coords.get(i)));
      CHECK(device_result.frame_velocity.get(i) ==
            approx(frame_velocity.get(i)));
      for (size_t j = 0; j < Dim; ++j) {
        CHECK(device_result.jacobian.get(i, j) == approx(jacobian.get(i, j)));
        CHECK(device_result.inverse_jacobian.get(i, j) ==
              approx(inv_jacobian.get(i, j)));
      }
    }
  }
}

template <size_t Dim>
void test_rot_scale_trans(const gsl::not_null<std::mt19937*> generator) {
  using RotScaleTrans =
      domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>;
  using Region = typename RotScaleTrans::BlockRegion;
  const auto functions_of_time = make_functions_of_time<Dim>();
  const double time = 1.3;
  const std::pair<std::string, std::string> expansion_names{"ExpansionA",
                                                            "ExpansionB"};
  for (const bool rotation : {false, true}) {
    for (const bool expansion : {false, true}) {
      for (const bool translation : {false, true}) {
        if (not(rotation or expansion or translation)) {
          continue;
        }
        for (const Region region :
             {Region::Inner, Region::Transition, Region::Outer}) {
          CAPTURE(Dim);
          CAPTURE(rotation);
          CAPTURE(expansion);
          CAPTURE(translation);
          CAPTURE(static_cast<int>(region));
          const domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                      RotScaleTrans>
              map{RotScaleTrans{
                  expansion ? std::optional{expansion_names} : std::nullopt,
                  rotation ? std::optional<std::string>{"Rotation"}
                           : std::nullopt,
                  translation ? std::optional<std::string>{"Translation"}
                              : std::nullopt,
                  inner_radius, outer_radius, region}};
          const bool affine =
              region != Region::Transition or not(expansion or translation);
          if (region == Region::Inner) {
            check_against_host<Dim>(generator, map, time, functions_of_time,
                                    0.0, inner_radius, affine);
          } else if (region == Region::Outer) {
            check_against_host<Dim>(generator, map, time, functions_of_time,
                                    outer_radius, 3.0 * outer_radius, affine);
          } else {
            // Points on both sides of the midpoint, where the formulas change
            check_against_host<Dim>(generator, map, time, functions_of_time,
                                    inner_radius, outer_radius, affine);
          }
        }
      }
    }
  }
}

// Two maps in sequence, composed as `CoordinateMap` does
void test_chain(const gsl::not_null<std::mt19937*> generator) {
  using Region = RotScaleTrans3D::BlockRegion;
  const auto functions_of_time = make_functions_of_time<3>();
  const std::pair<std::string, std::string> expansion_names{"ExpansionA",
                                                            "ExpansionB"};
  const domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotScaleTrans3D,
                              domain::CoordinateMaps::Identity<3>,
                              RotScaleTrans3D>
      map{RotScaleTrans3D{expansion_names, std::nullopt, "Translation",
                          inner_radius, outer_radius, Region::Transition},
          domain::CoordinateMaps::Identity<3>{},
          RotScaleTrans3D{std::nullopt, "Rotation", std::nullopt, inner_radius,
                          outer_radius, Region::Inner}};
  CHECK(map.device_map(0.4, functions_of_time)->number_of_stages == 2);
  check_against_host<3>(generator, map, 0.4, functions_of_time, inner_radius,
                        outer_radius, false);
}

// An affine map after a RotScaleTrans, checked against composing them by hand
void test_affine(const gsl::not_null<std::mt19937*> generator) {
  using Region = RotScaleTrans3D::BlockRegion;
  const auto functions_of_time = make_functions_of_time<3>();
  const double time = 0.4;
  const domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotScaleTrans3D>
      first{RotScaleTrans3D{
          std::pair<std::string, std::string>{"ExpansionA", "ExpansionB"},
          "Rotation", "Translation", inner_radius, outer_radius,
          Region::Transition}};
  domain::CoordinateMaps::DeviceAffineMap<3> affine{};
  std::uniform_real_distribution<double> distribution{-1.0, 1.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      affine.matrix.get(i, j) = (i == j ? 2.0 : 0.0) + distribution(*generator);
      affine.matrix_derivative.get(i, j) = distribution(*generator);
    }
    gsl::at(affine.offset, i) = distribution(*generator);
    gsl::at(affine.offset_derivative, i) = distribution(*generator);
  }
  auto device_map = first.device_map(time, functions_of_time).value();
  device_map.push_back(affine);
  CHECK(device_map.number_of_stages == 2);
  CHECK_FALSE(device_map.is_affine());
  domain::CoordinateMaps::DeviceMap<3> only_affine{};
  only_affine.push_back(affine);
  CHECK(only_affine.is_affine());
  for (size_t p = 0; p < 10; ++p) {
    const auto point = random_point<3>(generator, inner_radius, outer_radius);
    const auto [coords, inv_jacobian, jacobian, frame_velocity] =
        first.coords_frame_velocity_jacobians(point, time, functions_of_time);
    tnsr::I<double, 3, Frame::NoFrame> source{};
    for (size_t i = 0; i < 3; ++i) {
      source.get(i) = point.get(i);
    }
    const auto result = device_map(source);
    for (size_t i = 0; i < 3; ++i) {
      double expected_coords = gsl::at(affine.offset, i);
      double expected_velocity = gsl::at(affine.offset_derivative, i);
      for (size_t j = 0; j < 3; ++j) {
        expected_coords += affine.matrix.get(i, j) * coords.get(j);
        expected_velocity += affine.matrix.get(i, j) * frame_velocity.get(j) +
                             affine.matrix_derivative.get(i, j) * coords.get(j);
        double expected_jacobian = 0.0;
        for (size_t k = 0; k < 3; ++k) {
          expected_jacobian += affine.matrix.get(i, k) * jacobian.get(k, j);
        }
        CHECK(result.jacobian.get(i, j) == approx(expected_jacobian));
      }
      CHECK(result.coords.get(i) == approx(expected_coords));
      CHECK(result.frame_velocity.get(i) == approx(expected_velocity));
    }
    // The inverse Jacobian inverts the Jacobian
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        double product = 0.0;
        for (size_t k = 0; k < 3; ++k) {
          product +=
              result.jacobian.get(i, k) * result.inverse_jacobian.get(k, j);
        }
        CHECK(product == approx(i == j ? 1.0 : 0.0));
      }
    }
  }
}

void test_identity_and_unsupported() {
  const auto functions_of_time = make_functions_of_time<3>();
  const domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                              domain::CoordinateMaps::Identity<3>>
      identity{domain::CoordinateMaps::Identity<3>{}};
  const auto device_identity = identity.device_map(0.4, functions_of_time);
  REQUIRE(device_identity.has_value());
  CHECK(device_identity->is_identity());
  CHECK(device_identity->is_affine());
  const tnsr::I<double, 3, Frame::NoFrame> point{{{1.0, -2.0, 3.0}}};
  const auto mapped = (*device_identity)(point);
  CHECK(mapped.coords == point);
  CHECK(mapped.frame_velocity == tnsr::I<double, 3, Frame::NoFrame>{0.0});
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK(mapped.jacobian.get(i, j) == (i == j ? 1.0 : 0.0));
      CHECK(mapped.inverse_jacobian.get(i, j) == (i == j ? 1.0 : 0.0));
    }
  }

  // Maps without a device version give no device map
  const domain::CoordinateMap<
      Frame::Grid, Frame::Inertial,
      domain::CoordinateMaps::TimeDependent::CubicScale<3>>
      cubic_scale{domain::CoordinateMaps::TimeDependent::CubicScale<3>{
          outer_radius, "ExpansionA", "ExpansionB"}};
  CHECK_FALSE(cubic_scale.device_map(0.4, functions_of_time).has_value());

  const domain::CoordinateMaps::DeviceMap<3> empty{};
  CHECK(empty.is_identity());
  domain::CoordinateMaps::DeviceMap<3> full{};
  for (size_t i = 0;
       i < domain::CoordinateMaps::DeviceMap<3>::maximum_number_of_stages;
       ++i) {
    full.push_back(
        domain::CoordinateMaps::TimeDependent::DeviceRotScaleTrans<3>{});
  }
  CHECK_THROWS_WITH(
      full.push_back(
          domain::CoordinateMaps::TimeDependent::DeviceRotScaleTrans<3>{}),
      Catch::Matchers::ContainsSubstring("holds at most 4 maps"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.DeviceMap", "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test_rot_scale_trans<2>(make_not_null(&generator));
  test_rot_scale_trans<3>(make_not_null(&generator));
  test_chain(make_not_null(&generator));
  test_affine(make_not_null(&generator));
  test_identity_and_unsupported();
}
