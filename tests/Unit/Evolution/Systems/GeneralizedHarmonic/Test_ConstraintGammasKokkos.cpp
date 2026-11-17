// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintGammasKokkos.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/ConstraintDamping/Constant.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianPlusConstant.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/ConstraintDamping/TimeDependentTripleGaussian.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using DampingFunction = ConstraintDamping::DampingFunction<3, Frame::Grid>;
using FunctionsOfTime = std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
using coordinates_tag = domain::Tags::Coordinates<3, Frame::Grid>;
using tags = tmpl::list<gh::Tags::ConstraintGamma0, gh::Tags::ConstraintGamma1,
                        gh::Tags::ConstraintGamma2, coordinates_tag>;

constexpr size_t number_of_elements = 3;
constexpr size_t number_of_points = 50;

// Evaluate the damping functions of the three gammas at random points on the
// device and compare to the call operators on the host
void test_gammas(const gsl::not_null<std::mt19937*> generator,
                 const std::array<const DampingFunction*, 3>& damping_functions,
                 const double time, const FunctionsOfTime& functions_of_time) {
  std::uniform_real_distribution<> coordinate_distribution(-20.0, 20.0);
  BatchedVariables<tags, Kokkos::HostSpace> host_batch{number_of_points,
                                                       number_of_elements};
  std::vector<tnsr::I<DataVector, 3, Frame::Grid>> coordinates{};
  for (size_t e = 0; e < number_of_elements; ++e) {
    coordinates.push_back(
        make_with_random_values<tnsr::I<DataVector, 3, Frame::Grid>>(
            generator, make_not_null(&coordinate_distribution),
            DataVector(number_of_points)));
    Variables<tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    get<coordinates_tag>(element) = coordinates.back();
  }
  const Kokkos::DefaultExecutionSpace exec{};
  BatchedVariables<tags> batch{exec, number_of_points, number_of_elements};
  deep_copy(exec, make_not_null(&batch), host_batch);

  const Kokkos::View<ConstraintDamping::GaussianSum<3>*> gaussian_sums{
      "gaussian sums", 3};
  const auto host_gaussian_sums =
      Kokkos::create_mirror_view(Kokkos::HostSpace{}, gaussian_sums);
  for (size_t i = 0; i < 3; ++i) {
    host_gaussian_sums(i) =
        gsl::at(damping_functions, i)->as_gaussian_sum(time, functions_of_time);
  }
  Kokkos::deep_copy(exec, gaussian_sums, host_gaussian_sums);
  gh::constraint_gammas_on_device(
      exec, get<Tags::BatchView<gh::Tags::ConstraintGamma0>>(batch),
      get<Tags::BatchView<gh::Tags::ConstraintGamma1>>(batch),
      get<Tags::BatchView<gh::Tags::ConstraintGamma2>>(batch),
      get<Tags::BatchView<coordinates_tag>>(batch), gaussian_sums);
  deep_copy(exec, make_not_null(&host_batch), batch);
  exec.fence();

  for (size_t e = 0; e < number_of_elements; ++e) {
    CAPTURE(e);
    Variables<tags> element{};
    set_data_ref_to_element(make_not_null(&element), host_batch, e);
    std::array<Scalar<DataVector>, 3> expected{};
    for (size_t i = 0; i < 3; ++i) {
      (*gsl::at(damping_functions, i))(make_not_null(&gsl::at(expected, i)),
                                       coordinates[e], time, functions_of_time);
    }
    CHECK_ITERABLE_APPROX(get<gh::Tags::ConstraintGamma0>(element),
                          expected[0]);
    CHECK_ITERABLE_APPROX(get<gh::Tags::ConstraintGamma1>(element),
                          expected[1]);
    CHECK_ITERABLE_APPROX(get<gh::Tags::ConstraintGamma2>(element),
                          expected[2]);
  }
}

void test(const gsl::not_null<std::mt19937*> generator) {
  const double time = 1.3;
  FunctionsOfTime functions_of_time{};
  functions_of_time["GridCenters"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<1>>(
          0.0,
          std::array<DataVector, 2>{{{8.0, 0.5, -0.2, -9.0, -0.3, 0.1},
                                     {-0.5, 0.3, 0.0, 0.6, -0.2, 0.1}}},
          std::numeric_limits<double>::max());
  functions_of_time["Expansion"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<1>>(
          0.0, std::array<DataVector, 2>{{{1.0}, {-0.1}}},
          std::numeric_limits<double>::max());

  // With widths comparable to the range of the coordinates, so all Gaussians
  // contribute
  const ConstraintDamping::TimeDependentTripleGaussian object_centers{
      0.01,
      0.06,
      5.0,
      std::nullopt,
      0.07,
      4.0,
      std::nullopt,
      0.08,
      15.0,
      std::array{0.5, -0.2, 0.1},
      "ObjectCenters"};
  const ConstraintDamping::TimeDependentTripleGaussian expansion_factor{
      0.02,
      0.9,
      6.0,
      std::array{8.0, 0.0, 0.0},
      0.8,
      7.0,
      std::array{-8.0, 0.0, 0.0},
      0.2,
      20.0,
      std::array{0.0, 0.0, 0.0},
      "ExpansionFactor"};
  const ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>
      gaussian_plus_constant{-0.999, 0.999, 30.0, {{1.0, 2.0, -1.0}}};
  const ConstraintDamping::Constant<3, Frame::Grid> constant{0.3};

  test_gammas(generator,
              {{&object_centers, &gaussian_plus_constant, &expansion_factor}},
              time, functions_of_time);
  test_gammas(generator, {{&constant, &expansion_factor, &object_centers}},
              time, functions_of_time);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Gh.ConstraintGammas.Kokkos",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
}
