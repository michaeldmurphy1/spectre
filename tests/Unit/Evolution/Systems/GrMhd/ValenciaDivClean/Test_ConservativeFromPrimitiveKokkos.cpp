// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitiveOnDeviceKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDeviceKokkos.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// `ConservativeFromPrimitiveOnDevice` computes the same conserved variables as
// `ConservativeFromPrimitive` on the host
void test_on_device_mutator(const gsl::not_null<std::mt19937*> generator) {
  using Mutator = grmhd::ValenciaDivClean::ConservativeFromPrimitive;
  constexpr size_t number_of_points = 17;
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  const auto args = make_with_random_values<Variables<Mutator::argument_tags>>(
      generator, make_not_null(&dist), DataVector(number_of_points));
  Variables<Mutator::return_tags> expected(number_of_points);
  Variables<Mutator::return_tags> result(number_of_points);
  tmpl::as_pack<Mutator::return_tags>([&](auto... return_tags) {
    tmpl::as_pack<Mutator::argument_tags>([&](auto... argument_tags) {
      Mutator::apply(make_not_null(&get<tmpl::type_from<decltype(return_tags)>>(
                         expected))...,
                     get<tmpl::type_from<decltype(argument_tags)>>(args)...);
      grmhd::ValenciaDivClean::ConservativeFromPrimitiveOnDevice::apply(
          make_not_null(
              &get<tmpl::type_from<decltype(return_tags)>>(result))...,
          get<tmpl::type_from<decltype(argument_tags)>>(args)...);
    });
  });
  CHECK_VARIABLES_APPROX(result, expected);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GrMhd.ValenciaDivClean.ConservativeFromPrimitive.Kokkos",
    "[Unit][GrMhd]") {
  TestHelpers::check_pointwise_on_device<
      grmhd::ValenciaDivClean::ConservativeFromPrimitive>(
      &grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply<DataVector>,
      grmhd::ValenciaDivClean::ConservativeFromPrimitive::return_tags{},
      grmhd::ValenciaDivClean::ConservativeFromPrimitive::argument_tags{}, 5,
      0.0, 1.0);
  MAKE_GENERATOR(generator);
  test_on_device_mutator(make_not_null(&generator));
}
