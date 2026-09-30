// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/ApplyPointwiseOnDevice.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#ifdef SPECTRE_KOKKOS
namespace TestHelpers {
/*!
 * \brief Check that `apply_pointwise_on_device<Function>` gives the same
 * results as the host implementation `host_function`.
 *
 * Tensor arguments are filled with random values in `[lower, upper]` on
 * `num_points` grid points, and non-tensor arguments (e.g. `double`s) with a
 * single random value. Arguments that can't be random (e.g. a `Mesh`) are
 * taken from `fixed_args` instead. `host_function` is called as
 * `host_function(gsl::not_null<ResultTags::type*>..., ArgsTags::type...)`.
 */
template <typename Function, typename... ResultTags, typename... ArgsTags,
          typename HostFunction, typename... FixedTags>
void check_pointwise_on_device(
    const HostFunction& host_function, tmpl::list<ResultTags...> result_tags,
    tmpl::list<ArgsTags...> args_tags, const size_t num_points,
    const double lower, const double upper,
    const tuples::TaggedTuple<FixedTags...>& fixed_args = {}) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(lower, upper);
  const auto make_random_arg = [&generator, &distribution, &num_points,
                                &fixed_args]<typename Tag>(
                                   tmpl::type_<Tag> /*meta*/) {
    using type = typename Tag::type;
    if constexpr (tmpl::list_contains_v<tmpl::list<FixedTags...>, Tag>) {
      return get<Tag>(fixed_args);
    } else if constexpr (tt::is_a_v<Tensor, type>) {
      return make_with_random_values<type>(make_not_null(&generator),
                                           make_not_null(&distribution),
                                           DataVector{num_points});
    } else {
      return static_cast<type>(distribution(generator));
    }
  };
  const tuples::TaggedTuple<ArgsTags...> args{
      make_random_arg(tmpl::type_<ArgsTags>{})...};

  Variables<tmpl::list<ResultTags...>> host_results{num_points};
  host_function(make_not_null(&get<ResultTags>(host_results))...,
                get<ArgsTags>(args)...);

  Variables<tmpl::list<ResultTags...>> device_results{num_points};
  apply_pointwise_on_device<Function>(make_not_null(&device_results),
                                      result_tags, args_tags,
                                      get<ArgsTags>(args)...);

  // The device code may be compiled differently (e.g. with fused
  // multiply-adds), so allow for a few ulps of difference that can be
  // amplified by cancellations.
  Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_VARIABLES_CUSTOM_APPROX(device_results, host_results, custom_approx);
}
}  // namespace TestHelpers
#endif  // SPECTRE_KOKKOS
