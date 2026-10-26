// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/list/for_each_i.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <cstddef>
#include <random>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace TestHelpers::EquationsOfState {
namespace DeviceView_detail {
#define SPECTRE_EOS_DEVICE_VIEW_FUNCTIONS(DIM)                               \
  BOOST_PP_TUPLE_ELEM(BOOST_PP_SUB(DIM, 1), (EQUATION_OF_STATE_FUNCTIONS_1D, \
                                             EQUATION_OF_STATE_FUNCTIONS_2D, \
                                             EQUATION_OF_STATE_FUNCTIONS_3D))
#define SPECTRE_EOS_DEVICE_VIEW_ARGS_1 \
  Scalar<double> { args[0] }
#define SPECTRE_EOS_DEVICE_VIEW_ARGS_2 \
  Scalar<double>{args[0]}, Scalar<double> { args[1] }
#define SPECTRE_EOS_DEVICE_VIEW_ARGS_3 \
  Scalar<double>{args[0]}, Scalar<double>{args[1]}, Scalar<double> { args[2] }
#define SPECTRE_EOS_DEVICE_VIEW_EVALUATE(r, DIM, i, FUNCTION_NAME) \
  results[i] = get(                                                \
      eos.FUNCTION_NAME(BOOST_PP_CAT(SPECTRE_EOS_DEVICE_VIEW_ARGS_, DIM)));
#define SPECTRE_EOS_DEVICE_VIEW_EVALUATE_ALL(DIM) \
  BOOST_PP_LIST_FOR_EACH_I(                       \
      SPECTRE_EOS_DEVICE_VIEW_EVALUATE, DIM,      \
      BOOST_PP_TUPLE_TO_LIST(SPECTRE_EOS_DEVICE_VIEW_FUNCTIONS(DIM)))

// 1D EOSs also have `specific_entropy_from_density`, which isn't in
// `EQUATION_OF_STATE_FUNCTIONS_1D`
template <size_t Dim>
constexpr size_t number_of_functions =
    Dim == 1   ? BOOST_PP_TUPLE_SIZE(SPECTRE_EOS_DEVICE_VIEW_FUNCTIONS(1)) + 1
    : Dim == 2 ? BOOST_PP_TUPLE_SIZE(SPECTRE_EOS_DEVICE_VIEW_FUNCTIONS(2))
               : BOOST_PP_TUPLE_SIZE(SPECTRE_EOS_DEVICE_VIEW_FUNCTIONS(3));

// Evaluates all functions of a `Dim`-dimensional EOS (the EOS on the host or
// its `DeviceView`) at the thermodynamic arguments `args`
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <size_t Dim, typename Eos>
KOKKOS_FUNCTION Kokkos::Array<double, number_of_functions<Dim>> evaluate_all(
    const Eos& eos, const Kokkos::Array<double, Dim>& args) {
  Kokkos::Array<double, number_of_functions<Dim>> results{};
  if constexpr (Dim == 1) {
    SPECTRE_EOS_DEVICE_VIEW_EVALUATE_ALL(1)
    results[number_of_functions<1> - 1] =
        get(eos.specific_entropy_from_density(Scalar<double>{args[0]}));
  } else if constexpr (Dim == 2) {
    SPECTRE_EOS_DEVICE_VIEW_EVALUATE_ALL(2)
  } else {
    SPECTRE_EOS_DEVICE_VIEW_EVALUATE_ALL(3)
  }
  return results;
}

// Evaluates the bounds of an EOS (the EOS on the host or its `DeviceView`) at
// the thermodynamic arguments `args`, of which only the rest mass density (in
// 2D and 3D) and the electron fraction (in 3D) are used. Not all `DeviceView`s
// have a temperature bound yet.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <size_t Dim, bool WithTemperature, typename Eos>
KOKKOS_FUNCTION Kokkos::Array<double, 8> evaluate_bounds(
    const Eos& eos, const Kokkos::Array<double, Dim>& args) {
  Kokkos::Array<double, 8> results{};
  results[0] = eos.rest_mass_density_lower_bound();
  results[1] = eos.rest_mass_density_upper_bound();
  results[2] = eos.specific_enthalpy_lower_bound();
  if constexpr (Dim == 1) {
    results[3] = eos.specific_internal_energy_lower_bound();
    results[4] = eos.specific_internal_energy_upper_bound();
  } else if constexpr (Dim == 2) {
    results[3] = eos.specific_internal_energy_lower_bound(args[0]);
    results[4] = eos.specific_internal_energy_upper_bound(args[0]);
  } else {
    results[3] = eos.specific_internal_energy_lower_bound(args[0], args[2]);
    results[4] = eos.specific_internal_energy_upper_bound(args[0], args[2]);
    results[5] = eos.electron_fraction_lower_bound();
    results[6] = eos.electron_fraction_upper_bound();
  }
  if constexpr (WithTemperature) {
    results[7] = eos.temperature_lower_bound();
  }
  return results;
}

#undef SPECTRE_EOS_DEVICE_VIEW_EVALUATE_ALL
#undef SPECTRE_EOS_DEVICE_VIEW_EVALUATE
#undef SPECTRE_EOS_DEVICE_VIEW_ARGS_3
#undef SPECTRE_EOS_DEVICE_VIEW_ARGS_2
#undef SPECTRE_EOS_DEVICE_VIEW_ARGS_1
#undef SPECTRE_EOS_DEVICE_VIEW_FUNCTIONS
}  // namespace DeviceView_detail

/*!
 * \brief Check that all functions of `eos.device_view()` give the same
 * results in a Kokkos kernel as the EOS on the host.
 *
 * The thermodynamic arguments are random values in the `ranges`, which must be
 * valid for all functions of the EOS.
 */
template <typename EosType>
void check_device_view(const EosType& eos,
                       const std::array<std::pair<double, double>,
                                        EosType::thermodynamic_dim>& ranges) {
  constexpr size_t dim = EosType::thermodynamic_dim;
  constexpr size_t num_functions = DeviceView_detail::number_of_functions<dim>;
  const auto device_view = eos.device_view();
  static_assert(std::is_trivially_copyable_v<decltype(device_view)>,
                "The DeviceView must be trivially copyable so it can be passed "
                "to Kokkos kernels.");

  const size_t num_points = 20;
  MAKE_GENERATOR(generator);
  const Kokkos::View<double**> args("args", num_points, dim);
  const auto host_args = Kokkos::create_mirror_view(args);
  for (size_t d = 0; d < dim; ++d) {
    std::uniform_real_distribution<> distribution(gsl::at(ranges, d).first,
                                                  gsl::at(ranges, d).second);
    for (size_t p = 0; p < num_points; ++p) {
      host_args(p, d) = distribution(generator);
    }
  }
  Kokkos::deep_copy(args, host_args);

  const Kokkos::View<double**> results("results", num_points, num_functions);
  Kokkos::parallel_for(
      "check_device_view",
      Kokkos::RangePolicy<Kokkos::IndexType<size_t>>(0, num_points),
      KOKKOS_LAMBDA(const size_t p) {
        Kokkos::Array<double, dim> args_at_point{};
        for (size_t d = 0; d < dim; ++d) {
          args_at_point[d] = args(p, d);
        }
        const auto results_at_point =
            DeviceView_detail::evaluate_all<dim>(device_view, args_at_point);
        for (size_t i = 0; i < num_functions; ++i) {
          results(p, i) = results_at_point[i];
        }
      });
  const auto host_results =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);

  // The device code may be compiled differently (e.g. with fused
  // multiply-adds)
  const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  for (size_t p = 0; p < num_points; ++p) {
    Kokkos::Array<double, dim> args_at_point{};
    for (size_t d = 0; d < dim; ++d) {
      args_at_point[d] = host_args(p, d);
    }
    const auto expected =
        DeviceView_detail::evaluate_all<dim>(eos, args_at_point);
    for (size_t i = 0; i < num_functions; ++i) {
      CAPTURE(p);
      CAPTURE(i);
      CHECK(host_results(p, i) == custom_approx(expected[i]));
    }
  }
}

/*!
 * \brief Check that the bounds of `eos.device_view()` are the same in a Kokkos
 * kernel as those of the EOS on the host.
 *
 * The thermodynamic arguments are random values in the `ranges`, like for
 * `check_device_view`.
 */
template <typename EosType>
void check_device_view_bounds(
    const EosType& eos,
    const std::array<std::pair<double, double>, EosType::thermodynamic_dim>&
        ranges) {
  constexpr size_t dim = EosType::thermodynamic_dim;
  const auto device_view = eos.device_view();
  constexpr bool with_temperature =
      requires { device_view.temperature_lower_bound(); };
  MAKE_GENERATOR(generator);
  Kokkos::Array<double, dim> args{};
  for (size_t d = 0; d < dim; ++d) {
    std::uniform_real_distribution<> distribution(gsl::at(ranges, d).first,
                                                  gsl::at(ranges, d).second);
    args[d] = distribution(generator);
  }
  const Kokkos::View<double*> results("bounds", 8);
  Kokkos::parallel_for(
      "check_device_view_bounds", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int /*unused*/) {
        const auto bounds =
            DeviceView_detail::evaluate_bounds<dim, with_temperature>(
                device_view, args);
        for (size_t i = 0; i < 8; ++i) {
          results(i) = bounds[i];
        }
      });
  const auto host_results =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
  const auto expected =
      DeviceView_detail::evaluate_bounds<dim, with_temperature>(eos, args);
  for (size_t i = 0; i < 8; ++i) {
    CAPTURE(i);
    CHECK(host_results(i) == approx(expected[i]));
  }
}
}  // namespace TestHelpers::EquationsOfState
#endif  // SPECTRE_KOKKOS
