// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/FluxDivergenceKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
using ::fd::device::VolumeView;
namespace helpers = TestHelpers::fd::device;

constexpr size_t number_of_elements = 2;
constexpr size_t number_of_variables = 3;

// A device view with the data of each element in the layout of a `Variables`
template <size_t Dim>
VolumeView to_device(const std::vector<DataVector>& data,
                     const size_t number_of_vars, const Index<Dim>& extents) {
  const Kokkos::DefaultExecutionSpace exec{};
  const auto result = ::fd::device::make_volume_view(exec, "data", data.size(),
                                                     number_of_vars, extents);
  const auto host_result = Kokkos::create_mirror_view(result);
  exec.fence();
  for (size_t element = 0; element < data.size(); ++element) {
    REQUIRE(data[element].size() == number_of_vars * extents.product());
    std::copy(data[element].begin(), data[element].end(),
              &host_result(element, 0, 0, 0, 0));
  }
  Kokkos::deep_copy(exec, result, host_result);
  exec.fence();
  return result;
}

// Compare to `evolution::dg::subcell::add_cartesian_flux_divergence` with
// random data
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Dim);
  const Index<Dim> extents(5);
  const size_t num_pts = extents.product();
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto random_data = [&](const size_t size) {
    return make_with_random_values<DataVector>(generator, make_not_null(&dist),
                                               DataVector(size));
  };
  std::array<double, Dim> one_over_delta{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(one_over_delta, d) = 2.0 + static_cast<double>(d);
  }

  std::vector<DataVector> dt_vars(number_of_elements);
  std::vector<DataVector> det_inv_jacobian(number_of_elements);
  std::array<std::vector<DataVector>, Dim> corrections{};
  auto face_extents = make_array<Dim>(extents);
  for (size_t d = 0; d < Dim; ++d) {
    ++gsl::at(face_extents, d)[d];
  }
  for (size_t element = 0; element < number_of_elements; ++element) {
    dt_vars[element] = random_data(number_of_variables * num_pts);
    det_inv_jacobian[element] = random_data(num_pts);
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(corrections, d)
          .push_back(random_data(number_of_variables *
                                 gsl::at(face_extents, d).product()));
    }
  }

  const VolumeView device_dt_vars =
      to_device(dt_vars, number_of_variables, extents);
  std::array<::fd::device::ConstVolumeView, Dim> device_corrections{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(device_corrections, d) = to_device(
        gsl::at(corrections, d), number_of_variables, gsl::at(face_extents, d));
  }
  ::fd::device::add_cartesian_flux_divergence(
      Kokkos::DefaultExecutionSpace{}, device_dt_vars, device_corrections,
      to_device(det_inv_jacobian, 1, extents), one_over_delta, extents);
  const auto host_dt_vars =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_dt_vars);

  for (size_t element = 0; element < number_of_elements; ++element) {
    CAPTURE(element);
    DataVector expected = dt_vars[element];
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t var = 0; var < number_of_variables; ++var) {
        DataVector expected_var(&expected[var * num_pts], num_pts);
        const DataVector correction_var(
            &gsl::at(corrections,
                     d)[element][var * gsl::at(face_extents, d).product()],
            gsl::at(face_extents, d).product());
        evolution::dg::subcell::add_cartesian_flux_divergence(
            make_not_null(&expected_var), gsl::at(one_over_delta, d),
            det_inv_jacobian[element], correction_var, extents, d);
      }
    }
    CHECK_ITERABLE_APPROX(helpers::element_data(host_dt_vars, element),
                          expected);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.FluxDivergence.Kokkos",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
