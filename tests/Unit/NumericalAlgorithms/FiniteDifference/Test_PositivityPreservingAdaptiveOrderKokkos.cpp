// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/Minmod.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrderKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
using ::fd::device::StridedVolumeView;
using ::fd::device::VolumeView;
using ::fd::reconstruction::FallbackReconstructorType;
namespace helpers = TestHelpers::fd::device;

constexpr size_t number_of_elements = 2;
constexpr size_t number_of_variables = 2;

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

struct Options {
  bool positivity_preserving;
  FallbackReconstructorType low_order_reconstructor;
  bool use_7th_order;
  bool use_9th_order;

  // The host reconstruction takes ghost data of exactly this width
  size_t ghost_width() const {
    return use_9th_order ? 5 : (use_7th_order ? 4 : 3);
  }
};

// The host reconstruction of one element with the `options`
template <typename LowOrder, bool PositivityPreserving, bool Use9thOrder,
          bool Use7thOrder, size_t Dim>
void reconstruct_on_host(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*> upper,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*> lower,
    const helpers::ElementData<Dim>& data, const Index<Dim>& volume_extents,
    const std::array<double, 3>& alphas) {
  ::fd::reconstruction::positivity_preserving_adaptive_order<
      LowOrder, PositivityPreserving, Use9thOrder, Use7thOrder>(
      upper, lower,
      gsl::make_span(data.volume_vars.data(), data.volume_vars.size()),
      helpers::make_spans(data.ghost_data), volume_extents, number_of_variables,
      alphas[0], alphas[1], alphas[2]);
}

template <size_t Dim, typename... Args>
void dispatch_on_host(const Options& options, const Args&... args) {
  const auto with_orders = [&]<bool PositivityPreserving, typename LowOrder>() {
    if (options.use_7th_order) {
      if (options.use_9th_order) {
        reconstruct_on_host<LowOrder, PositivityPreserving, true, true, Dim>(
            args...);
      } else {
        reconstruct_on_host<LowOrder, PositivityPreserving, false, true, Dim>(
            args...);
      }
    } else {
      if (options.use_9th_order) {
        reconstruct_on_host<LowOrder, PositivityPreserving, true, false, Dim>(
            args...);
      } else {
        reconstruct_on_host<LowOrder, PositivityPreserving, false, false, Dim>(
            args...);
      }
    }
  };
  const auto with_low_order = [&]<bool PositivityPreserving>() {
    if (options.low_order_reconstructor == FallbackReconstructorType::Minmod) {
      with_orders.template
      operator()<PositivityPreserving,
                 ::fd::reconstruction::detail::MinmodReconstructor>();
    } else {
      with_orders.template operator()<
          PositivityPreserving,
          ::fd::reconstruction::detail::MonotonisedCentralReconstructor>();
    }
  };
  if (options.positivity_preserving) {
    with_low_order.template operator()<true>();
  } else {
    with_low_order.template operator()<false>();
  }
}

// Compare to `fd::reconstruction::positivity_preserving_adaptive_order`
template <size_t Dim>
void test_compare_to_host(const std::vector<helpers::ElementData<Dim>>& data,
                          const Index<Dim>& volume_extents,
                          const Options& options) {
  CAPTURE(options.positivity_preserving);
  CAPTURE(options.low_order_reconstructor);
  CAPTURE(options.use_7th_order);
  CAPTURE(options.use_9th_order);
  // 4^alpha_5, 6^alpha_7 and 8^alpha_9
  const std::array<double, 3> alphas{
      {pow(4.0, 4.0), pow(6.0, 4.0), pow(8.0, 4.0)}};
  const Kokkos::DefaultExecutionSpace exec{};
  const auto padded = helpers::pack(exec, data, number_of_variables,
                                    volume_extents, options.ghost_width());
  const auto upper_side = make_face_views("upper side", volume_extents);
  const auto lower_side = make_face_views("lower side", volume_extents);
  ::fd::device::positivity_preserving_adaptive_order(
      exec, strided(upper_side), strided(lower_side), padded, volume_extents,
      options.positivity_preserving, options.low_order_reconstructor, alphas[0],
      options.use_7th_order ? std::optional{alphas[1]} : std::nullopt,
      options.use_9th_order ? std::optional{alphas[2]} : std::nullopt);
  const auto upper = helpers::to_host(upper_side);
  const auto lower = helpers::to_host(lower_side);

  for (size_t element = 0; element < number_of_elements; ++element) {
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
    dispatch_on_host<Dim>(options, make_not_null(&expected_upper_spans),
                          make_not_null(&expected_lower_spans), data[element],
                          volume_extents, alphas);
    for (size_t d = 0; d < Dim; ++d) {
      CAPTURE(element);
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(helpers::element_data(gsl::at(upper, d), element),
                            gsl::at(expected_upper, d));
      CHECK_ITERABLE_APPROX(helpers::element_data(gsl::at(lower, d), element),
                            gsl::at(expected_lower, d));
    }
  }
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator,
          const Index<Dim>& volume_extents) {
  for (const bool positivity_preserving : {true, false}) {
    for (const auto low_order :
         {FallbackReconstructorType::Minmod,
          FallbackReconstructorType::MonotonisedCentral}) {
      for (const bool use_7th_order : {true, false}) {
        for (const bool use_9th_order : {true, false}) {
          const Options options{positivity_preserving, low_order, use_7th_order,
                                use_9th_order};
          // Rough data that changes sign, so the lower orders and the
          // positivity fallbacks are used, and nearly constant data, so the
          // higher orders are used
          test_compare_to_host(
              helpers::make_random_element_data(
                  generator, number_of_elements, number_of_variables,
                  volume_extents, options.ghost_width()),
              volume_extents, options);
          std::uniform_real_distribution<double> small(-1.0e-3, 1.0e-3);
          test_compare_to_host(
              helpers::make_element_data(
                  number_of_elements, number_of_variables, volume_extents,
                  options.ghost_width(),
                  [&generator, &small](const size_t /*element*/,
                                       const size_t /*var*/,
                                       const std::array<int, Dim>& /*cell*/) {
                    return 1.0 + small(*generator);
                  }),
              volume_extents, options);
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.FiniteDifference."
    "PositivityPreservingAdaptiveOrder.Kokkos",
    "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  // The host needs isotropic extents of at least `stencil_width - 1` cells,
  // i.e. 8 for the 9th order
  test(make_not_null(&generator), Index<1>{9});
  test(make_not_null(&generator), Index<2>{9, 9});
  test(make_not_null(&generator), Index<3>{9, 9, 9});
}
