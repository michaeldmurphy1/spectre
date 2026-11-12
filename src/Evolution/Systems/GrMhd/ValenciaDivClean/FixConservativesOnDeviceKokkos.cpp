// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Kokkos/ScratchViewKokkos.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
// `FixConservatives::AtGridPoint` with whether to assume a non-zero magnetic
// field given for each element of the batch
struct AtGridPointOfBatch {
  using AtGridPoint = FixConservatives::AtGridPoint;
  using return_tags = AtGridPoint::return_tags;
  using argument_tags = tmpl::push_back<
      tmpl::pop_back<AtGridPoint::argument_tags>,
      ::Tags::PerElement<FixConservatives::AssumeNonZeroMagneticField>>;

  template <typename... Args>
  KOKKOS_FUNCTION static PointwiseStatus apply(const Args&... args) {
    return AtGridPoint::apply(args...);
  }
};

// Whether to assume a non-zero magnetic field on each element, as
// `FixConservatives::assume_non_zero_magnetic_field` decides for an element
Kokkos::View<const bool*> assume_non_zero_magnetic_field_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const typename ::Tags::BatchView<Tags::TildeB<>>::type& tilde_b,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* const scratch) {
  const size_t number_of_grid_points = get<0>(tilde_b).extent(0);
  const size_t number_of_elements = get<0>(tilde_b).extent(1);
  const Kokkos::View<bool*> result = scratch_view(
      exec, scratch, "AssumeNonZeroMagneticField", number_of_elements);
  if (fix_conservatives.magnetic_field_treatment() !=
      hydro::MagneticFieldTreatment::CheckIfZero) {
    Kokkos::deep_copy(exec, result,
                      fix_conservatives.magnetic_field_treatment() ==
                          hydro::MagneticFieldTreatment::AssumeNonZero);
    return result;
  }
  const auto tilde_b_x = get<0>(tilde_b);
  const auto tilde_b_y = get<1>(tilde_b);
  const auto tilde_b_z = get<2>(tilde_b);
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  Kokkos::parallel_for(
      "AssumeNonZeroMagneticField",
      TeamPolicy(exec, static_cast<int>(number_of_elements), Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const auto e = static_cast<size_t>(team.league_rank());
        double max_abs_tilde_b = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, number_of_grid_points),
            [&](const size_t i, double& local_max) {
              local_max = Kokkos::max(
                  local_max,
                  Kokkos::max(Kokkos::abs(tilde_b_x(i, e)),
                              Kokkos::max(Kokkos::abs(tilde_b_y(i, e)),
                                          Kokkos::abs(tilde_b_z(i, e)))));
            },
            Kokkos::Max<double>(max_abs_tilde_b));
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { result(e) = max_abs_tilde_b > 0.0; });
      });
  return result;
}
}  // namespace

namespace {
// Fixes with a `PointwiseStatusSummary` result if `status` is `nullptr`, and
// else records the status in `*status`
template <typename StatusPointer>
PointwiseStatusSummary fix_conservatives_impl(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<FixConservativesOnDeviceTags>*>
        conservatives,
    const StatusPointer status,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* const per_element_scratch) {
  const auto& tilde_b = get<::Tags::BatchView<Tags::TildeB<>>>(*conservatives);
  const Kokkos::View<const bool*> assume_non_zero_magnetic_field =
      assume_non_zero_magnetic_field_on_device(exec, tilde_b, fix_conservatives,
                                               per_element_scratch);
  // The variables are both arguments and results. This is safe because the
  // arguments at a grid point are read before the results there are written.
  return ApplyPointwiseOnDevice_detail::apply_with_status<AtGridPointOfBatch>(
      exec, conservatives, status, AtGridPointOfBatch::return_tags{},
      AtGridPointOfBatch::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, get<::Tags::BatchView<Tags::TildeD>>(*conservatives),
            get<::Tags::BatchView<Tags::TildeYe>>(*conservatives),
            get<::Tags::BatchView<Tags::TildeTau>>(*conservatives),
            get<::Tags::BatchView<Tags::TildeS<>>>(*conservatives), tilde_b,
            spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
            fix_conservatives, assume_non_zero_magnetic_field);
      }(AtGridPointOfBatch::argument_tags{}));
}

// ERRORs with the variables before fixing at the failed grid point, which
// `point(tmpl::type_<Tag>{})` returns. `is_first` is whether it is the first
// failure, else any.
template <typename Point>
[[noreturn]] void error_fixing_conservatives(
    const PointwiseStatusSummary& status, const size_t number_of_grid_points,
    const std::string& element_description, const bool is_first,
    const Point& point) {
  ERROR(
      "Failed to fix conserved variables on the device because the root "
      "finder failed to find the lorentz factor at "
      << status.number_of_failures << " of " << number_of_grid_points
      << " grid points. At " << (is_first ? "the first one" : "one of them")
      << ", grid point " << status.first_failed_grid_point << " of "
      << element_description
      << ":\n  tilde_d = " << point(tmpl::type_<Tags::TildeD>{})
      << "\n  tilde_ye = " << point(tmpl::type_<Tags::TildeYe>{})
      << "\n  tilde_tau = " << point(tmpl::type_<Tags::TildeTau>{})
      << "\n  tilde_s =\n"
      << point(tmpl::type_<Tags::TildeS<>>{}) << "\n  tilde_b =\n"
      << point(tmpl::type_<Tags::TildeB<>>{}) << "\n  spatial_metric =\n"
      << point(tmpl::type_<gr::Tags::SpatialMetric<DataVector, 3>>{})
      << "\n  inv_spatial_metric =\n"
      << point(tmpl::type_<gr::Tags::InverseSpatialMetric<DataVector, 3>>{})
      << "\n  sqrt_det_spatial_metric = "
      << point(tmpl::type_<gr::Tags::SqrtDetSpatialMetric<DataVector>>{})
      << "\nRun on the host for the details of the root find.");
}
}  // namespace

PointwiseStatusSummary fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<FixConservativesOnDeviceTags>*>
        conservatives,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* const per_element_scratch) {
  return fix_conservatives_impl(exec, conservatives, nullptr, spatial_metric,
                                inv_spatial_metric, sqrt_det_spatial_metric,
                                fix_conservatives, per_element_scratch);
}

void fix_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<FixConservativesOnDeviceTags>*>
        conservatives,
    const gsl::not_null<
        DeferredPointwiseStatus<fix_conservatives_snapshot_tags>*>
        status,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const FixConservatives& fix_conservatives,
    Kokkos::View<bool*>* const per_element_scratch) {
  fix_conservatives_impl(exec, conservatives, status.get(), spatial_metric,
                         inv_spatial_metric, sqrt_det_spatial_metric,
                         fix_conservatives, per_element_scratch);
}

void error_fixing_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const PointwiseStatusSummary& status,
    const BatchedVariables<FixConservativesOnDeviceTags>& conservatives,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::InverseSpatialMetric<DataVector, 3>>::type&
        inv_spatial_metric,
    const typename ::Tags::BatchView<
        gr::Tags::SqrtDetSpatialMetric<DataVector>>::type&
        sqrt_det_spatial_metric,
    const std::string& element_description) {
  ASSERT(status.number_of_failures > 0,
         "There is no failure to report: fix_conservatives_on_device "
         "succeeded at all grid points.");
  const size_t i = status.first_failed_grid_point;
  const size_t e = status.first_failed_element;
  error_fixing_conservatives(
      status,
      conservatives.number_of_grid_points() *
          conservatives.number_of_elements(),
      element_description, true, [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const auto& batch_tensor = [&]() -> const auto& {
          if constexpr (std::is_same_v<
                            Tag, gr::Tags::SpatialMetric<DataVector, 3>>) {
            return spatial_metric;
          } else if constexpr (std::is_same_v<Tag,
                                              gr::Tags::InverseSpatialMetric<
                                                  DataVector, 3>>) {
            return inv_spatial_metric;
          } else if constexpr (std::is_same_v<Tag,
                                              gr::Tags::SqrtDetSpatialMetric<
                                                  DataVector>>) {
            return sqrt_det_spatial_metric;
          } else {
            return get<::Tags::BatchView<Tag>>(conservatives);
          }
        }();
        return extract_point_to_host(exec, batch_tensor, i, e);
      });
}

void error_fixing_conservatives_on_device(
    const DeferredPointwiseStatus<fix_conservatives_snapshot_tags>& status,
    const size_t number_of_grid_points,
    const std::string& element_description) {
  ASSERT(status.failed(),
         "There is no failure to report: fix_conservatives_on_device "
         "succeeded at all grid points.");
  const auto snapshot = status.snapshot();
  error_fixing_conservatives(
      status.summary(), number_of_grid_points, element_description, false,
      [&snapshot]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        return get<::Tags::AtIndex<Tag>>(snapshot);
      });
}
}  // namespace grmhd::ValenciaDivClean
#endif  // SPECTRE_KOKKOS
