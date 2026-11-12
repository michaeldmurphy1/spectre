// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iomanip>
#include <string>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DeferredPointwiseStatusKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

#ifdef SPECTRE_KOKKOS
namespace VariableFixing {
/// The variables that the errors of `FixToAtmosphere` on the device report,
/// as far as the fixed variables include them
using fix_to_atmosphere_error_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::SpecificInternalEnergy<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, 3>,
               hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
               hydro::Tags::LorentzFactor<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::Temperature<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>>;

/// The snapshot of a `DeferredPointwiseStatus` of
/// `fix_to_atmosphere_on_device`: the variables before fixing
using fix_to_atmosphere_snapshot_tags =
    tmpl::push_back<typename FixToAtmosphere<3>::return_tags,
                    hydro::Tags::ElectronFraction<DataVector>>;

/// @{
/*!
 * \brief Fixes the primitive variables of a batch of elements whose data is in
 * device memory to the atmosphere, like `FixToAtmosphere` does for each
 * element, in a single Kokkos kernel.
 *
 * `FixToAtmosphere::DeviceView::fix_at_point` is applied at every grid point
 * with the `fix_to_atmosphere` view (e.g. `FixToAtmosphere::device_view()`)
 * and the `DeviceView` of the `equation_of_state`. It is an error if the
 * equation of state has no `DeviceView`.
 *
 * Unlike `FixToAtmosphere`, a failed root find of the kappa limiting is never
 * an error, and the variables at grid points where it failed are left
 * unfixed. The first overload returns the number of failed grid points and
 * the element and grid point of the first one (see `PointwiseStatusSummary`),
 * so it waits for the kernel on `exec` to complete. The second records them
 * in `status` (see `DeferredPointwiseStatus`) and only enqueues the kernel.
 * Report a failure with `error_fixing_to_atmosphere_on_device`.
 *
 * The `primitives` are fixed in place (their electron fraction is only read).
 * Nothing is copied between the host and the device.
 */
PointwiseStatusSummary fix_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*> primitives,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>::DeviceView& fix_to_atmosphere);

void fix_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*> primitives,
    gsl::not_null<DeferredPointwiseStatus<fix_to_atmosphere_snapshot_tags>*>
        status,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>::DeviceView& fix_to_atmosphere);
/// @}

namespace FixToAtmosphereOnDevice_detail {
// ERRORs with the variables before fixing at the failed grid point, which
// `point(tmpl::type_<Tag>{})` returns for the tags in `TagsList`. `is_first`
// is whether it is the first failure, else any.
template <typename TagsList, typename Point>
[[noreturn]] void error(const PointwiseStatusSummary& status,
                        const size_t number_of_grid_points,
                        const std::string& element_description,
                        const bool is_first, const Point& point) {
  // The variables before fixing that are available, e.g. only the
  // reconstructed ones on faces
  std::string variables{};
  const auto print = [&point, &variables]<typename Tag>(const std::string& name,
                                                        tmpl::type_<Tag> tag) {
    if constexpr (tmpl::list_contains_v<TagsList, Tag>) {
      // As `ERROR` formats numbers
      variables += MakeString{} << std::setprecision(18) << std::scientific
                                << "\n  " << name << " = " << point(tag);
    } else {
      (void)point;
      (void)variables;
      (void)tag;
    }
  };
  print("rest_mass_density",
        tmpl::type_<hydro::Tags::RestMassDensity<DataVector>>{});
  print("specific_internal_energy",
        tmpl::type_<hydro::Tags::SpecificInternalEnergy<DataVector>>{});
  print("spatial_velocity",
        tmpl::type_<hydro::Tags::SpatialVelocity<DataVector, 3>>{});
  print("lorentz_factor_times_spatial_velocity",
        tmpl::type_<
            hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>{});
  print("lorentz_factor",
        tmpl::type_<hydro::Tags::LorentzFactor<DataVector>>{});
  print("pressure", tmpl::type_<hydro::Tags::Pressure<DataVector>>{});
  print("temperature", tmpl::type_<hydro::Tags::Temperature<DataVector>>{});
  print("electron_fraction",
        tmpl::type_<hydro::Tags::ElectronFraction<DataVector>>{});
  ERROR(
      "The root find of the kappa limiting in FixToAtmosphere failed on the "
      "device at "
      << status.number_of_failures << " of " << number_of_grid_points
      << " grid points. At " << (is_first ? "the first one" : "one of them")
      << ", grid point " << status.first_failed_grid_point << " of "
      << element_description << ", before fixing:" << variables
      << "\nRun on the host for the details of the root find.");
}
}  // namespace FixToAtmosphereOnDevice_detail

/// @{
/*!
 * \brief ERRORs with the variables at a grid point where
 * `fix_to_atmosphere_on_device` (or another function applying
 * `FixToAtmosphere` on the device) failed in the kappa limiting.
 *
 * In the first overload, the `batch` holds the variables passed to that
 * function, which are still unfixed at the failed grid points, and `status` is
 * its result, which must have failures. Only that grid point is copied from
 * the device. In the second, `status` is the `DeferredPointwiseStatus` of that
 * function after its `copy_to_host()` and a fence, and the variables are its
 * snapshot. The `element_description` identifies the element in the error,
 * e.g. its `ElementId`.
 */
template <typename TagsList>
[[noreturn]] void error_fixing_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const PointwiseStatusSummary& status,
    const BatchedVariables<TagsList>& batch,
    const std::string& element_description) {
  ASSERT(status.number_of_failures > 0,
         "There is no failure to report: FixToAtmosphere succeeded on the "
         "device at all grid points.");
  FixToAtmosphereOnDevice_detail::error<TagsList>(
      status, batch.number_of_grid_points() * batch.number_of_elements(),
      element_description, true, [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        return extract_point_to_host(exec, get<::Tags::BatchView<Tag>>(batch),
                                     status.first_failed_grid_point,
                                     status.first_failed_element);
      });
}

template <typename SnapshotTags>
[[noreturn]] void error_fixing_to_atmosphere_on_device(
    const DeferredPointwiseStatus<SnapshotTags>& status,
    const size_t number_of_grid_points,
    const std::string& element_description) {
  ASSERT(status.failed(),
         "There is no failure to report: FixToAtmosphere succeeded on the "
         "device at all grid points.");
  const auto snapshot = status.snapshot();
  FixToAtmosphereOnDevice_detail::error<SnapshotTags>(
      status.summary(), number_of_grid_points, element_description, false,
      [&snapshot]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        return get<::Tags::AtIndex<Tag>>(snapshot);
      });
}
/// @}
}  // namespace VariableFixing
#endif  // SPECTRE_KOKKOS
