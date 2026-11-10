// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iomanip>
#include <string>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
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
 * unfixed. This returns the number of failed grid points and the element and
 * grid point of the first one (see `PointwiseStatusSummary`). Report a failure
 * with `error_fixing_to_atmosphere_on_device`.
 *
 * The `primitives` are fixed in place (their electron fraction is only read).
 * Nothing is copied between the host and the device, but since the failures
 * are needed on the host, this function waits for the kernel on `exec` to
 * complete.
 */
PointwiseStatusSummary fix_to_atmosphere_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*> primitives,
    const typename ::Tags::BatchView<
        gr::Tags::SpatialMetric<DataVector, 3>>::type& spatial_metric,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const FixToAtmosphere<3>::DeviceView& fix_to_atmosphere);

/*!
 * \brief ERRORs with the variables at the first grid point where
 * `fix_to_atmosphere_on_device` (or another function applying
 * `FixToAtmosphere` on the device) failed in the kappa limiting.
 *
 * The `batch` holds the variables passed to that function, which are still
 * unfixed at the failed grid points, and `status` is its result, which must
 * have failures. Only that grid point is copied from the device. The
 * `element_description` identifies the element in the error, e.g. its
 * `ElementId`.
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
  const size_t i = status.first_failed_grid_point;
  const size_t e = status.first_failed_element;
  // The variables before fixing that the batch holds, e.g. only the
  // reconstructed ones on faces
  std::string variables{};
  const auto print = [&exec, &batch, i, e, &variables]<typename Tag>(
                         const std::string& name, tmpl::type_<Tag> /*meta*/) {
    if constexpr (tmpl::list_contains_v<TagsList, Tag>) {
      // As `ERROR` formats numbers
      variables += MakeString{}
                   << std::setprecision(18) << std::scientific << "\n  " << name
                   << " = "
                   << extract_point_to_host(
                          exec, get<::Tags::BatchView<Tag>>(batch), i, e);
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
      << status.number_of_failures << " of "
      << batch.number_of_grid_points() * batch.number_of_elements()
      << " grid points. At the first one, grid point " << i << " of "
      << element_description << ", before fixing:" << variables
      << "\nRun on the host for the details of the root find.");
}
}  // namespace VariableFixing
#endif  // SPECTRE_KOKKOS
