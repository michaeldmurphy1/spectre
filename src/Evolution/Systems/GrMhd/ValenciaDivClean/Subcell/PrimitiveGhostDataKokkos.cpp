// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"

#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"

namespace grmhd::ValenciaDivClean::subcell {
namespace {
// The ghost variables at a grid point, computed like
// `PrimitiveGhostVariables::apply`
struct GhostVariablesAtGridPoint {
  using return_tags = PrimitiveGhostVariables::ghost_variables_tag_list;
  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> ghost_rest_mass_density,
      const gsl::not_null<Scalar<double>*> ghost_electron_fraction,
      const gsl::not_null<Scalar<double>*> ghost_temperature,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          ghost_magnetic_field,
      const gsl::not_null<Scalar<double>*> ghost_divergence_cleaning_field,
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& electron_fraction,
      const Scalar<double>& temperature,
      const tnsr::I<double, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<double>& lorentz_factor,
      const tnsr::I<double, 3, Frame::Inertial>& magnetic_field,
      const Scalar<double>& divergence_cleaning_field) {
    *ghost_rest_mass_density = rest_mass_density;
    *ghost_electron_fraction = electron_fraction;
    *ghost_temperature = temperature;
    for (size_t i = 0; i < 3; ++i) {
      lorentz_factor_times_spatial_velocity->get(i) =
          get(lorentz_factor) * spatial_velocity.get(i);
    }
    *ghost_magnetic_field = magnetic_field;
    *ghost_divergence_cleaning_field = divergence_cleaning_field;
  }
};
}  // namespace

void primitive_ghost_variables_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<
        BatchedVariables<PrimitiveGhostVariables::ghost_variables_tag_list>*>
        ghost_variables,
    const BatchedVariables<hydro::grmhd_tags<DataVector>>& primitives) {
  apply_pointwise_on_device<GhostVariablesAtGridPoint>(
      exec, ghost_variables, GhostVariablesAtGridPoint::return_tags{},
      GhostVariablesAtGridPoint::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, get<::Tags::BatchView<ArgumentTags>>(primitives)...);
      }(GhostVariablesAtGridPoint::argument_tags{}));
}
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
