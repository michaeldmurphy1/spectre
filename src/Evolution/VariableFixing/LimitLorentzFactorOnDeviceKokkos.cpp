// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/LimitLorentzFactorOnDeviceKokkos.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/LimitLorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "Utilities/TMPL.hpp"

namespace VariableFixing {
namespace {
struct LimitLorentzFactorTag : db::SimpleTag {
  using type = LimitLorentzFactor;
};

struct LimitLorentzFactorAtGridPoint {
  using return_tags = typename LimitLorentzFactor::return_tags;
  using argument_tags = tmpl::list<hydro::Tags::LorentzFactor<DataVector>,
                                   hydro::Tags::SpatialVelocity<DataVector, 3>,
                                   hydro::Tags::RestMassDensity<DataVector>,
                                   LimitLorentzFactorTag>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<Scalar<double>*> lorentz_factor,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*>
          spatial_velocity,
      const Scalar<double>& unlimited_lorentz_factor,
      const tnsr::I<double, 3, Frame::Inertial>& unlimited_spatial_velocity,
      const Scalar<double>& rest_mass_density,
      const LimitLorentzFactor& limit_lorentz_factor) {
    *lorentz_factor = unlimited_lorentz_factor;
    *spatial_velocity = unlimited_spatial_velocity;
    if (const auto velocity_renorm_factor =
            limit_lorentz_factor.velocity_rescaling(
                get(unlimited_lorentz_factor), get(rest_mass_density));
        velocity_renorm_factor.has_value()) {
      get(*lorentz_factor) = limit_lorentz_factor.lorentz_factor_cap();
      for (size_t d = 0; d < 3; ++d) {
        spatial_velocity->get(d) *= *velocity_renorm_factor;
      }
    }
  }
};
}  // namespace

void limit_lorentz_factor_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<hydro::grmhd_tags<DataVector>>*>
        primitives,
    const LimitLorentzFactor& limit_lorentz_factor) {
  if (not limit_lorentz_factor.enabled()) {
    return;
  }
  using Function = LimitLorentzFactorAtGridPoint;
  // The variables are both arguments and results. This is safe because the
  // arguments at a grid point are read before the results there are written.
  const auto arguments =
      make_device_arguments<hydro::Tags::LorentzFactor<DataVector>,
                            hydro::Tags::SpatialVelocity<DataVector, 3>,
                            hydro::Tags::RestMassDensity<DataVector>,
                            LimitLorentzFactorTag>(
          exec,
          get<::Tags::BatchView<hydro::Tags::LorentzFactor<DataVector>>>(
              *primitives),
          get<::Tags::BatchView<hydro::Tags::SpatialVelocity<DataVector, 3>>>(
              *primitives),
          get<::Tags::BatchView<hydro::Tags::RestMassDensity<DataVector>>>(
              *primitives),
          limit_lorentz_factor);
  apply_pointwise_on_device<Function>(
      exec, primitives, typename Function::return_tags{},
      typename Function::argument_tags{}, arguments);
}
}  // namespace VariableFixing
#endif  // SPECTRE_KOKKOS
