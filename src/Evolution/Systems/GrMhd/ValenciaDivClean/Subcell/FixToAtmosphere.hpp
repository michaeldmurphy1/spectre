// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Fix the primitive variables to the atmosphere after an FD step.
 *
 * Applies the same fixes as
 * `VariableFixing::Actions::FixVariables<VariableFixing::FixToAtmosphere<3>>`,
 * but with `VariableFixing::fix_to_atmosphere_on_host_or_device`, so in Kokkos
 * builds the variables are fixed on the device.
 */
struct FixToAtmosphere {
  using fixer = VariableFixing::FixToAtmosphere<3>;
  using return_tags = typename fixer::return_tags;
  using argument_tags = tmpl::push_back<typename fixer::argument_tags,
                                        ::Tags::VariableFixer<fixer>>;
  using const_global_cache_tags = tmpl::list<::Tags::VariableFixer<fixer>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& electron_fraction,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const fixer& fix_to_atmosphere);
};
}  // namespace grmhd::ValenciaDivClean::subcell
