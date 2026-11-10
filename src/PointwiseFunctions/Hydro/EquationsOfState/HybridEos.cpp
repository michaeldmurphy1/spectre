// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace EquationsOfState {
template <typename ColdEquationOfState>
HybridEos<ColdEquationOfState>::HybridEos(ColdEquationOfState cold_eos,
                                          const double thermal_adiabatic_index,
                                          const double min_temperature)
    : cold_eos_(std::move(cold_eos)),
      thermal_adiabatic_index_(thermal_adiabatic_index),
      min_temperature_(min_temperature) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     HybridEos<ColdEquationOfState>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     HybridEos<ColdEquationOfState>, DataVector,
                                     2)

template <typename ColdEquationOfState>
std::unique_ptr<
    EquationOfState<HybridEos<ColdEquationOfState>::is_relativistic, 2>>
HybridEos<ColdEquationOfState>::get_clone() const {
  auto clone = std::make_unique<HybridEos<ColdEquationOfState>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 2>>(std::move(clone));
}

template <typename ColdEquationOfState>
std::unique_ptr<
    EquationOfState<HybridEos<ColdEquationOfState>::is_relativistic, 3>>
HybridEos<ColdEquationOfState>::promote_to_3d_eos() const {
  return std::make_unique<Equilibrium3D<HybridEos<ColdEquationOfState>>>(
      Equilibrium3D(*this));
}

template <typename ColdEquationOfState>
bool HybridEos<ColdEquationOfState>::operator==(
    const HybridEos<ColdEquationOfState>& rhs) const {
  return cold_eos_ == rhs.cold_eos_ and
         thermal_adiabatic_index_ == rhs.thermal_adiabatic_index_ and
         min_temperature_ == rhs.min_temperature_;
}

template <typename ColdEquationOfState>
bool HybridEos<ColdEquationOfState>::operator!=(
    const HybridEos<ColdEquationOfState>& rhs) const {
  return not(*this == rhs);
}

template <typename ColdEquationOfState>
bool HybridEos<ColdEquationOfState>::is_equal(
    const EquationOfState<is_relativistic, 2>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const HybridEos<ColdEquationOfState>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename ColdEquationOfState>
HybridEos<ColdEquationOfState>::HybridEos(CkMigrateMessage* msg)
    : EquationOfState<is_relativistic, 2>(msg) {}

template <typename ColdEquationOfState>
void HybridEos<ColdEquationOfState>::pup(PUP::er& p) {
  EquationOfState<is_relativistic, 2>::pup(p);
  p | cold_eos_;
  p | thermal_adiabatic_index_;
  p | min_temperature_;
}

EQUATION_OF_STATE_DELEGATE_IMPLS(template <typename ColdEquationOfState>,
                                 HybridEos<ColdEquationOfState>, 2, functions())
}  // namespace EquationsOfState

template class EquationsOfState::HybridEos<
    EquationsOfState::PolytropicFluid<true>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::PolytropicFluid<false>>;
template class EquationsOfState::HybridEos<EquationsOfState::Spectral>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::PolytropicFluid<true>>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::Enthalpy<
        EquationsOfState::Enthalpy<EquationsOfState::PolytropicFluid<true>>>>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::Spectral>>;
template class EquationsOfState::HybridEos<EquationsOfState::Enthalpy<
    EquationsOfState::Enthalpy<EquationsOfState::Spectral>>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::Enthalpy<
        EquationsOfState::Enthalpy<EquationsOfState::Spectral>>>>;
