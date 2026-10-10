// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace EquationsOfState {

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename EquilEos>,
                                     Equilibrium3D<EquilEos>, double, 3)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename EquilEos>,
                                     Equilibrium3D<EquilEos>, DataVector, 3)
template <typename EquilEos>
void Equilibrium3D<EquilEos>::pup(PUP::er& p) {
  EquationOfState<EquilEos::is_relativistic, 3>::pup(p);
  p | underlying_eos_;
}
template <typename EquilEos>
Equilibrium3D<EquilEos>::Equilibrium3D(CkMigrateMessage* msg)
    : EquationOfState<EquilEos::is_relativistic, 3>(msg) {}

template <typename EquilEos>
std::unique_ptr<EquationOfState<EquilEos::is_relativistic, 3>>
Equilibrium3D<EquilEos>::get_clone() const {
  auto clone = std::make_unique<Equilibrium3D<EquilEos>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 3>>(std::move(clone));
}

template <typename EquilEos>
bool Equilibrium3D<EquilEos>::is_equal(
    const EquationOfState<EquilEos::is_relativistic, 3>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Equilibrium3D<EquilEos>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename EquilEos>
bool Equilibrium3D<EquilEos>::operator==(
    const Equilibrium3D<EquilEos>& rhs) const {
  return this->underlying_eos_ == rhs.underlying_eos_;
}
template <typename EquilEos>
bool Equilibrium3D<EquilEos>::operator!=(
    const Equilibrium3D<EquilEos>& rhs) const {
  return !(*this == rhs);
}

EQUATION_OF_STATE_DELEGATE_IMPLS(template <typename EquilEos>,
                                 Equilibrium3D<EquilEos>, 3, functions())

template class Equilibrium3D<IdealFluid<true>>;
template class Equilibrium3D<IdealFluid<false>>;
template class Equilibrium3D<HybridEos<PolytropicFluid<true>>>;
template class Equilibrium3D<HybridEos<PolytropicFluid<false>>>;
template class Equilibrium3D<HybridEos<Spectral>>;
template class Equilibrium3D<HybridEos<Enthalpy<PolytropicFluid<true>>>>;
template class Equilibrium3D<
    HybridEos<Enthalpy<Enthalpy<Enthalpy<PolytropicFluid<true>>>>>>;
template class Equilibrium3D<HybridEos<Enthalpy<Spectral>>>;
template class Equilibrium3D<HybridEos<Enthalpy<Enthalpy<Spectral>>>>;
template class Equilibrium3D<HybridEos<Enthalpy<Enthalpy<Enthalpy<Spectral>>>>>;

}  // namespace EquationsOfState
