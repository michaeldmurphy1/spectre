// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PiecewisePolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace EquationsOfState {

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquilEos>,
                                     Barotropic3D<ColdEquilEos>, double, 3)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquilEos>,
                                     Barotropic3D<ColdEquilEos>, DataVector, 3)

template <typename ColdEquilEos>
std::unique_ptr<EquationOfState<ColdEquilEos::is_relativistic, 3>>
Barotropic3D<ColdEquilEos>::get_clone() const {
  auto clone = std::make_unique<Barotropic3D<ColdEquilEos>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 3>>(std::move(clone));
}

template <typename ColdEquilEos>
bool Barotropic3D<ColdEquilEos>::is_equal(
    const EquationOfState<ColdEquilEos::is_relativistic, 3>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Barotropic3D<ColdEquilEos>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename ColdEquilEos>
void Barotropic3D<ColdEquilEos>::pup(PUP::er& p) {
  EquationOfState<ColdEquilEos::is_relativistic, 3>::pup(p);
  p | underlying_eos_;
}
template <typename ColdEquilEos>
Barotropic3D<ColdEquilEos>::Barotropic3D(CkMigrateMessage* msg)
    : EquationOfState<ColdEquilEos::is_relativistic, 3>(msg) {}

template <typename ColdEquilEos>
bool Barotropic3D<ColdEquilEos>::operator==(
    const Barotropic3D<ColdEquilEos>& rhs) const {
  return this->underlying_eos_ == rhs.underlying_eos_;
}
template <typename ColdEquilEos>
bool Barotropic3D<ColdEquilEos>::operator!=(
    const Barotropic3D<ColdEquilEos>& rhs) const {
  return not(*this == rhs);
}

EQUATION_OF_STATE_DELEGATE_IMPLS(template <typename ColdEquilEos>,
                                 Barotropic3D<ColdEquilEos>, 3, functions())

template class Barotropic3D<PolytropicFluid<true>>;
template class Barotropic3D<PolytropicFluid<false>>;
template class Barotropic3D<PiecewisePolytropicFluid<true>>;
template class Barotropic3D<PiecewisePolytropicFluid<false>>;
template class Barotropic3D<Spectral>;
template class Barotropic3D<Enthalpy<PolytropicFluid<true>>>;
template class Barotropic3D<Enthalpy<Enthalpy<PolytropicFluid<true>>>>;
template class Barotropic3D<
    Enthalpy<Enthalpy<Enthalpy<PolytropicFluid<true>>>>>;
template class Barotropic3D<Enthalpy<Spectral>>;
template class Barotropic3D<Enthalpy<Enthalpy<Spectral>>>;
template class Barotropic3D<Enthalpy<Enthalpy<Enthalpy<Spectral>>>>;

}  // namespace EquationsOfState
