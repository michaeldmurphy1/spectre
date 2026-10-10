// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

#include <cmath>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace EquationsOfState {
template <bool IsRelativistic>
IdealFluid<IsRelativistic>::IdealFluid(const double adiabatic_index,
                                       const double min_temperature)
    : adiabatic_index_(adiabatic_index), min_temperature_(min_temperature) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     IdealFluid<IsRelativistic>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     IdealFluid<IsRelativistic>, DataVector, 2)
template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 2>>
IdealFluid<IsRelativistic>::get_clone() const {
  auto clone = std::make_unique<IdealFluid<IsRelativistic>>(*this);
  return std::unique_ptr<EquationOfState<IsRelativistic, 2>>(std::move(clone));
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 3>>
IdealFluid<IsRelativistic>::promote_to_3d_eos() const {
  return std::make_unique<Equilibrium3D<IdealFluid<IsRelativistic>>>(
      Equilibrium3D(*this));
}

template <bool IsRelativistic>
bool IdealFluid<IsRelativistic>::is_equal(
    const EquationOfState<IsRelativistic, 2>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const IdealFluid<IsRelativistic>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <bool IsRelativistic>
bool IdealFluid<IsRelativistic>::operator==(
    const IdealFluid<IsRelativistic>& rhs) const {
  return (adiabatic_index_ == rhs.adiabatic_index_) and
         (min_temperature_ == rhs.min_temperature_);
}

template <bool IsRelativistic>
bool IdealFluid<IsRelativistic>::operator!=(
    const IdealFluid<IsRelativistic>& rhs) const {
  return not(*this == rhs);
}

template <bool IsRelativistic>
IdealFluid<IsRelativistic>::IdealFluid(CkMigrateMessage* msg)
    : EquationOfState<IsRelativistic, 2>(msg) {}

template <bool IsRelativistic>
void IdealFluid<IsRelativistic>::pup(PUP::er& p) {
  EquationOfState<IsRelativistic, 2>::pup(p);
  p | adiabatic_index_;
  p | min_temperature_;
}

EQUATION_OF_STATE_DELEGATE_IMPLS(template <bool IsRelativistic>,
                                 IdealFluid<IsRelativistic>, 2, device_view())

template <bool IsRelativistic>
double IdealFluid<IsRelativistic>::specific_internal_energy_upper_bound(
    const double /* rest_mass_density */) const {
  // this bound comes from the dominant energy condition which implies
  // that the pressure is bounded by the total energy density,
  // i.e. p < e = rho * (1 + eps)
  if (IsRelativistic and adiabatic_index_ > 2.0) {
    return 1.0 / (adiabatic_index_ - 2.0);
  }
  return std::numeric_limits<double>::max();
}

}  // namespace EquationsOfState

template class EquationsOfState::IdealFluid<true>;
template class EquationsOfState::IdealFluid<false>;
