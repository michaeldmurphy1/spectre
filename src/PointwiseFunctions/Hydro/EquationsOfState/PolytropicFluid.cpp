// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"

#include <cmath>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace EquationsOfState {
template <bool IsRelativistic>
PolytropicFluid<IsRelativistic>::PolytropicFluid(
    const double polytropic_constant, const double polytropic_exponent)
    : polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     PolytropicFluid<IsRelativistic>, double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     PolytropicFluid<IsRelativistic>,
                                     DataVector, 1)

template <bool IsRelativistic>
bool PolytropicFluid<IsRelativistic>::operator==(
    const PolytropicFluid<IsRelativistic>& rhs) const {
  return (polytropic_constant_ == rhs.polytropic_constant_) and
         (polytropic_exponent_ == rhs.polytropic_exponent_);
}

template <bool IsRelativistic>
bool PolytropicFluid<IsRelativistic>::operator!=(
    const PolytropicFluid<IsRelativistic>& rhs) const {
  return not(*this == rhs);
}

template <bool IsRelativistic>
bool PolytropicFluid<IsRelativistic>::is_equal(
    const EquationOfState<IsRelativistic, 1>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const PolytropicFluid<IsRelativistic>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 1>>
PolytropicFluid<IsRelativistic>::get_clone() const {
  auto clone = std::make_unique<PolytropicFluid<IsRelativistic>>(*this);
  return std::unique_ptr<EquationOfState<IsRelativistic, 1>>(std::move(clone));
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 3>>
PolytropicFluid<IsRelativistic>::promote_to_3d_eos() const {
  return std::make_unique<Barotropic3D<PolytropicFluid<IsRelativistic>>>(*this);
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 2>>
PolytropicFluid<IsRelativistic>::promote_to_2d_eos() const {
  return std::make_unique<Barotropic2D<PolytropicFluid<IsRelativistic>>>(*this);
}

template <bool IsRelativistic>
PolytropicFluid<IsRelativistic>::PolytropicFluid(CkMigrateMessage* msg)
    : EquationOfState<IsRelativistic, 1>(msg) {}

template <bool IsRelativistic>
void PolytropicFluid<IsRelativistic>::pup(PUP::er& p) {
  EquationOfState<IsRelativistic, 1>::pup(p);
  p | polytropic_constant_;
  p | polytropic_exponent_;
}

EQUATION_OF_STATE_DELEGATE_IMPLS(template <bool IsRelativistic>,
                                 PolytropicFluid<IsRelativistic>, 1,
                                 device_view())
}  // namespace EquationsOfState

template class EquationsOfState::PolytropicFluid<true>;
template class EquationsOfState::PolytropicFluid<false>;
