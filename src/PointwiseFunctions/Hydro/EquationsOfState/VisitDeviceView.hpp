// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <typeinfo>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace EquationsOfState {
/*!
 * \brief Calls `f` with the `device_view()` of the concrete type of `eos`, so
 * code that runs in Kokkos kernels can be templated on the concrete EOS.
 *
 * It is an error if the concrete type of `eos` has no `device_view()` (see
 * `has_device_view`). Code that runs on the device never falls back to the EOS
 * on the host, so it is always unambiguous where the EOS is evaluated.
 *
 * Only the registered derived classes of `eos` that have a `device_view()` are
 * checked, so `f` is only instantiated for their `DeviceView`s.
 */
template <bool IsRelativistic, size_t ThermodynamicDim, typename F>
void visit_device_view(
    const EquationOfState<IsRelativistic, ThermodynamicDim>& eos, const F& f) {
  using classes_with_device_view = tmpl::filter<
      typename detail::DerivedClasses<IsRelativistic, ThermodynamicDim>::type,
      has_device_view<tmpl::_1>>;
  bool found = false;
  tmpl::for_each<classes_with_device_view>([&eos, &f, &found](auto class_v) {
    using EosType = tmpl::type_from<decltype(class_v)>;
    if (not found and typeid(eos) == typeid(EosType)) {
      found = true;
      f(static_cast<const EosType&>(eos).device_view());
    }
  });
  if (not found) {
    ERROR("The equation of state "
          << pretty_type::get_runtime_type_name(eos)
          << " has no DeviceView, so it can't be evaluated on the device. "
             "Either add a DeviceView to it or don't use it in Kokkos builds.");
  }
}
}  // namespace EquationsOfState
