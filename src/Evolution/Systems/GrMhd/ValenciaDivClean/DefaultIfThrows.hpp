// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <exception>

#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::
    DefaultIfThrows_detail {
// Returns `f()`, or a default-constructed result (e.g. `std::nullopt`) if `f`
// throws. The EOS can throw on the host, e.g. if a root find inside it fails,
// and the recovery schemes treat that as a failure, so the next scheme is
// tried. Exceptions aren't supported on the device, so there `f()` is
// returned.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <typename F>
KOKKOS_FUNCTION auto default_if_throws(const F& f) -> decltype(f()) {
  KOKKOS_IF_ON_HOST((
      try { return f(); } catch (const std::exception& /*exception*/) {
        return decltype(f()){};
      }))
  KOKKOS_IF_ON_DEVICE((return f();))
}
}  // namespace
   // grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::DefaultIfThrows_detail
