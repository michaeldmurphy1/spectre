// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#if __has_include(<Kokkos_Core.hpp>)
#include <Kokkos_Core.hpp>

/// \brief If defined then SpECTRE is using Kokkos
#define SPECTRE_KOKKOS 1

/*!
 * \brief Defined while compiling device code for a GPU backend.
 *
 * GPU compilers compile each source file twice: once for the host and once for
 * the device. This macro is defined only in the device pass, so code that is
 * never called on the device (e.g. explicit instantiations of `KOKKOS_FUNCTION`
 * templates for `DataVector`) can be excluded from device compilation:
 *
 * \code
 * GENERATE_INSTANTIATIONS(INSTANTIATE, (double))
 * #ifndef SPECTRE_KOKKOS_DEVICE_PASS
 * GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))
 * #endif
 * \endcode
 *
 * Only use this for code at namespace scope. Inside functions use Kokkos'
 * `KOKKOS_IF_ON_HOST` and `KOKKOS_IF_ON_DEVICE` instead.
 */
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__) or \
    defined(__SYCL_DEVICE_ONLY__)
#define SPECTRE_KOKKOS_DEVICE_PASS 1
#endif

#else  // #if __has_include(<Kokkos_Core.hpp>)
#define KOKKOS_FUNCTION
#define KOKKOS_INLINE_FUNCTION
// Without Kokkos all code runs on the host. The argument is wrapped in
// parentheses, e.g. `KOKKOS_IF_ON_HOST((ASSERT(...);))`, like in Kokkos.
#define SPECTRE_KOKKOS_STRIP_PARENS(...) __VA_ARGS__
#define KOKKOS_IF_ON_HOST(CODE) {SPECTRE_KOKKOS_STRIP_PARENS CODE}
#define KOKKOS_IF_ON_DEVICE(CODE) \
  {                               \
  }
#endif  // #if __has_include(<Kokkos_Core.hpp>)
