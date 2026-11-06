// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/DeviceMessage.hpp"

#include <cstdio>

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Printf.hpp>

#include "Utilities/Kokkos/KokkosCore.hpp"
#endif  // SPECTRE_KOKKOS

// A macro instead of a function so the format strings stay literals, which
// `-Wformat-nonliteral` requires
#ifdef SPECTRE_KOKKOS
#define SPECTRE_DEVICE_MESSAGE_PRINTF(...) ::Kokkos::printf(__VA_ARGS__)
#else
#define SPECTRE_DEVICE_MESSAGE_PRINTF(...) std::printf(__VA_ARGS__)
#endif  // SPECTRE_KOKKOS

void DeviceMessage::print_character(const char character) {
  SPECTRE_DEVICE_MESSAGE_PRINTF("%c", character);
}

void DeviceMessage::print_string(const char* const str) {
  SPECTRE_DEVICE_MESSAGE_PRINTF("%s", str == nullptr ? "(null)" : str);
}

void DeviceMessage::print_signed(const long long value) {
  SPECTRE_DEVICE_MESSAGE_PRINTF("%lld", value);
}

void DeviceMessage::print_unsigned(const unsigned long long value) {
  SPECTRE_DEVICE_MESSAGE_PRINTF("%llu", value);
}

void DeviceMessage::print_floating_point(const double value) {
  // Same as `std::scientific` with `std::setprecision(18)`
  SPECTRE_DEVICE_MESSAGE_PRINTF("%.18e", value);
}

#ifdef SPECTRE_KOKKOS
namespace DeviceMessage_detail {
namespace {
// The function-local statics in this file are shared by all threads of the
// device (and, separately, of the host). They are never reset because the
// program aborts.
KOKKOS_FUNCTION unsigned int& failure_message_done() {
  static unsigned int done = 0;
  return done;
}

// `Kokkos::abort` isn't `[[noreturn]]` in some configurations (e.g. CUDA with
// bounds checking). It doesn't return on CUDA and HIP devices. (On SYCL
// devices it doesn't abort yet, which SpECTRE doesn't support.)
[[noreturn]] KOKKOS_FUNCTION void abort_with_message(
    const char* const message) {
  ::Kokkos::abort(message);
  __builtin_unreachable();
}

// Stops the program on the device. Unlike `Kokkos::abort`, this doesn't print
// anything. Does nothing on the host.
[[maybe_unused]] KOKKOS_FUNCTION void trap_silently() {
#ifdef __CUDA_ARCH__
  __trap();
#elif defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
}
}  // namespace

bool claim_failure_message() {
  static unsigned int number_of_failures = 0;
  return ::Kokkos::atomic_fetch_add(&number_of_failures, 1U) == 0;
}

void print_failure_heading(const char* const heading,
                           const char* const description,
                           const char* const file, const int line,
                           const char* const pretty_function) {
  SPECTRE_DEVICE_MESSAGE_PRINTF(
      "\n############ %s ON THE DEVICE ############\n%s in %s:%d\n\n%s%s",
      heading, pretty_function, file, line, description,
      *description == '\0' ? "" : "\n");
}

void finish_failure_message(const char* const heading,
                            const char* const abort_message) {
  SPECTRE_DEVICE_MESSAGE_PRINTF(
      "\n############ %s ON THE DEVICE ############\n\n", heading);
  // Make the printed message visible before the waiting threads stop
  ::Kokkos::memory_fence();
  ::Kokkos::atomic_store(&failure_message_done(), 1U);
  abort_with_message(abort_message);
}

void wait_for_failure_message(const char* const abort_message) {
  // Bounded because the GPU may not run the printing thread while this one
  // waits, e.g. if both are in the same AMD wavefront. The printing thread
  // needs microseconds, and this many iterations take at least milliseconds.
  KOKKOS_IF_ON_DEVICE((for (unsigned int i = 0; i < (1U << 20); ++i) {
    if (::Kokkos::atomic_load(&failure_message_done()) != 0) {
      trap_silently();
      break;
    }
  }))
  abort_with_message(abort_message);
}
}  // namespace DeviceMessage_detail
#endif  // SPECTRE_KOKKOS
