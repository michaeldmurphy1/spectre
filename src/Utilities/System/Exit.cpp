// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/System/Exit.hpp"

#include <atomic>
#include <charm++.h>
#include <exception>

namespace sys {
namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> exiting{false};
}  // namespace

namespace Exit_detail {
void mark_exit_in_progress() { exiting = true; }
}  // namespace Exit_detail

bool exit_in_progress() { return exiting; }

[[noreturn]] void exit(const int exit_code) {
  Exit_detail::mark_exit_in_progress();
  CkExit(exit_code);
  // the following call is never reached, but suppresses the warning that
  // a 'noreturn' function does return
  std::terminate();  // LCOV_EXCL_LINE
}

}  // namespace sys
