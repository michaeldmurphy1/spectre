// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace sys {

/// \ingroup UtilitiesGroup
/// \brief Exit the program normally.
/// This should only be called once over all processors.
[[noreturn]] void exit(int exit_code = 0);

/// \ingroup UtilitiesGroup
/// \brief Whether `sys::exit` or `sys::abort` was called, i.e., whether the
/// program is shutting down.
bool exit_in_progress();

namespace Exit_detail {
// Called by `sys::exit` and `sys::abort` before they exit
void mark_exit_in_progress();
}  // namespace Exit_detail
}  // namespace sys
