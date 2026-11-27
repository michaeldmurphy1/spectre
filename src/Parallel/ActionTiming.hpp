// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

/*!
 * \brief Optional timing of the actions of parallel objects, for profiling.
 *
 * \details The timing is enabled by setting the environment variable
 * `SPECTRE_ACTION_TIMING_FILE` to a file name. Then, during the `Evolve`
 * phase, `Parallel::DistributedObject` times every iterable and simple action
 * it runs. The number of calls and the total wall time are accumulated per
 * object (e.g. element) and action on the thread that runs them, and written
 * to the file when the program exits, together with the span of the timed
 * actions on each processor. Without the environment variable the cost is one
 * check of a static bool per action.
 *
 * The file is tab-separated, with lines
 * - `proc <proc> <start of the first timed action> <end of the last>`, and
 * - `action <object> <proc> <action> <calls> <total seconds>`.
 */
namespace Parallel::action_timing {
/// Whether the timing is enabled, i.e. `SPECTRE_ACTION_TIMING_FILE` is set
bool enabled();

/// The wall time in seconds
double now();

/// Whether the `object` was named on this thread (see `set_name()`)
bool has_name(const void* object);

/// Names the `object` in the file, e.g. by its component and array index
void set_name(const void* object, std::string name);

/// Adds a call of the `action` on the `object` on the processor `proc` from
/// `start` to `end`. The `action` string is identified by its address and must
/// never be destroyed, because the file is written at exit (after the
/// destruction of static variables).
void record(const void* object, const std::string& action, int proc,
            double start, double end);
}  // namespace Parallel::action_timing
