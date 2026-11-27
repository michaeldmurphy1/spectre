// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/ActionTiming.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Parallel::action_timing {
namespace {
struct Statistic {
  size_t calls = 0;
  double seconds = 0.0;
};

using Key = std::pair<const void*, const std::string*>;

struct KeyHash {
  size_t operator()(const Key& key) const {
    return std::hash<const void*>{}(key.first) ^
           (std::hash<const void*>{}(key.second) << 1);
  }
};

// The data of one thread, so that recording needs no lock
struct ThreadData {
  std::unordered_map<Key, Statistic, KeyHash> statistics{};
  std::unordered_map<const void*, std::string> names{};
  // The (start, duration) of the actions recorded on this thread that no
  // enclosing action has claimed yet. An action runs inside another one when
  // data sent to an object on the same processor is received inline (array
  // `receive_data` is an [inline] entry method). Actions on a thread are
  // nested or sequential, so the actions that ran inside an action that
  // started at `start` are the unclaimed ones that started after it.
  std::deque<std::pair<double, double>> unclaimed{};
  int proc = -1;
  double first = std::numeric_limits<double>::infinity();
  double last = -std::numeric_limits<double>::infinity();
};

// The mutex and the data of all threads are never freed, so that they are
// still there when the file is written at exit, after the threads of Charm++
// have ended and the static variables were destroyed.
std::mutex& threads_mutex() {
  static auto* const mutex = new std::mutex{};
  return *mutex;
}

std::vector<ThreadData*>& all_threads() {
  static auto* const threads = new std::vector<ThreadData*>{};
  return *threads;
}

ThreadData& this_thread() {
  thread_local ThreadData* data = nullptr;
  if (data == nullptr) {
    data = new ThreadData{};
    const std::scoped_lock lock{threads_mutex()};
    all_threads().push_back(data);
  }
  return *data;
}

const char* file_name() {
  static const char* const name =
      // NOLINTNEXTLINE(concurrency-mt-unsafe)
      std::getenv("SPECTRE_ACTION_TIMING_FILE");
  return name;
}

void write_file() {
  const std::scoped_lock lock{threads_mutex()};
  std::ofstream file{file_name()};
  file << std::setprecision(17);
  for (const ThreadData* data : all_threads()) {
    if (data->proc >= 0) {
      file << "proc\t" << data->proc << '\t' << data->first << '\t'
           << data->last << '\n';
    }
  }
  for (const ThreadData* data : all_threads()) {
    for (const auto& [key, statistic] : data->statistics) {
      const auto name = data->names.find(key.first);
      file << "action\t"
           << (name == data->names.end() ? std::string{"unknown"}
                                         : name->second)
           << '\t' << data->proc << '\t' << *key.second << '\t'
           << statistic.calls << '\t' << statistic.seconds << '\n';
    }
  }
}
}  // namespace

bool enabled() {
  static const bool result = []() {
    return file_name() != nullptr and std::atexit(write_file) == 0;
  }();
  return result;
}

double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool has_name(const void* const object) {
  return this_thread().names.contains(object);
}

void set_name(const void* const object, std::string name) {
  this_thread().names[object] = std::move(name);
}

void record(const void* const object, const std::string& action, const int proc,
            const double start, const double end) {
  ThreadData& data = this_thread();
  // Count only the time not spent in actions that ran inside this one, so
  // that the times of all actions on a processor add up to its busy time
  double nested = 0.0;
  while (not data.unclaimed.empty() and data.unclaimed.back().first >= start) {
    nested += data.unclaimed.back().second;
    data.unclaimed.pop_back();
  }
  data.unclaimed.emplace_back(start, end - start);
  // Actions that finished long ago without an enclosing action are never
  // claimed; forget the oldest
  if (data.unclaimed.size() > 65536) {
    data.unclaimed.pop_front();
  }
  Statistic& statistic = data.statistics[Key{object, &action}];
  ++statistic.calls;
  statistic.seconds += end - start - nested;
  data.proc = proc;
  data.first = std::min(data.first, start);
  data.last = std::max(data.last, end);
}
}  // namespace Parallel::action_timing
