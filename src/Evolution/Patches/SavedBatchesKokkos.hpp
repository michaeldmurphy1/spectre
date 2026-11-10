// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace evolution::patches {
/*!
 * \brief Copies of batches saved on the device, e.g. the variables of the
 * elements of a patch at the start of the self start.
 *
 * \details Copying a `BatchedVariables` doesn't copy its data (see
 * `BatchedVariables`), so this makes deep copies.
 */
template <typename... Batches>
class SavedBatches {
 public:
  /// Save copies of the `batches`
  void save(const Kokkos::DefaultExecutionSpace& exec,
            const Batches&... batches) {
    saved_ =
        std::tuple<Batches...>{Batches{exec, batches.number_of_grid_points(),
                                       batches.number_of_elements()}...};
    std::apply(
        [&](auto&... saved) {
          (deep_copy(exec, make_not_null(&saved), batches), ...);
        },
        saved_);
  }

  /// Copy the saved values into the `batches`
  void restore(const Kokkos::DefaultExecutionSpace& exec,
               const gsl::not_null<Batches*>... batches) const {
    std::apply(
        [&](const auto&... saved) { (deep_copy(exec, batches, saved), ...); },
        saved_);
  }

  /// Free the saved values
  void discard() { saved_ = {}; }

 private:
  std::tuple<Batches...> saved_{};
};
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
