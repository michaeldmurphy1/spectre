// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::patches {
namespace Tags {
/*!
 * \brief The partial reductions that the first patch has received for
 * `Actions::CombineReductions`, and from how many parallel components
 *
 * \details The key is the name of the reduction and its first datum, which
 * identifies a reduction of the same kind (e.g. the slab or the time of a
 * measurement). The data is serialized, so that reductions of all types share
 * the tag.
 */
struct PendingReductions : db::SimpleTag {
  using type = std::map<std::string, std::pair<size_t, std::vector<char>>>;
};
}  // namespace Tags

namespace CombinedReduction_detail {
// The `ReductionData` with the same data and combine operations as
// `ReductionDataType`, but without finalizers: each component reduces this,
// and `CombineReductions` finalizes the combined data
template <typename ReductionDataType>
struct partial;

template <typename... Ts, typename... InvokeCombines, typename... InvokeFinals,
          typename... InvokeFinalExtraArgsIndices>
struct partial<Parallel::ReductionData<Parallel::ReductionDatum<
    Ts, InvokeCombines, InvokeFinals, InvokeFinalExtraArgsIndices>...>> {
  using type =
      Parallel::ReductionData<Parallel::ReductionDatum<Ts, InvokeCombines>...>;
};

template <typename ReductionDataType>
using partial_t = typename partial<ReductionDataType>::type;

// The `ReductionData` of type `Result` with the data of `data`
template <typename Result, typename ReductionDataType>
Result with_data_of(const ReductionDataType& data) {
  return std::apply([](const auto&... values) { return Result{values...}; },
                    data.data());
}
}  // namespace CombinedReduction_detail

namespace Actions {
/*!
 * \brief Combine the reductions of `ReductionDataType` over the elements of the
 * `DgComponent` and over the patches of the `PatchComponent`, and call `Action`
 * on each of the `TargetComponents` with the result, as a reduction over a
 * single component calls it (see `CombinedReduction`).
 *
 * \details Runs on the first patch as the reduction action of both reductions,
 * which contribute the data without finalizing it. When both have arrived
 * (only that of the patches if the `DgComponent` has no elements, since an
 * empty array can't reduce), the data is combined and finalized, and `Action`
 * is called as a simple action on each of the `TargetComponents` (a broadcast
 * for an array). The `DgComponent` is skipped if it has no elements.
 */
template <typename ReductionDataType, typename Action,
          typename TargetComponents, typename PatchComponent,
          typename DgComponent>
struct CombineReductions {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename... Args>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const Args&... args) {
    using Partial = CombinedReduction_detail::partial_t<ReductionDataType>;
    Partial data{args...};
    const bool dg_contributes =
        Parallel::get<typename Metavariables::patch_table_tag>(cache)
            .number_of_elements_outside_patches() > 0;
    const size_t expected = dg_contributes ? 2 : 1;
    const std::vector<char> first_datum = serialize(std::get<0>(data.data()));
    const std::string key = pretty_type::get_name<CombineReductions>() + "/" +
                            std::string(first_datum.begin(), first_datum.end());
    std::optional<Partial> combined{};
    db::mutate<Tags::PendingReductions>(
        [&](const gsl::not_null<
            std::map<std::string, std::pair<size_t, std::vector<char>>>*>
                pending) {
          const auto entry = pending->find(key);
          if (entry != pending->end()) {
            auto earlier = deserialize<Partial>(entry->second.second.data());
            earlier.combine(std::move(data));
            data = std::move(earlier);
          }
          const size_t received =
              (entry == pending->end() ? 0 : entry->second.first) + 1;
          if (received == expected) {
            combined = std::move(data);
            if (entry != pending->end()) {
              pending->erase(entry);
            }
          } else {
            (*pending)[key] = std::pair{received, serialize(data)};
          }
        },
        make_not_null(&box));
    if (not combined.has_value()) {
      return;
    }
    auto result =
        CombinedReduction_detail::with_data_of<ReductionDataType>(*combined);
    result.finalize();
    tmpl::for_each<TargetComponents>(
        [&]<typename Target>(tmpl::type_<Target> /*meta*/) {
          if (std::is_same_v<Target, DgComponent> and not dg_contributes) {
            return;
          }
          std::apply(
              [&cache](const auto&... values) {
                Parallel::simple_action<Action>(
                    Parallel::get_parallel_component<Target>(cache), values...);
              },
              result.data());
        });
  }
};
}  // namespace Actions

/*!
 * \brief Reductions over all elements of executables with patches: those of
 * the `DgComponent` and those of the patches of the `PatchComponent`.
 *
 * \details Events that reduce data over the elements (e.g. control-system
 * measurements) contribute with `contribute` instead of
 * `Parallel::contribute_to_reduction` if the `Metavariables` set
 * `combined_reduction` to this. The elements of the `DgComponent` and the
 * patches each reduce to the first patch, which combines the two (see
 * `Actions::CombineReductions`). Without patches, the elements reduce as if
 * this weren't set.
 */
template <typename PatchComponent, typename DgComponent>
struct CombinedReduction {
  /*!
   * \brief Contribute `data` from `sender_proxy` (an element of the
   * `DgComponent` or a patch) to the reduction over all elements, whose result
   * is passed to `Action` on each of the `TargetComponents`.
   *
   * \details The `TargetComponents` may include the `PatchComponent` and the
   * `DgComponent`, e.g. to broadcast the result to all elements. Without
   * patches, the reduction is over the `DgComponent` to the one other target.
   */
  template <typename Action, typename TargetComponents, typename Metavariables,
            typename SenderProxy, typename... Ts>
  static void contribute(Parallel::GlobalCache<Metavariables>& cache,
                         Parallel::ReductionData<Ts...> data,
                         const SenderProxy& sender_proxy) {
    using ReductionDataType = Parallel::ReductionData<Ts...>;
    if (Parallel::get<typename Metavariables::patch_table_tag>(cache)
            .number_of_patches() == 0) {
      using targets_without_patches =
          tmpl::remove<TargetComponents, PatchComponent>;
      static_assert(tmpl::size<targets_without_patches>::value == 1,
                    "Without patches, the reduction has one target.");
      Parallel::contribute_to_reduction<Action>(
          std::move(data), sender_proxy,
          Parallel::get_parallel_component<
              tmpl::front<targets_without_patches>>(cache));
    } else {
      using Partial = CombinedReduction_detail::partial_t<ReductionDataType>;
      Parallel::contribute_to_reduction<Actions::CombineReductions<
          ReductionDataType, Action, TargetComponents, PatchComponent,
          DgComponent>>(
          CombinedReduction_detail::with_data_of<Partial>(data), sender_proxy,
          Parallel::get_parallel_component<PatchComponent>(cache)[0]);
    }
  }
};
}  // namespace evolution::patches
