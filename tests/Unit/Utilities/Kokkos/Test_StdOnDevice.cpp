// Distributed under the MIT License.
// See LICENSE.txt for details.

// Temporary experiment, not to be merged: which parts of the host standard
// library (and of SpECTRE's `Utilities/Algorithm.hpp`) give correct results
// in Kokkos kernels?
//
// Every probe is its own test case (`Unit.Scratch.StdOnDevice.<Probe>`), so
// ctest runs each in its own process and a probe that crashes on the device
// doesn't hide the results of the others. Each probe's `compute()` returns an
// array of values. It is called in a kernel and on the host (see
// `TestHelpers::check_on_device`), and the host values are the reference. The
// values are chosen to be nonzero, since the device values start as zero, so a
// kernel that silently doesn't write its results fails too. `CAPTURE(i)` in a
// failure is the index into the array returned by `compute()`.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>

#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t N>
using Values = Kokkos::Array<double, N>;

KOKKOS_FUNCTION double as_double(const bool b) { return b ? 11.0 : 10.0; }

// Calls `Probe::compute()` in a kernel and on the host
template <typename Probe>
void check_probe() {
  TestHelpers::check_on_device(KOKKOS_LAMBDA() { return Probe::compute(); });
}

// Calls `Probe::compute(t)` for `Probe::number_of_threads` threads `t` in a
// kernel and on the host
template <typename Probe>
void check_threads_probe() {
  constexpr size_t number_of_threads = Probe::number_of_threads;
  using Result = decltype(Probe::compute(size_t{0}));
  constexpr size_t number_of_values = Result::size();
  const Kokkos::View<double**> device_values("device_values", number_of_threads,
                                             number_of_values);
  Kokkos::parallel_for(
      "threads_probe", Kokkos::RangePolicy<>(0, number_of_threads),
      KOKKOS_LAMBDA(const size_t t) {
        const Result values = Probe::compute(t);
        for (size_t i = 0; i < number_of_values; ++i) {
          device_values(t, i) = values[i];
        }
      });
  const auto host_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_values);
  for (size_t t = 0; t < number_of_threads; ++t) {
    const Result expected = Probe::compute(t);
    for (size_t i = 0; i < number_of_values; ++i) {
      CAPTURE(t);
      CAPTURE(i);
      CHECK(host_values(t, i) == approx(expected[i]));
    }
  }
}
}  // namespace

namespace {
// ---------------------------------------------------------------------------
// Baselines without the standard library
// ---------------------------------------------------------------------------

struct ReturnLiterals {
  KOKKOS_FUNCTION static Values<4> compute() { return {{1.0, 2.0, 3.0, 4.0}}; }
};

struct FillLoop {
  KOKKOS_FUNCTION static Values<8> compute() {
    Values<8> r{};
    for (size_t i = 0; i < 8; ++i) {
      r[i] = static_cast<double>(i * i) + 1.0;
    }
    return r;
  }
};

struct CArray {
  KOKKOS_FUNCTION static Values<4> compute() {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    double c[4] = {4.0, 3.0, 2.0, 1.0};
    Values<4> r{};
    for (size_t i = 0; i < 4; ++i) {
      r[i] = c[3 - i] + 10.0;
    }
    return r;
  }
};

struct KokkosArrayAndMath {
  KOKKOS_FUNCTION static Values<4> compute() {
    const Kokkos::Array<double, 3> a{{3.0, 1.0, 2.0}};
    return {{a[0] + a[1] + a[2], static_cast<double>(a.size()),
             Kokkos::max(a[0], a[1]), Kokkos::sqrt(a[0] + 1.0)}};
  }
};

// ---------------------------------------------------------------------------
// std::array
// ---------------------------------------------------------------------------

struct StdArrayIndex {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 4> a{};
    for (size_t i = 0; i < 4; ++i) {
      a[i] = 2.0 * static_cast<double>(i) + 1.0;
    }
    return {{a[0], a[1], a[2], a[3]}};
  }
};

struct StdArrayIterators {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    double range_for_sum = 0.0;
    for (const double x : a) {
      range_for_sum += x;
    }
    double iterator_sum = 0.0;
    for (auto it = a.begin(); it != a.end(); ++it) {
      iterator_sum += *it;
    }
    return {{range_for_sum, iterator_sum, a.front(), a.back(), a.data()[2],
             static_cast<double>(a.size())}};
  }
};

struct StdArrayFillAndSwapMembers {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 2> a{};
    std::array<double, 2> b{{5.0, 6.0}};
    a.fill(3.5);
    a.swap(b);
    return {{a[0], a[1], b[0], b[1]}};
  }
};

// `==` was broken on the GPU on 2026-10-17
struct StdArrayCompare {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::array<size_t, 2> a{{0, 2}};
    const std::array<size_t, 2> b{{0, 2}};
    const std::array<size_t, 2> c{{1, 2}};
    return {{as_double(a == b), as_double(a != c), as_double(a < c),
             as_double(a == c), as_double(c > a), as_double(a <= b)}};
  }
};

struct StdArrayCopy {
  KOKKOS_FUNCTION static Values<4> compute() {
    const std::array<double, 2> a{{7.0, 8.0}};
    const std::array<double, 2> b = a;
    std::array<double, 2> c{};
    c = b;
    return {{b[0], b[1], c[0], c[1]}};
  }
};

// ---------------------------------------------------------------------------
// <algorithm>: values, copies and queries
// ---------------------------------------------------------------------------

struct MinMax {
  KOKKOS_FUNCTION static Values<6> compute() {
    const double x = 2.5;
    const double y = -1.5;
    return {{std::min(x, y), std::max(x, y),
             static_cast<double>(std::min(3, 7)),
             static_cast<double>(std::max(size_t{3}, size_t{7})),
             std::min(x, 1.0), std::max(y, 1.0)}};
  }
};

struct MinMaxInitializerList {
  KOKKOS_FUNCTION static Values<2> compute() {
    return {{std::max({1.0, 4.0, 2.0}), std::min({3.0, 0.5, 2.0})}};
  }
};

struct Clamp {
  KOKKOS_FUNCTION static Values<3> compute() {
    return {{std::clamp(5.0, 0.0, 1.0), std::clamp(-5.0, -1.0, 1.0) + 2.0,
             std::clamp(0.25, 0.0, 1.0)}};
  }
};

struct SwapAndExchange {
  KOKKOS_FUNCTION static Values<4> compute() {
    double a = 1.0;
    double b = 2.0;
    std::swap(a, b);
    double c = 3.0;
    const double old_c = std::exchange(c, 9.0);
    return {{a, b, old_c, c}};
  }
};

struct FillAndCopy {
  KOKKOS_FUNCTION static Values<8> compute() {
    std::array<double, 4> a{};
    std::fill(a.begin(), a.end(), 2.0);
    std::array<double, 4> b{};
    std::copy(a.begin(), a.end(), b.begin());
    b[1] = 5.0;
    std::array<double, 4> c{};
    std::fill_n(c.begin(), 2, 7.0);
    std::copy_n(b.begin(), 2, c.begin() + 2);
    return {{b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]}};
  }
};

struct ReverseAndRotate {
  KOKKOS_FUNCTION static Values<8> compute() {
    std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    std::reverse(a.begin(), a.end());
    std::array<double, 4> b{{1.0, 2.0, 3.0, 4.0}};
    std::rotate(b.begin(), b.begin() + 1, b.end());
    return {{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]}};
  }
};

struct CountAndFind {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::array<int, 6> a{{3, 1, 4, 1, 5, 9}};
    return {{static_cast<double>(std::count(a.begin(), a.end(), 1)),
             static_cast<double>(std::count_if(
                 a.begin(), a.end(), [](const int x) { return x > 2; })),
             static_cast<double>(
                 std::distance(a.begin(), std::find(a.begin(), a.end(), 4))),
             static_cast<double>(std::distance(
                 a.begin(), std::find_if(a.begin(), a.end(),
                                         [](const int x) { return x > 4; }))),
             as_double(std::find(a.begin(), a.end(), 7) == a.end()),
             static_cast<double>(*std::find(a.begin(), a.end(), 9))}};
  }
};

struct AllAnyNoneOf {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::array<int, 4> a{{2, 4, 6, 7}};
    const auto even = [](const int x) { return x % 2 == 0; };
    return {{as_double(std::all_of(a.begin(), a.end(), even)),
             as_double(std::any_of(a.begin(), a.end(), even)),
             as_double(std::none_of(a.begin(), a.end(), even)),
             as_double(std::all_of(a.begin(), a.begin() + 3, even)),
             as_double(std::any_of(a.begin(), a.end(),
                                   [](const int x) { return x > 6; })),
             as_double(std::none_of(a.begin(), a.end(),
                                    [](const int x) { return x > 9; }))}};
  }
};

struct ForEach {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    std::for_each(a.begin(), a.end(), [](double& x) { x *= 3.0; });
    return {{a[0], a[1], a[2], a[3]}};
  }
};

struct ForEachN {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    std::for_each_n(a.begin(), 2, [](double& x) { x += 10.0; });
    return {{a[0], a[1], a[2], a[3]}};
  }
};

struct Transform {
  KOKKOS_FUNCTION static Values<8> compute() {
    const std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    const std::array<double, 4> b{{4.0, 3.0, 2.0, 1.0}};
    std::array<double, 4> unary{};
    std::array<double, 4> binary{};
    std::transform(a.begin(), a.end(), unary.begin(),
                   [](const double x) { return x * x; });
    std::transform(a.begin(), a.end(), b.begin(), binary.begin(),
                   [](const double x, const double y) { return x + 2.0 * y; });
    return {{unary[0], unary[1], unary[2], unary[3], binary[0], binary[1],
             binary[2], binary[3]}};
  }
};

struct EqualAndLexicographicalCompare {
  KOKKOS_FUNCTION static Values<4> compute() {
    const std::array<int, 3> a{{1, 2, 3}};
    const std::array<int, 3> b{{1, 2, 3}};
    const std::array<int, 3> c{{1, 3, 0}};
    return {
        {as_double(std::equal(a.begin(), a.end(), b.begin())),
         as_double(std::equal(a.begin(), a.end(), c.begin())),
         as_double(std::lexicographical_compare(a.begin(), a.end(), c.begin(),
                                                c.end())),
         static_cast<double>(std::distance(
             a.begin(), std::mismatch(a.begin(), a.end(), c.begin()).first))}};
  }
};

struct ReplaceAndRemove {
  KOKKOS_FUNCTION static Values<7> compute() {
    std::array<int, 4> a{{1, 2, 1, 3}};
    std::replace(a.begin(), a.end(), 1, 5);
    std::replace_if(a.begin(), a.end(), [](const int x) { return x == 3; }, 8);
    std::array<int, 4> b{{1, 2, 1, 3}};
    const auto new_end = std::remove(b.begin(), b.end(), 1);
    return {{static_cast<double>(a[0]), static_cast<double>(a[1]),
             static_cast<double>(a[2]), static_cast<double>(a[3]),
             static_cast<double>(std::distance(b.begin(), new_end)),
             static_cast<double>(b[0]), static_cast<double>(b[1])}};
  }
};

// ---------------------------------------------------------------------------
// <algorithm>: sorting, permutations and searching
// ---------------------------------------------------------------------------

// libstdc++ sorts up to 16 elements with insertion sort only, so small and
// large arrays take different code paths. `std::sort` was broken on the GPU on
// 2026-10-17.
struct SortTwo {
  KOKKOS_FUNCTION static Values<2> compute() {
    std::array<double, 2> a{{2.0, 1.0}};
    std::sort(a.begin(), a.end());
    return {{a[0], a[1]}};
  }
};

struct SortThree {
  KOKKOS_FUNCTION static Values<3> compute() {
    std::array<double, 3> a{{2.0, 3.0, 1.0}};
    std::sort(a.begin(), a.end());
    return {{a[0], a[1], a[2]}};
  }
};

struct SortSizeT {
  KOKKOS_FUNCTION static Values<3> compute() {
    std::array<size_t, 3> a{{1, 2, 0}};
    std::sort(a.begin(), a.end());
    return {{static_cast<double>(a[0]) + 1.0, static_cast<double>(a[1]) + 1.0,
             static_cast<double>(a[2]) + 1.0}};
  }
};

struct SortWithComparator {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 4> a{{2.0, 4.0, 1.0, 3.0}};
    std::sort(a.begin(), a.end(), std::greater<>{});
    return {{a[0], a[1], a[2], a[3]}};
  }
};

struct SortTwenty {
  KOKKOS_FUNCTION static Values<20> compute() {
    std::array<double, 20> a{};
    for (size_t i = 0; i < 20; ++i) {
      a[i] = static_cast<double>((7 * i + 3) % 20) + 1.0;
    }
    std::sort(a.begin(), a.end());
    Values<20> r{};
    for (size_t i = 0; i < 20; ++i) {
      r[i] = a[i];
    }
    return r;
  }
};

struct PartialSortAndNthElement {
  KOKKOS_FUNCTION static Values<6> compute() {
    std::array<double, 5> a{{5.0, 1.0, 4.0, 2.0, 3.0}};
    std::partial_sort(a.begin(), a.begin() + 2, a.end());
    std::array<double, 5> b{{5.0, 1.0, 4.0, 2.0, 3.0}};
    std::nth_element(b.begin(), b.begin() + 2, b.end());
    std::array<double, 5> c{{5.0, 1.0, 4.0, 2.0, 3.0}};
    std::partition(c.begin(), c.end(), [](const double x) { return x < 3.0; });
    return {{a[0], a[1], b[2], std::min(c[0], c[1]), std::max(c[0], c[1]),
             c[2] + c[3] + c[4]}};
  }
};

// `std::unique` was broken on the GPU on 2026-10-17
struct Unique {
  KOKKOS_FUNCTION static Values<5> compute() {
    std::array<int, 6> a{{1, 1, 2, 2, 2, 3}};
    const auto new_end = std::unique(a.begin(), a.end());
    std::array<int, 3> b{{2, 0, 1}};
    return {{static_cast<double>(std::distance(a.begin(), new_end)),
             static_cast<double>(a[0]), static_cast<double>(a[1]),
             static_cast<double>(a[2]),
             as_double(std::unique(b.begin(), b.end()) == b.end())}};
  }
};

// The check in `Mesh::slice_through` that fired on the GPU on 2026-10-17
struct SortThenUnique {
  KOKKOS_FUNCTION static Values<2> compute() {
    std::array<size_t, 2> a{{2, 0}};
    std::sort(a.begin(), a.end());
    return {{as_double(std::unique(a.begin(), a.end()) == a.end()),
             static_cast<double>(a[1])}};
  }
};

struct MinMaxElement {
  KOKKOS_FUNCTION static Values<4> compute() {
    const std::array<double, 5> a{{3.0, -1.0, 4.0, 1.0, 5.0}};
    const auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    return {{*std::min_element(a.begin(), a.end()),
             *std::max_element(a.begin(), a.end()), *min_it, *max_it}};
  }
};

struct SortedSearch {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::array<int, 6> a{{1, 3, 3, 5, 7, 9}};
    return {{as_double(std::is_sorted(a.begin(), a.end())),
             as_double(std::binary_search(a.begin(), a.end(), 5)),
             as_double(std::binary_search(a.begin(), a.end(), 4)),
             static_cast<double>(std::distance(
                 a.begin(), std::lower_bound(a.begin(), a.end(), 3))),
             static_cast<double>(std::distance(
                 a.begin(), std::upper_bound(a.begin(), a.end(), 3))),
             static_cast<double>(std::distance(
                 a.begin(),
                 std::partition_point(a.begin(), a.end(),
                                      [](const int x) { return x < 6; })))}};
  }
};

struct NextAndPrevPermutation {
  KOKKOS_FUNCTION static Values<7> compute() {
    std::array<int, 3> a{{1, 2, 3}};
    int count = 0;
    do {
      ++count;
    } while (std::next_permutation(a.begin(), a.end()));
    std::array<int, 3> b{{1, 3, 2}};
    std::next_permutation(b.begin(), b.end());
    std::array<int, 3> c = b;
    std::prev_permutation(c.begin(), c.end());
    return {{static_cast<double>(count), static_cast<double>(b[0]),
             static_cast<double>(b[1]), static_cast<double>(b[2]),
             static_cast<double>(c[0]), static_cast<double>(c[1]),
             static_cast<double>(c[2])}};
  }
};

// ---------------------------------------------------------------------------
// <numeric>
// ---------------------------------------------------------------------------

struct Numeric {
  KOKKOS_FUNCTION static Values<10> compute() {
    std::array<double, 4> a{};
    std::iota(a.begin(), a.end(), 1.0);
    std::array<double, 4> sums{};
    std::partial_sum(a.begin(), a.end(), sums.begin());
    std::array<double, 4> differences{};
    std::adjacent_difference(sums.begin(), sums.end(), differences.begin());
    return {{std::accumulate(a.begin(), a.end(), 0.0),
             std::accumulate(a.begin(), a.end(), 1.0, std::multiplies<>{}),
             std::inner_product(a.begin(), a.end(), a.begin(), 0.0), sums[3],
             sums[1], differences[2], std::reduce(a.begin(), a.end()),
             std::transform_reduce(a.begin(), a.end(), 0.0, std::plus<>{},
                                   [](const double x) { return 2.0 * x; }),
             static_cast<double>(std::gcd(12, 18)),
             static_cast<double>(std::lcm(4, 6))}};
  }
};

// ---------------------------------------------------------------------------
// <ranges>
// ---------------------------------------------------------------------------

// nvcc rejects these in device code with "identifier "std::ranges::sort" is
// undefined in device code" (and similarly for the others): the `std::ranges`
// algorithms, `std::views::reverse` and `std::ignore` are global objects in
// host memory. So these probes only run without a GPU backend.
#if not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))
struct RangesSort {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 4> a{{3.0, 1.0, 4.0, 2.0}};
    std::ranges::sort(a);
    return {{a[0], a[1], a[2], a[3]}};
  }
};

struct RangesForEachAndTransform {
  KOKKOS_FUNCTION static Values<8> compute() {
    std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    std::ranges::for_each(a, [](double& x) { x += 1.0; });
    std::array<double, 4> b{};
    std::ranges::transform(a, b.begin(),
                           [](const double x) { return 3.0 * x; });
    return {{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]}};
  }
};

struct RangesQueries {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::array<int, 5> a{{4, 2, 7, 2, 1}};
    return {
        {static_cast<double>(std::ranges::count(a, 2)),
         static_cast<double>(std::ranges::max(a)),
         static_cast<double>(std::ranges::min(a)) + 1.0,
         as_double(std::ranges::all_of(a, [](const int x) { return x > 0; })),
         static_cast<double>(std::distance(a.begin(), std::ranges::find(a, 7))),
         static_cast<double>(*std::ranges::max_element(a))}};
  }
};

struct RangesFillAndReverse {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 4> a{};
    std::ranges::fill(a, 6.0);
    a[0] = 1.0;
    std::ranges::reverse(a);
    return {{a[0], a[1], a[2], a[3]}};
  }
};

#endif  // not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))

// The `std::ranges` algorithms copied into local `constexpr` objects first
struct RangesAlgorithmsAsLocalObjects {
  KOKKOS_FUNCTION static Values<6> compute() {
    constexpr auto sort = std::ranges::sort;
    constexpr auto max = std::ranges::max;
    constexpr auto count = std::ranges::count;
    std::array<double, 4> a{{3.0, 1.0, 4.0, 2.0}};
    sort(a);
    return {{a[0], a[1], a[2], a[3], max(a),
             static_cast<double>(count(a, 4.0)) + 1.0}};
  }
};

struct ViewsIota {
  KOKKOS_FUNCTION static Values<2> compute() {
    double sum = 0.0;
    for (const int i : std::views::iota(1, 5)) {
      sum += static_cast<double>(i);
    }
    size_t count = 0;
    for ([[maybe_unused]] const size_t i :
         std::views::iota(size_t{0}, size_t{7})) {
      ++count;
    }
    return {{sum, static_cast<double>(count)}};
  }
};

struct ViewsPipeline {
  KOKKOS_FUNCTION static Values<2> compute() {
    const std::array<int, 6> a{{1, 2, 3, 4, 5, 6}};
    double sum_of_even_squares = 0.0;
    for (const int x :
         a | std::views::filter([](const int y) { return y % 2 == 0; }) |
             std::views::transform([](const int y) { return y * y; })) {
      sum_of_even_squares += static_cast<double>(x);
    }
    double sum_of_first_two = 0.0;
    for (const int x : a | std::views::take(2)) {
      sum_of_first_two += static_cast<double>(x);
    }
    return {{sum_of_even_squares, sum_of_first_two}};
  }
};

// nvcc rejects these in device code with "identifier "std::ranges::sort" is
// undefined in device code" (and similarly for the others): the `std::ranges`
// algorithms, `std::views::reverse` and `std::ignore` are global objects in
// host memory. So these probes only run without a GPU backend.
#if not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))
struct ViewsReverse {
  KOKKOS_FUNCTION static Values<1> compute() {
    const std::array<int, 6> a{{1, 2, 3, 4, 5, 6}};
    double last = 0.0;
    for (const int x : a | std::views::reverse) {
      last = static_cast<double>(x);
      break;
    }
    return {{last}};
  }
};
#endif  // not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))

// ---------------------------------------------------------------------------
// <cmath> and <limits>
// ---------------------------------------------------------------------------

struct CmathStd {
  KOKKOS_FUNCTION static Values<16> compute() {
    const double x = 2.0;
    return {{std::sqrt(x), std::abs(-x), std::fabs(-3.5), std::pow(x, 3.5),
             std::exp(x), std::log(x), std::sin(x) + 2.0, std::atan2(1.0, x),
             std::hypot(3.0, 4.0), std::fmax(x, 5.0), std::fmin(x, -5.0),
             std::fma(x, 3.0, 1.0), std::floor(2.7) + std::ceil(2.2),
             std::round(2.5), std::copysign(4.0, -1.0), std::cbrt(27.0)}};
  }
};

struct CmathUnqualified {
  KOKKOS_FUNCTION static Values<6> compute() {
    const double x = 2.0;
    return {
        {sqrt(x), fabs(-x), pow(x, 0.5), exp(1.0), log(10.0), fmax(x, 3.0)}};
  }
};

struct CmathClassify {
  KOKKOS_FUNCTION static Values<6> compute() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    return {{as_double(std::isnan(nan)), as_double(std::isinf(inf)),
             as_double(std::isfinite(1.0)), as_double(std::isfinite(inf)),
             as_double(std::signbit(-2.0)), as_double(std::isnan(1.0))}};
  }
};

struct NumericLimitsAtRunTime {
  KOKKOS_FUNCTION static Values<6> compute() {
    return {{std::numeric_limits<double>::epsilon() * 1.0e16,
             std::numeric_limits<double>::max() * 1.0e-300,
             std::numeric_limits<double>::min() * 1.0e300,
             -std::numeric_limits<double>::lowest() * 1.0e-300,
             static_cast<double>(std::numeric_limits<int>::max()),
             as_double(std::numeric_limits<size_t>::max() > 1000)}};
  }
};

struct NumericLimitsConstexpr {
  KOKKOS_FUNCTION static Values<3> compute() {
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    constexpr double largest = std::numeric_limits<double>::max();
    constexpr int digits = std::numeric_limits<double>::digits;
    return {
        {epsilon * 1.0e16, largest * 1.0e-300, static_cast<double>(digits)}};
  }
};

// ---------------------------------------------------------------------------
// <utility>, <tuple>, <optional>, <complex>, <span>, <functional>
// ---------------------------------------------------------------------------

struct Pair {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::pair<double, int> p{1.5, 2};
    const auto q = std::make_pair(3.0, 4);
    const auto [first, second] = q;
    return {{p.first, static_cast<double>(p.second), first,
             static_cast<double>(second), as_double(p < q),
             as_double(p == std::pair<double, int>{1.5, 2})}};
  }
};

struct Tuple {
  KOKKOS_FUNCTION static Values<5> compute() {
    const std::tuple<double, int, double> t{1.5, 2, 3.5};
    double x = 0.0;
    int y = 0;
    double z = 0.0;
    std::tie(x, y, z) = t;
    return {{std::get<0>(t), static_cast<double>(std::get<1>(t)),
             std::get<2>(t), x + static_cast<double>(y) + z,
             std::apply(
                 [](const double a, const int b, const double c) {
                   return a + static_cast<double>(b) + c;
                 },
                 t)}};
  }
};

// nvcc rejects these in device code with "identifier "std::ranges::sort" is
// undefined in device code" (and similarly for the others): the `std::ranges`
// algorithms, `std::views::reverse` and `std::ignore` are global objects in
// host memory. So these probes only run without a GPU backend.
#if not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))
struct TupleTieWithIgnore {
  KOKKOS_FUNCTION static Values<1> compute() {
    const std::tuple<double, int, double> t{1.5, 2, 3.5};
    double x = 0.0;
    int y = 0;
    std::tie(x, y, std::ignore) = t;
    return {{x + static_cast<double>(y)}};
  }
};
#endif  // not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))

struct Optional {
  KOKKOS_FUNCTION static Values<5> compute() {
    std::optional<double> o{};
    const double empty_value = o.value_or(7.0);
    const bool empty_has_value = o.has_value();
    o = 2.5;
    return {{empty_value, as_double(empty_has_value), *o, o.value_or(7.0),
             as_double(o.has_value())}};
  }
};

struct Complex {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::complex<double> z{1.0, 2.0};
    const std::complex<double> w = z * z + 1.0;
    return {{w.real() + 10.0, w.imag(), std::abs(z), std::conj(z).imag() + 10.0,
             std::norm(z),
             std::real(std::exp(std::complex<double>{0.0, 1.0})) + 10.0}};
  }
};

struct Span {
  KOKKOS_FUNCTION static Values<3> compute() {
    std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    const std::span<double> s{a};
    s[1] = 9.0;
    double sum = 0.0;
    for (const double x : s.subspan(1, 2)) {
      sum += x;
    }
    return {{sum, static_cast<double>(s.size()), a[1]}};
  }
};

struct InitializerList {
  KOKKOS_FUNCTION static Values<2> compute() {
    double sum = 0.0;
    size_t count = 0;
    for (const double x : {1.0, 2.0, 3.0}) {
      sum += x;
      ++count;
    }
    return {{sum, static_cast<double>(count)}};
  }
};

struct FunctionObjects {
  KOKKOS_FUNCTION static Values<5> compute() {
    return {{std::plus<>{}(1.0, 2.0), std::multiplies<double>{}(3.0, 4.0),
             as_double(std::less<>{}(1.0, 2.0)),
             std::invoke([](const double x) { return x + 1.0; }, 4.0),
             std::negate<>{}(-6.0)}};
  }
};

// ---------------------------------------------------------------------------
// SpECTRE's Utilities/Algorithm.hpp and other utilities
// ---------------------------------------------------------------------------

struct AlgCountAndAllAnyNoneOf {
  KOKKOS_FUNCTION static Values<6> compute() {
    const std::array<int, 4> a{{2, 4, 6, 7}};
    const auto even = [](const int x) { return x % 2 == 0; };
    return {{static_cast<double>(alg::count(a, 4)),
             static_cast<double>(alg::count_if(a, even)),
             as_double(alg::all_of(a, even)), as_double(alg::any_of(a, even)),
             as_double(alg::none_of(a, even)),
             as_double(alg::any_of(a, [](const int x) { return x > 6; }))}};
  }
};

// `alg::find` and `alg::found` use SpECTRE's own `cpp20::find`
struct AlgFind {
  KOKKOS_FUNCTION static Values<5> compute() {
    const std::array<int, 4> a{{3, 1, 4, 1}};
    return {
        {static_cast<double>(std::distance(a.begin(), alg::find(a, 4))),
         as_double(alg::found(a, 1)), as_double(alg::found(a, 7)),
         static_cast<double>(std::distance(
             a.begin(), alg::find_if(a, [](const int x) { return x > 3; }))),
         as_double(alg::found_if_not(a, [](const int x) { return x > 0; }))}};
  }
};

struct AlgForEachAndTransform {
  KOKKOS_FUNCTION static Values<8> compute() {
    std::array<double, 4> a{{1.0, 2.0, 3.0, 4.0}};
    alg::for_each(a, [](double& x) { x *= 2.0; });
    std::array<double, 4> b{};
    alg::transform(a, b.begin(), [](const double x) { return x + 0.5; });
    return {{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]}};
  }
};

struct AlgEqualAndMinMaxElement {
  KOKKOS_FUNCTION static Values<5> compute() {
    const std::array<double, 3> a{{2.0, -1.0, 3.0}};
    const std::array<double, 3> b{{2.0, -1.0, 3.0}};
    const auto [min_it, max_it] = alg::minmax_element(a);
    return {{as_double(alg::equal(a, b)), *alg::min_element(a),
             *alg::max_element(a), *min_it, *max_it}};
  }
};

struct AlgSort {
  KOKKOS_FUNCTION static Values<4> compute() {
    std::array<double, 4> a{{3.0, 1.0, 4.0, 2.0}};
    // `alg::sort` aborts on the device, so probe the device sort
    alg::bubble_sort(a);
    return {{a[0], a[1], a[2], a[3]}};
  }
};

struct Cpp20Reimplementations {
  KOKKOS_FUNCTION static Values<8> compute() {
    std::array<int, 4> a{{1, 2, 3, 4}};
    cpp20::reverse(a.begin(), a.end());
    std::array<int, 3> b{{1, 3, 2}};
    cpp20::next_permutation(b.begin(), b.end());
    int x = 5;
    int y = 9;
    cpp20::swap(x, y);
    return {{static_cast<double>(a[0]), static_cast<double>(a[1]),
             static_cast<double>(a[2]), static_cast<double>(a[3]),
             static_cast<double>(b[0]), static_cast<double>(b[1]),
             static_cast<double>(b[2]), static_cast<double>(x)}};
  }
};

struct GslAtMakeArraySquare {
  KOKKOS_FUNCTION static Values<5> compute() {
    std::array<double, 3> a{{1.0, 2.0, 3.0}};
    gsl::at(a, 1) = 7.0;
    const auto b = make_array<3>(1.5);
    return {{gsl::at(a, 1), b[2], square(3.0), cube(2.0),
             static_cast<double>(gsl::narrow_cast<int>(4.0))}};
  }
};

// ---------------------------------------------------------------------------
// Many threads, each with different data
// ---------------------------------------------------------------------------

KOKKOS_FUNCTION double thread_value(const size_t t, const size_t i) {
  return static_cast<double>((t + 3 * i) % 7) + 1.0;
}

struct ThreadsFill {
  static constexpr size_t number_of_threads = 64;
  KOKKOS_FUNCTION static Values<4> compute(const size_t t) {
    Values<4> r{};
    for (size_t i = 0; i < 4; ++i) {
      r[i] = static_cast<double>(4 * t + i) + 1.0;
    }
    return r;
  }
};

struct ThreadsStdArrayLoop {
  static constexpr size_t number_of_threads = 64;
  KOKKOS_FUNCTION static Values<4> compute(const size_t t) {
    std::array<double, 4> a{};
    for (size_t i = 0; i < 4; ++i) {
      a[i] = thread_value(t, i);
    }
    return {{a[0], a[1], a[2], a[3]}};
  }
};

struct ThreadsSort {
  static constexpr size_t number_of_threads = 64;
  KOKKOS_FUNCTION static Values<5> compute(const size_t t) {
    std::array<double, 5> a{};
    for (size_t i = 0; i < 5; ++i) {
      a[i] = thread_value(t, i);
    }
    std::sort(a.begin(), a.end());
    return {{a[0], a[1], a[2], a[3], a[4]}};
  }
};

struct ThreadsMinMaxAccumulate {
  static constexpr size_t number_of_threads = 64;
  KOKKOS_FUNCTION static Values<4> compute(const size_t t) {
    std::array<double, 5> a{};
    for (size_t i = 0; i < 5; ++i) {
      a[i] = thread_value(t, i);
    }
    return {{*std::min_element(a.begin(), a.end()),
             *std::max_element(a.begin(), a.end()),
             std::accumulate(a.begin(), a.end(), 0.0), std::max(a[0], a[4])}};
  }
};

struct ThreadsArrayEqual {
  static constexpr size_t number_of_threads = 64;
  KOKKOS_FUNCTION static Values<2> compute(const size_t t) {
    const std::array<size_t, 2> a{{t, t + 1}};
    const std::array<size_t, 2> b{{t, t + (t % 2)}};
    return {{as_double(a == b), as_double(alg::equal(a, b))}};
  }
};
}  // namespace

// A kernel that does nothing leaves the device values at zero
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.DoNothing", "[Unit]") {
  const Kokkos::View<double*> device_values("device_values", 4);
  Kokkos::parallel_for("do_nothing", Kokkos::RangePolicy<>(0, 1),
                       KOKKOS_LAMBDA(const int /*unused*/){});
  const auto host_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_values);
  for (size_t i = 0; i < 4; ++i) {
    CAPTURE(i);
    CHECK(host_values(i) == 0.0);
  }
}

SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ReturnLiterals", "[Unit]") {
  check_probe<ReturnLiterals>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.FillLoop", "[Unit]") {
  check_probe<FillLoop>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.CArray", "[Unit]") {
  check_probe<CArray>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.KokkosArrayAndMath", "[Unit]") {
  check_probe<KokkosArrayAndMath>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.StdArrayIndex", "[Unit]") {
  check_probe<StdArrayIndex>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.StdArrayIterators", "[Unit]") {
  check_probe<StdArrayIterators>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.StdArrayFillAndSwapMembers",
                  "[Unit]") {
  check_probe<StdArrayFillAndSwapMembers>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.StdArrayCompare", "[Unit]") {
  check_probe<StdArrayCompare>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.StdArrayCopy", "[Unit]") {
  check_probe<StdArrayCopy>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.MinMax", "[Unit]") {
  check_probe<MinMax>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.MinMaxInitializerList", "[Unit]") {
  check_probe<MinMaxInitializerList>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Clamp", "[Unit]") {
  check_probe<Clamp>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SwapAndExchange", "[Unit]") {
  check_probe<SwapAndExchange>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.FillAndCopy", "[Unit]") {
  check_probe<FillAndCopy>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ReverseAndRotate", "[Unit]") {
  check_probe<ReverseAndRotate>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.CountAndFind", "[Unit]") {
  check_probe<CountAndFind>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.AllAnyNoneOf", "[Unit]") {
  check_probe<AllAnyNoneOf>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ForEach", "[Unit]") {
  check_probe<ForEach>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ForEachN", "[Unit]") {
  check_probe<ForEachN>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Transform", "[Unit]") {
  check_probe<Transform>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.EqualAndLexicographicalCompare",
                  "[Unit]") {
  check_probe<EqualAndLexicographicalCompare>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ReplaceAndRemove", "[Unit]") {
  check_probe<ReplaceAndRemove>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SortTwo", "[Unit]") {
  check_probe<SortTwo>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SortThree", "[Unit]") {
  check_probe<SortThree>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SortSizeT", "[Unit]") {
  check_probe<SortSizeT>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SortWithComparator", "[Unit]") {
  check_probe<SortWithComparator>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SortTwenty", "[Unit]") {
  check_probe<SortTwenty>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.PartialSortAndNthElement",
                  "[Unit]") {
  check_probe<PartialSortAndNthElement>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Unique", "[Unit]") {
  check_probe<Unique>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SortThenUnique", "[Unit]") {
  check_probe<SortThenUnique>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.MinMaxElement", "[Unit]") {
  check_probe<MinMaxElement>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.SortedSearch", "[Unit]") {
  check_probe<SortedSearch>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.NextAndPrevPermutation", "[Unit]") {
  check_probe<NextAndPrevPermutation>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Numeric", "[Unit]") {
  check_probe<Numeric>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.RangesAlgorithmsAsLocalObjects",
                  "[Unit]") {
  check_probe<RangesAlgorithmsAsLocalObjects>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ViewsIota", "[Unit]") {
  check_probe<ViewsIota>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ViewsPipeline", "[Unit]") {
  check_probe<ViewsPipeline>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.CmathStd", "[Unit]") {
  check_probe<CmathStd>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.CmathUnqualified", "[Unit]") {
  check_probe<CmathUnqualified>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.CmathClassify", "[Unit]") {
  check_probe<CmathClassify>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.NumericLimitsAtRunTime", "[Unit]") {
  check_probe<NumericLimitsAtRunTime>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.NumericLimitsConstexpr", "[Unit]") {
  check_probe<NumericLimitsConstexpr>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Pair", "[Unit]") {
  check_probe<Pair>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Tuple", "[Unit]") {
  check_probe<Tuple>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Optional", "[Unit]") {
  check_probe<Optional>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Complex", "[Unit]") {
  check_probe<Complex>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Span", "[Unit]") {
  check_probe<Span>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.InitializerList", "[Unit]") {
  check_probe<InitializerList>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.FunctionObjects", "[Unit]") {
  check_probe<FunctionObjects>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.AlgCountAndAllAnyNoneOf",
                  "[Unit]") {
  check_probe<AlgCountAndAllAnyNoneOf>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.AlgFind", "[Unit]") {
  check_probe<AlgFind>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.AlgForEachAndTransform", "[Unit]") {
  check_probe<AlgForEachAndTransform>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.AlgEqualAndMinMaxElement",
                  "[Unit]") {
  check_probe<AlgEqualAndMinMaxElement>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.AlgSort", "[Unit]") {
  check_probe<AlgSort>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.Cpp20Reimplementations", "[Unit]") {
  check_probe<Cpp20Reimplementations>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.GslAtMakeArraySquare", "[Unit]") {
  check_probe<GslAtMakeArraySquare>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ThreadsFill", "[Unit]") {
  check_threads_probe<ThreadsFill>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ThreadsStdArrayLoop", "[Unit]") {
  check_threads_probe<ThreadsStdArrayLoop>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ThreadsSort", "[Unit]") {
  check_threads_probe<ThreadsSort>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ThreadsMinMaxAccumulate",
                  "[Unit]") {
  check_threads_probe<ThreadsMinMaxAccumulate>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ThreadsArrayEqual", "[Unit]") {
  check_threads_probe<ThreadsArrayEqual>();
}

#if not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.RangesSort", "[Unit]") {
  check_probe<RangesSort>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.RangesForEachAndTransform",
                  "[Unit]") {
  check_probe<RangesForEachAndTransform>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.RangesQueries", "[Unit]") {
  check_probe<RangesQueries>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.RangesFillAndReverse", "[Unit]") {
  check_probe<RangesFillAndReverse>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.ViewsReverse", "[Unit]") {
  check_probe<ViewsReverse>();
}
SPECTRE_TEST_CASE("Unit.Scratch.StdOnDevice.TupleTieWithIgnore", "[Unit]") {
  check_probe<TupleTieWithIgnore>();
}
#endif  // not(defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))
