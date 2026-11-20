// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Tensor expressions for a grid point of a batch of tensors in device memory

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDeviceCapable.hpp"

#ifdef SPECTRE_KOKKOS
namespace tenex {
/// \cond
template <typename BatchTensor>
class AtPoint;

namespace detail {
template <typename IndexList, typename TensorIndexList>
struct BatchIndicesCheck;

template <typename... Indices, typename... TensorIndices>
struct BatchIndicesCheck<tmpl::list<Indices...>, tmpl::list<TensorIndices...>> {
  static constexpr bool valences_match =
      std::is_same_v<tmpl::integral_list<UpLo, TensorIndices::valence...>,
                     tmpl::integral_list<UpLo, Indices::ul...>>;
  static constexpr bool no_spacetime_index_for_spatial_index =
      (... and (not(TensorIndices::is_spacetime and
                    (Indices::index_type == IndexType::Spatial))));
};
}  // namespace detail
/// \endcond

/*!
 * \ingroup TensorExpressionsGroup
 * \brief A leaf of a tensor expression: a grid point of a batch of tensors in
 * device memory
 *
 * \details The batch is a `Tensor` of rank-2 `Kokkos::View`s indexed by grid
 * point and element, `(i, e)`, such as `Tags::BatchView`. The components are
 * read from memory where the expression uses them (see
 * `TensorExpression::get<MultiIndex>()`), like hand-written kernels that index
 * the batch directly. This keeps fewer values live than loading the whole
 * tensor at the point first, which matters for large expressions. Create it
 * with `tenex::at_point`.
 *
 * Only expressions of device-capable data types, which are evaluated with
 * compile-time multi-indices, can use this leaf.
 */
template <typename BatchTensor, typename ArgsList>
struct TensorAtPointAsExpression;

/// \cond
template <typename View, template <typename...> class Symm,
          typename... SymmValues, template <typename...> class IndexList,
          typename... Indices, template <typename...> class ArgsList,
          typename... Args>
struct TensorAtPointAsExpression<
    Tensor<View, Symm<SymmValues...>, IndexList<Indices...>>, ArgsList<Args...>>
    : public TensorExpression<
          TensorAtPointAsExpression<
              Tensor<View, Symm<SymmValues...>, IndexList<Indices...>>,
              ArgsList<Args...>>,
          std::remove_const_t<typename View::value_type>,
          typename detail::TensorAsExpressionSymm<Symm<SymmValues...>,
                                                  IndexList<Indices...>,
                                                  ArgsList<Args...>>::type,
          IndexList<Indices...>, ArgsList<Args...>> {
  using batch_type = Tensor<View, Symm<SymmValues...>, IndexList<Indices...>>;
  using type = std::remove_const_t<typename View::value_type>;
  static_assert(tt::is_device_capable_v<type>,
                "Batches of tensors in tensor expressions must hold a "
                "device-capable data type, e.g. double.");
  using symmetry = typename detail::TensorAsExpressionSymm<
      Symm<SymmValues...>, IndexList<Indices...>, ArgsList<Args...>>::type;
  using index_list = IndexList<Indices...>;
  using args_list = ArgsList<Args...>;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;

  // A leaf, like `TensorAsExpression` (see `TensorExpression` for the meaning)
  static constexpr size_t num_ops_left_child = 0;
  static constexpr size_t num_ops_right_child = 0;
  static constexpr size_t num_ops_subtree = 0;
  static constexpr size_t height_relative_to_closest_tensor_leaf_in_subtree = 0;
  static constexpr bool is_primary_end = true;
  static constexpr size_t num_ops_to_evaluate_primary_left_child = 0;
  static constexpr size_t num_ops_to_evaluate_primary_right_child = 0;
  static constexpr size_t num_ops_to_evaluate_primary_subtree = 0;
  static constexpr bool is_primary_start = false;
  static constexpr bool primary_child_subtree_contains_primary_start = false;
  static constexpr bool primary_subtree_contains_primary_start = false;

  SPECTRE_ALWAYS_INLINE constexpr explicit TensorAtPointAsExpression(
      const AtPoint<batch_type>& point)
      : point_(point) {}

  // The batch is never a LHS `Tensor`, so there is nothing to check
  template <typename LhsTensor>
  SPECTRE_ALWAYS_INLINE constexpr void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*> /*lhs_tensor*/) const {}
  template <typename LhsTensorIndices, typename LhsTensor>
  SPECTRE_ALWAYS_INLINE constexpr void assert_lhs_tensorindices_same_in_rhs(
      const gsl::not_null<LhsTensor*> /*lhs_tensor*/) const {}

  /// The component at the compile-time multi-index `MultiIndex`
  template <std::array<size_t, num_tensor_indices> MultiIndex>
  SPECTRE_ALWAYS_INLINE KOKKOS_FUNCTION type get() const {
    return point_.template component<batch_type::structure::get_storage_index(
        MultiIndex)>();
  }

 private:
  AtPoint<batch_type> point_;
};
/// \endcond

/*!
 * \ingroup TensorExpressionsGroup
 * \brief A grid point of a batch of tensors in device memory, for use in
 * tensor expressions
 *
 * \details The batch is a `Tensor` of rank-2 `Kokkos::View`s indexed by grid
 * point and element, `(i, e)`, such as `Tags::BatchView`. All components must
 * have the same layout (which is the case for the tensors of a
 * `BatchedVariables`).
 *
 * - `point(ti::a, ti::b)` is a tensor expression of the components at the grid
 *   point, read where the expression uses them.
 * - `tenex::evaluate<ti::a, ti::b>(point, expression)` stores each component
 *   of the result at the grid point as soon as it is computed.
 *
 * For example, in a kernel over the grid points `i` of the elements `e`:
 * \snippet Test_EvaluateKokkos.cpp at_point_example
 *
 * Storing each component when it is computed, rather than computing the result
 * into a local `Tensor` and storing it at the end, lets the compiler re-read
 * inputs instead of keeping them all in registers. In large kernels this
 * avoids register spills.
 *
 * The components are accessed through their data pointers, without Kokkos'
 * bounds checks (whose code would be inlined at every access). Instead,
 * `tenex::at_point` checks that the point is in bounds in debug builds.
 */
template <typename BatchTensor>
class AtPoint {
 public:
  using batch_type = BatchTensor;

  /// The grid point `point` of element `element` of `tensor`
  ///
  /// The point must be in bounds; `tenex::at_point` checks this in debug
  /// builds. The constructor is `constexpr` so that expressions holding the
  /// point can be built by the `constexpr` tensor expression operators in
  /// device code.
  SPECTRE_ALWAYS_INLINE constexpr AtPoint(const BatchTensor& tensor,
                                          const size_t point,
                                          const size_t element)
      : tensor_(&tensor), point_(point), element_(element) {}
  /// The point refers to the batch, which must outlive it
  AtPoint(const BatchTensor&& tensor, size_t point, size_t element) = delete;

  /// A tensor expression of the batch at the grid point
  template <typename... TensorIndices>
  SPECTRE_ALWAYS_INLINE KOKKOS_FUNCTION auto operator()(
      TensorIndices... /*meta*/) const {
    // The same checks as `Tensor::operator()`
    static_assert((... and tt::is_tensor_index<TensorIndices>::value),
                  "The tensor expression must be created using TensorIndex "
                  "objects to represent generic indices, e.g. ti::a, ti::b, "
                  "etc.");
    static_assert(
        tenex::tensorindex_list_is_valid<tmpl::list<TensorIndices...>>::value,
        "Cannot create a tensor expression with a repeated generic index. "
        "(Note that the concrete time indices, ti::T and ti::t, can be "
        "repeated.) If you intend to contract, ensure that the indices to "
        "contract have opposite valences.");
    using index_check =
        detail::BatchIndicesCheck<typename BatchTensor::index_list,
                                  tmpl::list<TensorIndices...>>;
    static_assert(index_check::valences_match,
                  "The valences of the generic indices in the expression do "
                  "not match the valences of the indices in the batch.");
    static_assert(index_check::no_spacetime_index_for_spatial_index,
                  "Cannot use a spacetime index for a spatial index. e.g. "
                  "Cannot do R(ti::a), where R's index is spatial, because "
                  "ti::a denotes a generic spacetime index.");
    return tenex::contract(
        TensorAtPointAsExpression<BatchTensor, tmpl::list<TensorIndices...>>{
            *this});
  }

  /// The component with storage index `StorageIndex` at the grid point
  template <size_t StorageIndex>
  SPECTRE_ALWAYS_INLINE KOKKOS_FUNCTION auto& component() const {
    const auto& view = (*tensor_)[StorageIndex];
    return view.data()[point_ * view.stride(0) + element_ * view.stride(1)];
  }

 private:
  const BatchTensor* tensor_;
  size_t point_;
  size_t element_;
};

/// \ingroup TensorExpressionsGroup
/// \brief The grid point `point` of element `element` of a batch of tensors,
/// for use in tensor expressions (see `tenex::AtPoint`)
///
/// In debug builds this checks that the point is in bounds for all components
/// (instead of Kokkos' bounds checks at every access, see `tenex::AtPoint`).
template <typename BatchTensor>
SPECTRE_ALWAYS_INLINE KOKKOS_FUNCTION AtPoint<BatchTensor> at_point(
    const BatchTensor& tensor, const size_t point, const size_t element) {
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < BatchTensor::size(); ++i) {
    ASSERT(point < tensor[i].extent(0) and element < tensor[i].extent(1),
           "Grid point " << point << " of element " << element
                         << " is out of bounds for component " << i
                         << " of the batch, which has extents ("
                         << tensor[i].extent(0) << ", " << tensor[i].extent(1)
                         << ").");
  }
#endif  // SPECTRE_DEBUG
  return {tensor, point, element};
}

/// \cond
// The point refers to the batch, which must outlive it
template <typename BatchTensor>
AtPoint<BatchTensor> at_point(const BatchTensor&& tensor, size_t point,
                              size_t element) = delete;
/// \endcond

namespace detail {
/// Assigns values to the components of a batch at a grid point (see
/// `evaluate_components`)
template <typename BatchTensor>
struct AssignToPoint {
  AtPoint<BatchTensor> point;

  template <size_t StorageIndex, typename T>
  SPECTRE_ALWAYS_INLINE KOKKOS_FUNCTION void assign(const T& value) const {
    point.template component<StorageIndex>() = value;
  }
};
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a tensor expression and store the result at a grid point of
 * a batch of tensors in device memory
 *
 * \details Each component of the result is stored as soon as it is computed.
 * See `tenex::AtPoint` and `tenex::evaluate`.
 */
template <auto&... LhsTensorIndices, typename BatchTensor, typename Derived,
          typename RhsDataType, typename RhsSymmetry, typename RhsIndexList,
          typename... RhsTensorIndices>
SPECTRE_ALWAYS_INLINE KOKKOS_FUNCTION void evaluate(
    const AtPoint<BatchTensor>& lhs,
    const TensorExpression<Derived, RhsDataType, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  using lhs_data_type =
      std::remove_const_t<typename BatchTensor::type::value_type>;
  static_assert(not std::is_const_v<typename BatchTensor::type::value_type>,
                "Cannot evaluate into a batch of const data.");
  using rhs_expression_type =
      typename std::decay_t<decltype(~rhs_tensorexpression)>;
  detail::assert_valid_evaluation<
      lhs_data_type, typename BatchTensor::symmetry,
      typename BatchTensor::index_list, rhs_expression_type,
      std::decay_t<decltype(LhsTensorIndices)>...>();
  detail::evaluate_components<
      detail::LhsComponentMap<BatchTensor, rhs_expression_type,
                              std::decay_t<decltype(LhsTensorIndices)>...>>(
      detail::AssignToPoint<BatchTensor>{lhs}, ~rhs_tensorexpression,
      std::make_index_sequence<BatchTensor::size()>{});
}
}  // namespace tenex
#endif  // SPECTRE_KOKKOS
