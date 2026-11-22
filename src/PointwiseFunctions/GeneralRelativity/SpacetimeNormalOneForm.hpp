// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDeviceCapable.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
/// @{
/*!
 * \brief Computes spacetime normal one-form from lapse.
 *
 * \details If \f$\alpha\f$ is the lapse, then
 *
 * \f{align}
 *     n_t &= - \alpha \\
 *     n_i &= 0
 * \f}
 *
 * is computed.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void spacetime_normal_one_form(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> normal_one_form,
    const Scalar<DataType>& lapse);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse);
/// @}

namespace Tags {
/*!
 * \brief Compute item for spacetime normal oneform \f$n_a\f$ from
 * the lapse \f$\alpha\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeNormalOneForm`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpacetimeNormalOneFormCompute
    : SpacetimeNormalOneForm<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<Lapse<DataType>>;

  using return_type = tnsr::a<DataType, SpatialDim, Frame>;

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <typename DataTypeT>
  KOKKOS_FUNCTION static void function(
      const gsl::not_null<tnsr::a<DataTypeT, SpatialDim, Frame>*> result,
      const Scalar<DataTypeT>& lapse) {
    ASSERT_DEVICE_CAPABLE(DataTypeT);
    spacetime_normal_one_form<DataTypeT, SpatialDim, Frame>(result, lapse);
  }

  using base = SpacetimeNormalOneForm<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
