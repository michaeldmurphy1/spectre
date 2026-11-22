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
 * \ingroup GeneralRelativityGroup
 * \brief Compute shift from spacetime metric and inverse spatial metric.
 *
 * \details Computes
 * \f{align}
 *    \beta^i &= \gamma^{ij} g_{jt}
 * \f}
 * where \f$ \beta^i\f$, \f$ \gamma^{ij}\f$, and \f$g_{ab}\f$ are the shift,
 * inverse spatial metric, and spacetime metric.
 * This can be derived, e.g., from Eqs. 2.121--2.122 of Baumgarte & Shapiro.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::I<DataType, SpatialDim, Frame> shift(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void shift(
    gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);
/// @}

namespace Tags {
/*!
 * \brief Compute item for shift \f$\beta^i\f$ from the spacetime metric
 * \f$g_{ab}\f$ and the inverse spatial metric \f$\gamma^{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::Shift`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct ShiftCompute : Shift<DataType, SpatialDim, Frame>, db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeMetric<DataType, SpatialDim, Frame>,
                 InverseSpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::I<DataType, SpatialDim, Frame>;

  SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
  template <typename DataTypeT>
  KOKKOS_FUNCTION static void function(
      const gsl::not_null<tnsr::I<DataTypeT, SpatialDim, Frame>*> result,
      const tnsr::aa<DataTypeT, SpatialDim, Frame>& spacetime_metric,
      const tnsr::II<DataTypeT, SpatialDim, Frame>& inverse_spatial_metric) {
    ASSERT_DEVICE_CAPABLE(DataTypeT);
    shift<DataTypeT, SpatialDim, Frame>(result, spacetime_metric,
                                        inverse_spatial_metric);
  }

  using base = Shift<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
