// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/*!
 * \brief What a pointwise function reports at a grid point when it is called
 * with `apply_pointwise_on_device`.
 *
 * Return this from `Function::apply` instead of a `bool` if the function can
 * fail at a grid point. `apply_pointwise_on_device` reduces it over all grid
 * points to a `PointwiseStatusSummary`.
 */
struct PointwiseStatus {
  /// A property of the result, e.g. whether the function changed anything.
  bool flag = false;
  /// Whether the function failed at this grid point.
  bool failed = false;
};

/*!
 * \brief A `PointwiseStatus` reduced over all grid points of a batch of
 * elements.
 *
 * Callers typically ERROR if there are any failures, reporting the inputs at
 * the first failed grid point. Only that grid point has to be copied to the
 * host for this, so it also works when the data stays on the device.
 */
struct PointwiseStatusSummary {
  /// Whether `PointwiseStatus::flag` was `true` at any grid point.
  bool any_flag = false;
  /// The number of grid points at which the function failed.
  size_t number_of_failures = 0;
  /// The grid point and element of the first failure, i.e. the one with the
  /// smallest element index and, in that element, the smallest grid point
  /// index. Only meaningful if `number_of_failures > 0`.
  size_t first_failed_grid_point = 0;
  /// See `first_failed_grid_point`.
  size_t first_failed_element = 0;
};
