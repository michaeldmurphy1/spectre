// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Events {
/*!
 * \brief The values over the grid points of an element from which
 * `Events::ObserveNorms` computes all its norms of a tensor component
 *
 * \details For a component \f$u\f$ with values \f$u_i\f$ at the grid points,
 * Jacobian determinant \f$J_i\f$ and quadrature weights \f$w_i\f$ these are
 * \f$\max_i u_i\f$, \f$\min_i u_i\f$, \f$\sum_i |u_i|\f$, \f$\sum_i u_i^2\f$,
 * \f$\sum_i w_i J_i |u_i|\f$, \f$\sum_i w_i J_i u_i^2\f$ and
 * \f$\sum_i w_i J_i u_i\f$. Passing them to
 * `ObserveNorms::contribute_statistics` gives the same observation as
 * evaluating the tensor in the DataBox of the element, which lets them be
 * computed elsewhere, e.g. on a device.
 */
struct NormStatistics {
  double max;
  double min;
  double sum_abs;
  double sum_square;
  double integral_abs;
  double integral_square;
  double integral;
};

/// The names of the components of an observed tensor and their
/// `NormStatistics` on an element, by tensor name (see
/// `ObserveNorms::contribute_statistics`)
using NormStatisticsOfTensors =
    std::unordered_map<std::string, std::pair<std::vector<std::string>,
                                              std::vector<NormStatistics>>>;
}  // namespace Events
