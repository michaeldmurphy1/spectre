// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ParallelAlgorithms/Events/NormStatistics.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace evolution::patches {
/*!
 * \brief Tensors that a patch computed on the device for observations, at all
 * grid points of all its elements
 *
 * \details Patches fill this in `Patch::device_observables` for the tensors
 * that events observe (e.g. `Events::ObserveNorms`), so that the observations
 * are reduced on the device and only their results are copied to the host
 * (see `norm_statistics_on_device`).
 */
struct DeviceObservables {
  /// An observed tensor and where its components are in `values`
  struct ObservedTensor {
    std::string name;
    size_t first_component;
    /// The names of the components, as `Tensor::get_vector_of_data()`
    std::vector<std::string> component_names;
  };

  /// The components of the tensors, indexed by grid point, component and
  /// element (the layout of a `BatchedVariables`)
  Kokkos::View<double***, Kokkos::LayoutLeft> values{};
  /// The tensors, in the order of their components in `values`
  std::vector<ObservedTensor> tensors{};
  /// The determinant of the inverse Jacobian from the element logical to the
  /// inertial frame at the grid points, indexed by grid point and element
  Kokkos::View<const double**, Kokkos::LayoutStride> det_inverse_jacobian{};
  /// The quadrature weight of the grid points, which is the same for all
  /// grid points (e.g. of the finite-difference meshes of patches)
  double point_weight = 0.0;
};

/*!
 * \brief The `Events::NormStatistics` of every component of the
 * `observables` of each element, computed on the device and copied to the
 * host
 *
 * \details The result is indexed by element and component. The extra last
 * component is the constant 1, whose `integral` is the volume of the element.
 * The integrals use the Jacobian determinant `1 / det_inverse_jacobian` and
 * the `point_weight`, as `definite_integral` does on the host. Each
 * (element, component) is reduced by a team of threads, so the sums are in a
 * different order than on the host and agree to roundoff.
 *
 * This waits for `exec` to complete.
 */
Kokkos::View<Events::NormStatistics**, Kokkos::LayoutRight, Kokkos::HostSpace>
norm_statistics_on_device(const Kokkos::DefaultExecutionSpace& exec,
                          const DeviceObservables& observables);
}  // namespace evolution::patches
#endif  // SPECTRE_KOKKOS
