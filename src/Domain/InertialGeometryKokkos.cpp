// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/InertialGeometryKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <cstddef>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/DeviceMap.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace {
template <bool WithMeshVelocity>
struct InertialGeometry {
  using return_tags = inertial_geometry_tags<3, WithMeshVelocity>;
  using argument_tags =
      tmpl::push_back<logical_to_grid_geometry_tags<3>,
                      ::Tags::PerElement<Tags::DeviceGridToInertial<3>>>;

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> inertial_coords,
      const gsl::not_null<
          InverseJacobian<double, 3, Frame::ElementLogical, Frame::Inertial>*>
          inverse_jacobian,
      const gsl::not_null<Scalar<double>*> det_inverse_jacobian,
      const tnsr::I<double, 3, Frame::Grid>& grid_coords,
      const InverseJacobian<double, 3, Frame::ElementLogical, Frame::Grid>&
          logical_to_grid_inverse_jacobian,
      const CoordinateMaps::DeviceMap<3>& grid_to_inertial) {
    tnsr::I<double, 3, Frame::NoFrame> mesh_velocity{};
    compute(inertial_coords, inverse_jacobian, det_inverse_jacobian,
            make_not_null(&mesh_velocity), grid_coords,
            logical_to_grid_inverse_jacobian, grid_to_inertial);
  }

  KOKKOS_FUNCTION static void apply(
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> inertial_coords,
      const gsl::not_null<
          InverseJacobian<double, 3, Frame::ElementLogical, Frame::Inertial>*>
          inverse_jacobian,
      const gsl::not_null<Scalar<double>*> det_inverse_jacobian,
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> mesh_velocity,
      const tnsr::I<double, 3, Frame::Grid>& grid_coords,
      const InverseJacobian<double, 3, Frame::ElementLogical, Frame::Grid>&
          logical_to_grid_inverse_jacobian,
      const CoordinateMaps::DeviceMap<3>& grid_to_inertial) {
    tnsr::I<double, 3, Frame::NoFrame> velocity{};
    compute(inertial_coords, inverse_jacobian, det_inverse_jacobian,
            make_not_null(&velocity), grid_coords,
            logical_to_grid_inverse_jacobian, grid_to_inertial);
    for (size_t i = 0; i < 3; ++i) {
      mesh_velocity->get(i) = velocity.get(i);
    }
  }

 private:
  KOKKOS_FUNCTION static void compute(
      const gsl::not_null<tnsr::I<double, 3, Frame::Inertial>*> inertial_coords,
      const gsl::not_null<
          InverseJacobian<double, 3, Frame::ElementLogical, Frame::Inertial>*>
          inverse_jacobian,
      const gsl::not_null<Scalar<double>*> det_inverse_jacobian,
      const gsl::not_null<tnsr::I<double, 3, Frame::NoFrame>*> mesh_velocity,
      const tnsr::I<double, 3, Frame::Grid>& grid_coords,
      const InverseJacobian<double, 3, Frame::ElementLogical, Frame::Grid>&
          logical_to_grid_inverse_jacobian,
      const CoordinateMaps::DeviceMap<3>& grid_to_inertial) {
    tnsr::I<double, 3, Frame::NoFrame> source{};
    for (size_t i = 0; i < 3; ++i) {
      source.get(i) = grid_coords.get(i);
    }
    const CoordinateMaps::DeviceMappedPoint<3> mapped =
        grid_to_inertial(source);
    for (size_t i = 0; i < 3; ++i) {
      inertial_coords->get(i) = mapped.coords.get(i);
      mesh_velocity->get(i) = mapped.frame_velocity.get(i);
    }
    // As `domain::Tags::ElementToInertialInverseJacobian`
    for (size_t logical_i = 0; logical_i < 3; ++logical_i) {
      for (size_t inertial_i = 0; inertial_i < 3; ++inertial_i) {
        inverse_jacobian->get(logical_i, inertial_i) =
            logical_to_grid_inverse_jacobian.get(logical_i, 0) *
            mapped.inverse_jacobian.get(0, inertial_i);
        for (size_t grid_i = 1; grid_i < 3; ++grid_i) {
          inverse_jacobian->get(logical_i, inertial_i) +=
              logical_to_grid_inverse_jacobian.get(logical_i, grid_i) *
              mapped.inverse_jacobian.get(grid_i, inertial_i);
        }
      }
    }
    // As `determinant`
    const auto& t = *inverse_jacobian;
    get(*det_inverse_jacobian) =
        get<0, 0>(t) *
            (get<1, 1>(t) * get<2, 2>(t) - get<1, 2>(t) * get<2, 1>(t)) -
        get<0, 1>(t) *
            (get<1, 0>(t) * get<2, 2>(t) - get<1, 2>(t) * get<2, 0>(t)) +
        get<0, 2>(t) *
            (get<1, 0>(t) * get<2, 1>(t) - get<1, 1>(t) * get<2, 0>(t));
  }
};
}  // namespace

template <bool WithMeshVelocity>
void inertial_geometry_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<
        BatchedVariables<inertial_geometry_tags<3, WithMeshVelocity>>*>
        inertial,
    const BatchedVariables<logical_to_grid_geometry_tags<3>>& logical_to_grid,
    const Kokkos::View<const CoordinateMaps::DeviceMap<3>*>& grid_to_inertial) {
  ASSERT(inertial->number_of_grid_points() ==
                 logical_to_grid.number_of_grid_points() and
             inertial->number_of_elements() ==
                 logical_to_grid.number_of_elements() and
             grid_to_inertial.extent(0) == logical_to_grid.number_of_elements(),
         "The inertial geometry ("
             << inertial->number_of_grid_points() << " points, "
             << inertial->number_of_elements()
             << " elements), the logical to grid geometry ("
             << logical_to_grid.number_of_grid_points() << " points, "
             << logical_to_grid.number_of_elements()
             << " elements) and the maps (" << grid_to_inertial.extent(0)
             << ") must have the same number of points and elements.");
  using Function = InertialGeometry<WithMeshVelocity>;
  apply_pointwise_on_device<Function>(
      exec, inertial, typename Function::return_tags{},
      typename Function::argument_tags{},
      make_device_arguments<
          Tags::Coordinates<3, Frame::Grid>,
          Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Grid>,
          ::Tags::PerElement<Tags::DeviceGridToInertial<3>>>(
          exec,
          get<::Tags::BatchView<Tags::Coordinates<3, Frame::Grid>>>(
              logical_to_grid),
          get<::Tags::BatchView<
              Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Grid>>>(
              logical_to_grid),
          grid_to_inertial));
}

template void inertial_geometry_on_device<false>(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<inertial_geometry_tags<3, false>>*> inertial,
    const BatchedVariables<logical_to_grid_geometry_tags<3>>& logical_to_grid,
    const Kokkos::View<const CoordinateMaps::DeviceMap<3>*>& grid_to_inertial);
template void inertial_geometry_on_device<true>(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<inertial_geometry_tags<3, true>>*> inertial,
    const BatchedVariables<logical_to_grid_geometry_tags<3>>& logical_to_grid,
    const Kokkos::View<const CoordinateMaps::DeviceMap<3>*>& grid_to_inertial);
}  // namespace domain
#endif  // SPECTRE_KOKKOS
