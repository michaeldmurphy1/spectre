// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FdPatch/ObservablesHostValues.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace TestHelpers::GhValenciaDivClean::fd_patch {
HostValues host_values(
    const Mesh<3>& mesh, const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,
    const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const tnsr::I<DataVector, 3, Frame::Grid>& grid_coords,
    const ConstraintDamping::GaussianSum<3>& gamma2) {
  const size_t number_of_points = mesh.number_of_grid_points();
  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto [det_spatial_metric, inverse_spatial_metric] =
      determinant_and_inverse(spatial_metric);
  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto gauge_h = make_with_value<tnsr::a<DataVector, 3>>(lapse, 0.0);
  const auto d_gauge_h = make_with_value<tnsr::ab<DataVector, 3>>(lapse, 0.0);
  Scalar<DataVector> gamma2_values{number_of_points};
  for (size_t i = 0; i < number_of_points; ++i) {
    get(gamma2_values)[i] =
        gamma2(std::array{get<0>(grid_coords)[i], get<1>(grid_coords)[i],
                          get<2>(grid_coords)[i]});
  }
  tnsr::iaa<DataVector, 3> d_spacetime_metric{number_of_points};
  partial_derivative(make_not_null(&d_spacetime_metric), spacetime_metric, mesh,
                     inverse_jacobian);
  tnsr::iaa<DataVector, 3> d_pi{number_of_points};
  partial_derivative(make_not_null(&d_pi), pi, mesh, inverse_jacobian);
  tnsr::ijaa<DataVector, 3> d_phi{number_of_points};
  partial_derivative(make_not_null(&d_phi), phi, mesh, inverse_jacobian);

  HostValues result{};
  auto& values = result.values;
  get<gr::Tags::Lapse<DataVector>>(values) = lapse;
  get<gr::Tags::DetSpatialMetric<DataVector>>(values) = det_spatial_metric;
  get<gr::Tags::InverseSpacetimeMetric<DataVector, 3>>(values) =
      inverse_spacetime_metric;
  auto& gauge = get<gh::Tags::GaugeConstraint<DataVector, 3>>(values);
  gauge = gh::gauge_constraint(gauge_h, normal_one_form, normal_vector,
                               inverse_spatial_metric, inverse_spacetime_metric,
                               pi, phi);
  auto& three_index =
      get<gh::Tags::ThreeIndexConstraint<DataVector, 3>>(values);
  three_index = gh::three_index_constraint(d_spacetime_metric, phi);
  auto& two_index = get<gh::Tags::TwoIndexConstraint<DataVector, 3>>(values);
  two_index = gh::two_index_constraint(d_gauge_h, normal_one_form,
                                       normal_vector, inverse_spatial_metric,
                                       inverse_spacetime_metric, pi, phi, d_pi,
                                       d_phi, gamma2_values, three_index);
  auto& four_index = get<gh::Tags::FourIndexConstraint<DataVector, 3>>(values);
  four_index = gh::four_index_constraint(d_phi);
  auto& f_constraint = get<gh::Tags::FConstraint<DataVector, 3>>(values);
  f_constraint =
      gh::f_constraint(gauge_h, d_gauge_h, normal_one_form, normal_vector,
                       inverse_spatial_metric, inverse_spacetime_metric, pi,
                       phi, d_pi, d_phi, gamma2_values, three_index);
  auto& energy = get<gh::Tags::ConstraintEnergy<DataVector, 3>>(values);
  energy = gh::constraint_energy(gauge, f_constraint, two_index, three_index,
                                 four_index, inverse_spatial_metric,
                                 det_spatial_metric);
  get<::Tags::PointwiseL2Norm<gh::Tags::GaugeConstraint<DataVector, 3>>>(
      values) = pointwise_l2_norm(gauge);
  get<::Tags::PointwiseL2Norm<gh::Tags::ConstraintEnergy<DataVector, 3>>>(
      values) = pointwise_l2_norm(energy);
  get<hydro::Tags::MassWeightedInternalEnergy<DataVector>>(values) =
      hydro::mass_weighted_internal_energy(tilde_d, specific_internal_energy);
  get<hydro::Tags::MassWeightedKineticEnergy<DataVector>>(values) =
      hydro::mass_weighted_kinetic_energy(tilde_d, lorentz_factor);
  get<hydro::Tags::TildeDUnboundUtCriterion<DataVector>>(values) =
      hydro::tilde_d_unbound_ut_criterion(tilde_d, lorentz_factor,
                                          spatial_velocity, spatial_metric,
                                          lapse, shift);
  get<hydro::Tags::RestMassDensity<DataVector>>(values) = rest_mass_density;
  get<gh::Tags::Phi<DataVector, 3>>(values) = phi;
  return result;
}
}  // namespace TestHelpers::GhValenciaDivClean::fd_patch
