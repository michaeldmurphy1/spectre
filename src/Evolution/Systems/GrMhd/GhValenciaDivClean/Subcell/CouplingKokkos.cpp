// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ThreePlusOneKokkos.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::GhValenciaDivClean::subcell {
namespace {
template <typename Tag>
using deriv = ::Tags::deriv<Tag, tmpl::size_t<3>, Frame::Inertial>;

// Writes `value` to the grid point `(i, e)` of the batch tensor `batch`
template <typename BatchTensor, typename TensorType>
KOKKOS_FORCEINLINE_FUNCTION void write_at(const BatchTensor& batch,
                                          const TensorType& value,
                                          const size_t i, const size_t e) {
  for (size_t component = 0; component < value.size(); ++component) {
    batch[component](i, e) = value[component];
  }
}

Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>,
                      Kokkos::IndexType<size_t>>
policy(const Kokkos::DefaultExecutionSpace& exec, const size_t number_of_points,
       const size_t number_of_elements) {
  return {exec, {0, 0}, {number_of_points, number_of_elements}};
}
}  // namespace

void grmhd_spacetime_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<computed_cell_spacetime_tags>*>
        computed_cell_spacetime,
    const Batch<gr::Tags::SpacetimeMetric<DataVector, 3>>& spacetime_metric,
    const Batch<gh::Tags::Pi<DataVector, 3>>& pi,
    const Batch<gh::Tags::Phi<DataVector, 3>>& phi,
    const Batch<gr::Tags::Lapse<DataVector>>& lapse,
    const Batch<gr::Tags::Shift<DataVector, 3>>& shift,
    const Batch<gr::Tags::InverseSpatialMetric<DataVector, 3>>&
        inverse_spatial_metric) {
  const size_t number_of_points =
      computed_cell_spacetime->number_of_grid_points();
  const size_t number_of_elements =
      computed_cell_spacetime->number_of_elements();
  ASSERT((get<0, 0>(spacetime_metric).extent(0) == number_of_points and
          get<0, 0>(spacetime_metric).extent(1) == number_of_elements),
         "The GH variables must have "
             << number_of_points << " grid points and " << number_of_elements
             << " elements");
  const auto& vars = *computed_cell_spacetime;
  const auto d_lapse_out =
      get<::Tags::BatchView<deriv<gr::Tags::Lapse<DataVector>>>>(vars);
  const auto d_shift_out =
      get<::Tags::BatchView<deriv<gr::Tags::Shift<DataVector, 3>>>>(vars);
  const auto sqrt_det_spatial_metric_out =
      get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(vars);
  const auto extrinsic_curvature_out =
      get<::Tags::BatchView<gr::Tags::ExtrinsicCurvature<DataVector, 3>>>(vars);

  Kokkos::parallel_for(
      "grmhd::GhValenciaDivClean::subcell::grmhd_spacetime_on_device",
      policy(exec, number_of_points, number_of_elements),
      KOKKOS_LAMBDA(const size_t i, const size_t e) {
        const auto g = make_at_index(spacetime_metric, i, e);
        const auto pi_i = make_at_index(pi, i, e);
        const auto phi_i = make_at_index(phi, i, e);
        const auto alpha = make_at_index(lapse, i, e);
        const auto beta = make_at_index(shift, i, e);
        const auto inverse = make_at_index(inverse_spatial_metric, i, e);

        const tnsr::A<double, 3> normal =
            gr::device::spacetime_normal_vector(get(alpha), beta);
        const tnsr::AA<double, 3> inverse_metric =
            gr::device::inverse_spacetime_metric(get(alpha), beta, inverse);
        // As `gh::TimeDerivative`
        tnsr::ia<double, 3> phi_one_normal{};
        for (size_t n = 0; n < 3; ++n) {
          for (size_t nu = 0; nu < 4; ++nu) {
            phi_one_normal.get(n, nu) = get<0>(normal) * phi_i.get(n, 0, nu);
            for (size_t mu = 1; mu < 4; ++mu) {
              phi_one_normal.get(n, nu) +=
                  normal.get(mu) * phi_i.get(n, mu, nu);
            }
          }
        }
        tnsr::i<double, 3> half_phi_two_normals{};
        for (size_t n = 0; n < 3; ++n) {
          half_phi_two_normals.get(n) =
              get<0>(normal) * phi_one_normal.get(n, 0);
          for (size_t mu = 1; mu < 4; ++mu) {
            half_phi_two_normals.get(n) +=
                normal.get(mu) * phi_one_normal.get(n, mu);
          }
          half_phi_two_normals.get(n) *= 0.5;
        }

        // As `subcell::TimeDerivative`
        tnsr::iJ<double, 3> d_shift{};
        for (size_t k = 0; k < 3; ++k) {
          const double phi_two_normals_k = 2.0 * half_phi_two_normals.get(k);
          for (size_t j = 0; j < 3; ++j) {
            d_shift.get(k, j) = normal.get(j + 1) * phi_two_normals_k;
            for (size_t a = 0; a < 4; ++a) {
              d_shift.get(k, j) +=
                  inverse_metric.get(j + 1, a) * phi_one_normal.get(k, a);
            }
            d_shift.get(k, j) *= get(alpha);
          }
        }
        tnsr::i<double, 3> d_lapse{};
        for (size_t k = 0; k < 3; ++k) {
          d_lapse.get(k) = -get(alpha) * half_phi_two_normals.get(k);
        }
        tnsr::ii<double, 3> extrinsic_curvature{};
        tnsr::ii<double, 3> spatial_metric{};
        for (size_t m = 0; m < 3; ++m) {
          for (size_t n = m; n < 3; ++n) {
            extrinsic_curvature.get(m, n) =
                0.5 * (pi_i.get(m + 1, n + 1) + phi_one_normal.get(m, n + 1) +
                       phi_one_normal.get(n, m + 1));
            spatial_metric.get(m, n) = g.get(m + 1, n + 1);
          }
        }
        const double det =
            gr::device::determinant_and_inverse(spatial_metric).determinant;

        write_at(d_lapse_out, d_lapse, i, e);
        write_at(d_shift_out, d_shift, i, e);
        get(sqrt_det_spatial_metric_out)(i, e) = sqrt(det);
        write_at(extrinsic_curvature_out, extrinsic_curvature, i, e);
      });
}

ValenciaDivClean::subcell::CellSpacetimeViews cell_spacetime_views(
    const BatchedVariables<computed_cell_spacetime_tags>&
        computed_cell_spacetime,
    const Batch<gr::Tags::SpacetimeMetric<DataVector, 3>>& spacetime_metric,
    const Batch<gh::Tags::Phi<DataVector, 3>>& phi,
    const Batch<gr::Tags::Lapse<DataVector>>& lapse,
    const Batch<gr::Tags::Shift<DataVector, 3>>& shift,
    const Batch<gr::Tags::InverseSpatialMetric<DataVector, 3>>&
        inverse_spatial_metric) {
  ValenciaDivClean::subcell::CellSpacetimeViews result{};
  tmpl::for_each<computed_cell_spacetime_tags>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        tuples::get<::Tags::BatchView<Tag>>(result) =
            get<::Tags::BatchView<Tag>>(computed_cell_spacetime);
      });
  tuples::get<::Tags::BatchView<gr::Tags::Lapse<DataVector>>>(result) = lapse;
  tuples::get<::Tags::BatchView<gr::Tags::Shift<DataVector, 3>>>(result) =
      shift;
  tuples::get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, 3>>>(
      result) = inverse_spatial_metric;
  auto& spatial_metric =
      tuples::get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
          result);
  auto& d_spatial_metric = tuples::get<
      ::Tags::BatchView<deriv<gr::Tags::SpatialMetric<DataVector, 3>>>>(result);
  for (size_t m = 0; m < 3; ++m) {
    for (size_t n = m; n < 3; ++n) {
      spatial_metric.get(m, n) = spacetime_metric.get(m + 1, n + 1);
      for (size_t k = 0; k < 3; ++k) {
        d_spatial_metric.get(k, m, n) = phi.get(k, m + 1, n + 1);
      }
    }
  }
  return result;
}

void fixing_metric_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<fixing_metric_tags>*> metric,
    const Batch<gr::Tags::SpacetimeMetric<DataVector, 3>>& spacetime_metric) {
  const size_t number_of_points = metric->number_of_grid_points();
  const size_t number_of_elements = metric->number_of_elements();
  ASSERT((get<0, 0>(spacetime_metric).extent(0) == number_of_points and
          get<0, 0>(spacetime_metric).extent(1) == number_of_elements),
         "The spacetime metric must have "
             << number_of_points << " grid points and " << number_of_elements
             << " elements, but has " << (get<0, 0>(spacetime_metric).extent(0))
             << " and " << (get<0, 0>(spacetime_metric).extent(1)));
  const auto spatial_metric_out =
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(*metric);
  const auto inverse_spatial_metric_out =
      get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, 3>>>(
          *metric);
  const auto sqrt_det_spatial_metric_out =
      get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
          *metric);
  Kokkos::parallel_for(
      "grmhd::GhValenciaDivClean::subcell::fixing_metric_on_device",
      policy(exec, number_of_points, number_of_elements),
      KOKKOS_LAMBDA(const size_t i, const size_t e) {
        const auto g = make_at_index(spacetime_metric, i, e);
        const tnsr::ii<double, 3> spatial_metric =
            gr::device::spatial_metric(g);
        const auto [det, inverse] =
            gr::device::determinant_and_inverse(spatial_metric);
        write_at(spatial_metric_out, spatial_metric, i, e);
        write_at(inverse_spatial_metric_out, inverse, i, e);
        get(sqrt_det_spatial_metric_out)(i, e) = sqrt(det);
      });
}

void add_stress_energy_to_dt_pi_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const Batch<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>& dt_pi,
    const Batch<hydro::Tags::RestMassDensity<DataVector>>& rest_mass_density,
    const Batch<hydro::Tags::SpecificInternalEnergy<DataVector>>&
        specific_internal_energy,
    const Batch<hydro::Tags::Pressure<DataVector>>& pressure,
    const Batch<hydro::Tags::LorentzFactor<DataVector>>& lorentz_factor,
    const Batch<hydro::Tags::SpatialVelocity<DataVector, 3>>& spatial_velocity,
    const Batch<hydro::Tags::MagneticField<DataVector, 3>>& magnetic_field,
    const Batch<gr::Tags::SpacetimeMetric<DataVector, 3>>& spacetime_metric,
    const Batch<gr::Tags::Lapse<DataVector>>& lapse,
    const Batch<gr::Tags::Shift<DataVector, 3>>& shift) {
  const size_t number_of_points = get<0, 0>(dt_pi).extent(0);
  const size_t number_of_elements = get<0, 0>(dt_pi).extent(1);
  Kokkos::parallel_for(
      "grmhd::GhValenciaDivClean::subcell::"
      "add_stress_energy_to_dt_pi_on_device",
      policy(exec, number_of_points, number_of_elements),
      KOKKOS_LAMBDA(const size_t i, const size_t e) {
        const auto g = make_at_index(spacetime_metric, i, e);
        const auto alpha = make_at_index(lapse, i, e);
        const auto beta = make_at_index(shift, i, e);
        const auto velocity = make_at_index(spatial_velocity, i, e);
        const auto field = make_at_index(magnetic_field, i, e);
        const auto w = make_at_index(lorentz_factor, i, e);

        // As the tensor expressions in `subcell::TimeDerivative`, which sum
        // the terms of the contractions from the first to the last
        tnsr::i<double, 3> velocity_one_form{};
        tnsr::i<double, 3> field_one_form{};
        for (size_t k = 0; k < 3; ++k) {
          velocity_one_form.get(k) = velocity.get(0) * g.get(k + 1, 1);
          field_one_form.get(k) = field.get(0) * g.get(k + 1, 1);
          for (size_t j = 1; j < 3; ++j) {
            velocity_one_form.get(k) += velocity.get(j) * g.get(k + 1, j + 1);
            field_one_form.get(k) += field.get(j) * g.get(k + 1, j + 1);
          }
        }
        Scalar<double> field_squared{};
        Scalar<double> field_dot_velocity{};
        get(field_squared) = field.get(0) * field_one_form.get(0);
        get(field_dot_velocity) = velocity.get(0) * field_one_form.get(0);
        for (size_t j = 1; j < 3; ++j) {
          get(field_squared) += field.get(j) * field_one_form.get(j);
          get(field_dot_velocity) += velocity.get(j) * field_one_form.get(j);
        }
        Scalar<double> one_over_w_squared{};
        get(one_over_w_squared) = 1.0 / square(get(w));

        tnsr::aa<double, 3> stress_energy{};
        tnsr::a<double, 3> four_velocity_buffer{};
        tnsr::a<double, 3> comoving_field_buffer{};
        trace_reversed_stress_energy(
            make_not_null(&stress_energy), make_not_null(&four_velocity_buffer),
            make_not_null(&comoving_field_buffer),
            make_at_index(rest_mass_density, i, e), velocity_one_form,
            field_one_form, field_squared, field_dot_velocity, w,
            one_over_w_squared, make_at_index(pressure, i, e),
            make_at_index(specific_internal_energy, i, e), g, beta, alpha);
        auto dt_pi_i = make_at_index(dt_pi, i, e);
        add_stress_energy_term_to_dt_pi(make_not_null(&dt_pi_i), stress_energy,
                                        alpha);
        write_at(dt_pi, dt_pi_i, i, e);
      });
}
}  // namespace grmhd::GhValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
