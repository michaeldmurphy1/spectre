// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariablesKokkos.hpp"

/// Kokkos kernels that couple the GH and GRMHD parts of the FD time derivative
/// of a batch of elements
namespace grmhd::GhValenciaDivClean::subcell {
template <typename Tag>
using Batch = typename ::Tags::BatchView<Tag>::type;

/*!
 * \brief The spacetime variables at the cell centers that the GRMHD sources
 * need (`grmhd::ValenciaDivClean::subcell::cell_spacetime_tags`), from the GH
 * variables, in a Kokkos kernel on `exec`.
 *
 * \details This computes the same values as the FD time derivative of
 * GhValenciaDivClean on the host (`subcell::TimeDerivative`), with the same
 * operations in the same order (up to fused multiply-adds, see
 * `add_stress_energy_to_dt_pi_on_device`), from the same intermediate GH
 * quantities (\f$n^a\f$, \f$g^{ab}\f$, \f$n^b\Phi_{iba}\f$ and
 * \f$\frac{1}{2}n^a n^b\Phi_{iab}\f$), which are recomputed here. The lapse,
 * shift and inverse spatial metric are copied from the arguments, e.g. those
 * that `gh::TimeDerivativeOnDevice` computes. The inertial coordinates are
 * left unchanged.
 *
 * \param exec the execution space instance that runs the kernel.
 * \param cell_spacetime_variables the spacetime variables.
 * \param spacetime_metric the spacetime metric \f$g_{ab}\f$.
 * \param pi the GH variable \f$\Pi_{ab}\f$.
 * \param phi the GH variable \f$\Phi_{iab}\f$.
 * \param lapse the lapse computed from `spacetime_metric`.
 * \param shift the shift computed from `spacetime_metric`.
 * \param inverse_spatial_metric the inverse spatial metric computed from
 * `spacetime_metric`.
 */
void grmhd_spacetime_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<
        BatchedVariables<ValenciaDivClean::subcell::cell_spacetime_tags>*>
        cell_spacetime_variables,
    const Batch<gr::Tags::SpacetimeMetric<DataVector, 3>>& spacetime_metric,
    const Batch<gh::Tags::Pi<DataVector, 3>>& pi,
    const Batch<gh::Tags::Phi<DataVector, 3>>& phi,
    const Batch<gr::Tags::Lapse<DataVector>>& lapse,
    const Batch<gr::Tags::Shift<DataVector, 3>>& shift,
    const Batch<gr::Tags::InverseSpatialMetric<DataVector, 3>>&
        inverse_spatial_metric);

/// The spatial metric, its inverse and the square root of its determinant,
/// which the fixing of the GRMHD variables and the primitive recovery use
using fixing_metric_tags =
    tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>>;

/*!
 * \brief The `fixing_metric_tags` from the spacetime metric, in a Kokkos
 * kernel on `exec`.
 *
 * \details This computes the same values as the compute tags of an element on
 * the host (`gr::Tags::SpatialMetricCompute`,
 * `gr::Tags::DetAndInverseSpatialMetricCompute` and
 * `gr::Tags::SqrtDetSpatialMetricCompute`) and
 * `FixConservativesAndComputePrims`, with the same operations in the same
 * order.
 */
void fixing_metric_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    gsl::not_null<BatchedVariables<fixing_metric_tags>*> metric,
    const Batch<gr::Tags::SpacetimeMetric<DataVector, 3>>& spacetime_metric);

/*!
 * \brief Add the trace-reversed stress energy of the fluid to
 * \f$\partial_t\Pi_{ab}\f$, in a Kokkos kernel on `exec`.
 *
 * \details This computes the same values as the FD time derivative of
 * GhValenciaDivClean on the host (`subcell::TimeDerivative`), with
 * `trace_reversed_stress_energy` and `add_stress_energy_term_to_dt_pi`. The
 * spatial metric is the spatial part of the spacetime metric. The results
 * agree to roundoff: the host computes the contractions with vectorized tensor
 * expressions, in which the compiler fuses multiplications and additions
 * differently.
 *
 * \param exec the execution space instance that runs the kernel.
 * \param dt_pi the time derivative \f$\partial_t\Pi_{ab}\f$, to which the
 * term is added.
 * \param rest_mass_density the rest mass density.
 * \param specific_internal_energy the specific internal energy.
 * \param pressure the pressure.
 * \param lorentz_factor the Lorentz factor.
 * \param spatial_velocity the spatial velocity.
 * \param magnetic_field the magnetic field.
 * \param spacetime_metric the spacetime metric \f$g_{ab}\f$.
 * \param lapse the lapse computed from `spacetime_metric`.
 * \param shift the shift computed from `spacetime_metric`.
 */
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
    const Batch<gr::Tags::Shift<DataVector, 3>>& shift);
}  // namespace grmhd::GhValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
