// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariablesKokkos.hpp"

namespace gh {
/*!
 * \brief The constraint damping parameters \f$\gamma_0\f$, \f$\gamma_1\f$ and
 * \f$\gamma_2\f$ at the grid points of a batch of elements, in a Kokkos kernel
 * on `exec`.
 *
 * \details This computes the same values as the damping functions
 * `gh::Tags::DampingFunctionGamma0`, etc. on the host. The damping functions
 * at the current time are passed as `ConstraintDamping::GaussianSum`s (see
 * `ConstraintDamping::DampingFunction::as_gaussian_sum`), which change every
 * step when they depend on functions of time, e.g. in a buffer of parameters
 * that is copied to the device every step.
 *
 * \param exec the execution space instance that runs the kernel.
 * \param gamma0 the values of \f$\gamma_0\f$.
 * \param gamma1 the values of \f$\gamma_1\f$.
 * \param gamma2 the values of \f$\gamma_2\f$.
 * \param grid_coordinates the grid coordinates of the points, with the same
 * number of grid points and elements as the \f$\gamma\f$s.
 * \param damping_functions the damping functions of \f$\gamma_0\f$,
 * \f$\gamma_1\f$ and \f$\gamma_2\f$ in device memory, in this order.
 */
void constraint_gammas_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const typename ::Tags::BatchView<Tags::ConstraintGamma0>::type& gamma0,
    const typename ::Tags::BatchView<Tags::ConstraintGamma1>::type& gamma1,
    const typename ::Tags::BatchView<Tags::ConstraintGamma2>::type& gamma2,
    const typename ::Tags::BatchView<
        domain::Tags::Coordinates<3, Frame::Grid>>::type& grid_coordinates,
    const Kokkos::View<const ConstraintDamping::GaussianSum<3>*>&
        damping_functions);
}  // namespace gh
#endif  // SPECTRE_KOKKOS
