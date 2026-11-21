// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariablesKokkos.hpp"

namespace gh {
/*!
 * \brief The time derivative of the generalized harmonic system
 * (`gh::TimeDerivative`) with the harmonic gauge (\f$H_a = 0\f$) for a batch
 * of elements on the device.
 *
 * \details This is a GPU-specific implementation of the same equations (see
 * the policy on hot kernels in the GPU support guide), unit tested against
 * `gh::TimeDerivative`, with which it agrees to roundoff. As there, the
 * advection of the variables by the mesh velocity is left to the caller, and
 * the mesh velocity only enters the constraint damping terms.
 *
 * A single kernel per grid point would hold hundreds of temporaries and spill
 * to local memory. Instead the time derivative is split into five kernels,
 * which communicate through a cache in device memory that is allocated once:
 * 1. the lapse, shift and inverse spatial metric;
 * 2. the Christoffel symbols of the first kind with the last index raised,
 *    \f$\Gamma_{ab}{}^c\f$, and their trace, the gauge constraint
 *    \f$C_a = \Gamma_{ab}{}^b\f$ (for the harmonic gauge);
 * 3. \f$\partial_t g_{ab}\f$;
 * 4. \f$\partial_t \Pi_{ab}\f$, with one thread per grid point that computes
 *    all ten components, recomputing the cheap contractions of \f$\Pi\f$
 *    and \f$\Phi\f$ with the inverse metrics where they are used instead of
 *    storing them;
 * 5. \f$\partial_t \Phi_{iab}\f$.
 *
 * All tensor indices are compile-time constants, so the compiler keeps the
 * values in registers. The grid points of a component are contiguous in
 * memory (see `BatchedVariables`), so neighboring threads read neighboring
 * values.
 *
 * \par Performance
 * Measured on a GH200 (Release, 504 elements of \f$17^3\f$ cells, November
 * 2026), one evaluation takes 6.3 ms, 3.9 times a copy of the same number of
 * values (1.6 ms). Kernels 1, 2, 3 and 5 take 0.12, 0.83, 0.47 and 1.9 ms at
 * 65-90% of the memory bandwidth. \f$\partial_t \Pi_{ab}\f$ takes 3.4 ms
 * at 54% of the bandwidth: with 255 registers and 464 bytes of spills per
 * thread (12% occupancy) it moves 7.3 GB, while its inputs and outputs are
 * about 4.5 GB, so it could be up to ~2x faster. Splitting the components
 * among threads avoids the spills but was slower, because the threads read
 * the same inputs again: four kernels by row \f$a\f$ (125-255 registers)
 * moved 10.5 GB in 5.0 ms, ten kernels by component (126 registers) moved
 * 18.6 GB in 5.2 ms, and one kernel with a thread per component was latency
 * bound (53 ms). Ways to speed up \f$\partial_t \Pi_{ab}\f$:
 * - remove the spills while still reading each input once, e.g. by staging
 *   the values that all components use (\f$n^a\Pi_{ab}\f$, \f$g^{ab}\f$,
 *   \f$C_a\f$, ...) in shared memory for threads that each compute some
 *   components at the same grid points;
 * - split the terms (not the components) among kernels, so that each kernel
 *   reads different inputs, e.g. the terms with \f$\partial_i\Phi_{jab}\f$
 *   and \f$\partial_i\Pi_{ab}\f$ and the remaining ones;
 * - fuse kernels that read the same inputs (e.g. \f$\partial_t\Phi_{iab}\f$
 *   also reads \f$\Pi_{ab}\f$, \f$\Phi_{iab}\f$ and their derivatives), or
 *   recompute the Christoffel symbols instead of reading them from the cache
 *   if the registers allow it.
 */
class TimeDerivativeOnDevice {
 public:
  static constexpr size_t dim = 3;
  template <typename Tag>
  using Batch = typename ::Tags::BatchView<Tag>::type;

  using spacetime_metric_tag = gr::Tags::SpacetimeMetric<DataVector, dim>;
  using pi_tag = Tags::Pi<DataVector, dim>;
  using phi_tag = Tags::Phi<DataVector, dim>;
  template <typename Tag>
  using deriv = ::Tags::deriv<Tag, tmpl::size_t<dim>, Frame::Inertial>;

  /// The mesh velocity, as a tensor
  struct MeshVelocity : db::SimpleTag {
    using type = tnsr::I<DataVector, dim, Frame::Inertial>;
  };

  TimeDerivativeOnDevice() = default;

  /// The intermediate results in device memory
  using cache_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, dim>,
                 gr::Tags::InverseSpatialMetric<DataVector, dim>,
                 ::Tags::TempTensor<0, tnsr::abC<DataVector, dim>>,
                 Tags::GaugeConstraint<DataVector, dim, Frame::Inertial>>;

  /*!
   * \brief Enqueue computing the time derivatives of the batch on `exec`.
   *
   * \details All arguments are batches with the same number of grid points and
   * elements. Without a `mesh_velocity` the mesh doesn't move. With a
   * `mesh_velocity` \f$v^i\f$, the time derivatives include the advection
   * term \f$v^i \partial_i u\f$ that the host adds after
   * `gh::TimeDerivative` (in the evolution loop), so that the moving mesh
   * needs no extra pass over the derivatives.
   */
  void apply(const Kokkos::DefaultExecutionSpace& exec,
             const Batch<::Tags::dt<spacetime_metric_tag>>& dt_spacetime_metric,
             const Batch<::Tags::dt<pi_tag>>& dt_pi,
             const Batch<::Tags::dt<phi_tag>>& dt_phi,
             const Batch<deriv<spacetime_metric_tag>>& d_spacetime_metric,
             const Batch<deriv<pi_tag>>& d_pi,
             const Batch<deriv<phi_tag>>& d_phi,
             const Batch<spacetime_metric_tag>& spacetime_metric,
             const Batch<pi_tag>& pi, const Batch<phi_tag>& phi,
             const Batch<Tags::ConstraintGamma0>& gamma0,
             const Batch<Tags::ConstraintGamma1>& gamma1,
             const Batch<Tags::ConstraintGamma2>& gamma2,
             const std::optional<Batch<MeshVelocity>>& mesh_velocity);

  /// The intermediate results of the last `apply()`, e.g. for tests
  const BatchedVariables<cache_tags>& cache() const { return cache_; }

 private:
  BatchedVariables<cache_tags> cache_{};
};
}  // namespace gh
#endif  // SPECTRE_KOKKOS
