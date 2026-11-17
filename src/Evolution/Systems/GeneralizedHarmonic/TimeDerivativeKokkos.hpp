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
 * 4. \f$\partial_t \Pi_{ab}\f$, recomputing the cheap contractions of
 *    \f$\Pi\f$ and \f$\Phi\f$ with the inverse metrics where they are used
 *    instead of storing them. How the threads split the work is a choice
 *    (`DtPiLayout`), since computing all ten components in one thread spills
 *    registers;
 * 5. \f$\partial_t \Phi_{iab}\f$.
 *
 * All tensor indices are compile-time constants, so the compiler keeps the
 * values in registers. The grid points of a component are contiguous in
 * memory (see `BatchedVariables`), so neighboring threads read neighboring
 * values.
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

  /*!
   * \brief How the threads compute \f$\partial_t \Pi_{ab}\f$.
   *
   * \details Different layouts are only kept until benchmarks decide between
   * them. All give the same results.
   */
  enum class DtPiLayout {
    /// One kernel, in which each thread computes all components at a grid
    /// point
    Point,
    /// One kernel per row \f$a\f$, in which each thread computes the
    /// components \f$b \geq a\f$ at a grid point
    Row,
    /// One kernel per component, with one thread per grid point
    Component,
    /// One kernel, in which each thread computes one component at a grid
    /// point. The threads of the ten components at the same 32 grid points
    /// run together.
    WarpComponent
  };

  TimeDerivativeOnDevice() = default;
  explicit TimeDerivativeOnDevice(DtPiLayout dt_pi_layout)
      : dt_pi_layout_(dt_pi_layout) {}

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
   * elements. Without a `mesh_velocity` the mesh doesn't move.
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
  DtPiLayout dt_pi_layout_{DtPiLayout::Row};
  BatchedVariables<cache_tags> cache_{};
};
}  // namespace gh
#endif  // SPECTRE_KOKKOS
