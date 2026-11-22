// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/FdPatch/GeometryKokkos.hpp"
#include "Evolution/Patches/DeviceComputeTagsKokkos.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/CouplingKokkos.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivativeKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
#include "DataStructures/BatchedVariablesKokkos.hpp"

namespace grmhd::GhValenciaDivClean::fd_patch {
/// The compute tags of the DataBox of an element on the host that
/// `EvolveGhValenciaDivClean` adds
/// (`gh::Actions::InitializeGhAnd3Plus1Variables` and the square root of the
/// determinant of the spatial metric), so that the DataBoxes of the elements of
/// patches for events hold them too
using element_box_compute_tags = tmpl::push_back<
    typename gh::Actions::InitializeGhAnd3Plus1Variables<3>::compute_tags,
    gr::Tags::SqrtDetSpatialMetricCompute<DataVector, 3, Frame::Inertial>>;

namespace ObservablesKokkos_detail {
using View = evolution::patches::device_compute::View;
using ElementRange = evolution::patches::device_compute::ElementRange;

using DetSpatialMetric = gr::Tags::DetSpatialMetric<DataVector>;
using gauge_source_tags =
    tmpl::list<gh::Tags::GaugeH<DataVector, 3, Frame::Inertial>,
               gh::Tags::SpacetimeDerivGaugeH<DataVector, 3, Frame::Inertial>>;
using gamma_tags =
    tmpl::list<gh::Tags::ConstraintGamma0, gh::Tags::ConstraintGamma1,
               gh::Tags::ConstraintGamma2>;

// The first component of `Tag` in a `BatchedVariables` of `TagsList`
template <typename Tag, typename TagsList>
constexpr size_t first_component() {
  size_t result = 0;
  bool found = false;
  tmpl::for_each<TagsList>([&]<typename T>(tmpl::type_<T> /*meta*/) {
    if (std::is_same_v<T, Tag>) {
      found = true;
    }
    if (not found) {
      result += T::type::size();
    }
  });
  return result;
}

// The components of `Tag` of the elements of `range` in `batch`
template <typename Tag, typename TagsList>
View components(const BatchedVariables<TagsList>& batch,
                const ElementRange& range) {
  constexpr size_t first = first_component<Tag, TagsList>();
  return Kokkos::subview(batch.view(), Kokkos::ALL(),
                         std::pair{first, first + Tag::type::size()},
                         std::pair{range.first, range.first + range.size});
}

// The determinant of the spatial metric, as
// `gr::Tags::DetAndInverseSpatialMetricCompute`
void det_spatial_metric(const Kokkos::DefaultExecutionSpace& exec,
                        const View& result, const View& spatial_metric);

// The constraint damping parameters at the grid coordinates
void constraint_gammas(
    const Kokkos::DefaultExecutionSpace& exec, const View& gamma0,
    const View& gamma1, const View& gamma2, const View& grid_coordinates,
    const Kokkos::View<const ConstraintDamping::GaussianSum<3>*>&
        damping_functions);

}  // namespace ObservablesKokkos_detail

/*!
 * \brief What GhValenciaDivClean FD patches provide to evaluate observable
 * tensors on the device (see `evolution::patches::device_compute`)
 *
 * \details The batches of the patch:
 * - the evolved and primitive variables,
 * - the spatial metric, its inverse and the square root of its determinant
 *   that the patch computes for the fixing (which the host computes with the
 *   compute tags of the element),
 * - the geometry at the cell centers at the time of the observation,
 *
 * and tensors it computes when they are needed:
 * - the determinant of the spatial metric,
 * - the gauge source function and its derivatives, which vanish in the
 *   harmonic gauge (the only gauge of patches, see `InitializePatch`),
 * - the constraint damping parameters, from the damping functions at the
 *   time of the observation.
 *
 * The other tensors are evaluated from the compute tags of the elements
 * (`compute_tags`) and of the events.
 */
template <typename PrimitiveTagsList>
struct DeviceObservableSources {
  using View = ObservablesKokkos_detail::View;
  using ElementRange = ObservablesKokkos_detail::ElementRange;
  using evolved_tags = subcell::evolved_tags_on_device;
  using primitive_tags = PrimitiveTagsList;
  using fixing_metric_tags = subcell::fixing_metric_tags;
  using geometry_tags = evolution::dg::subcell::fd_patch::cell_geometry_tags;
  using computed_tags =
      tmpl::append<tmpl::list<ObservablesKokkos_detail::DetSpatialMetric>,
                   ObservablesKokkos_detail::gauge_source_tags,
                   ObservablesKokkos_detail::gamma_tags>;
  /// The tensors the patch provides
  using base_tags =
      tmpl::append<evolved_tags, primitive_tags, fixing_metric_tags,
                   geometry_tags, computed_tags>;
  /// The compute tags of the elements
  using compute_tags = element_box_compute_tags;

  const BatchedVariables<evolved_tags>* evolved_variables;
  const BatchedVariables<primitive_tags>* primitives;
  const BatchedVariables<fixing_metric_tags>* fixing_metric;
  const BatchedVariables<geometry_tags>* cell_geometry;
  /// The grid coordinates of the cell centers, indexed by grid point,
  /// component and element
  View grid_coordinates;
  /// The damping functions of the constraint damping parameters at the time
  /// of the observation
  Kokkos::View<const ConstraintDamping::GaussianSum<3>*> damping_functions;
  Mesh<3> mesh;
  /// The quadrature weight of all cell centers
  double weight;
  evolution::patches::device_compute::Scratch* buffers;

  size_t number_of_grid_points() const {
    return evolved_variables->number_of_grid_points();
  }
  size_t number_of_elements() const {
    return evolved_variables->number_of_elements();
  }
  const Mesh<3>& subcell_mesh() const { return mesh; }
  double point_weight() const { return weight; }
  evolution::patches::device_compute::Scratch& scratch() const {
    return *buffers;
  }

  /// The batch of `Tag` (one of the `base_tags`) for the elements of `range`
  template <typename Tag, typename Evaluator>
  View base(const Kokkos::DefaultExecutionSpace& exec,
            const ElementRange& range, Evaluator& evaluator) const {
    namespace detail = ObservablesKokkos_detail;
    if constexpr (tmpl::list_contains_v<evolved_tags, Tag>) {
      return detail::components<Tag>(*evolved_variables, range);
    } else if constexpr (tmpl::list_contains_v<primitive_tags, Tag>) {
      return detail::components<Tag>(*primitives, range);
    } else if constexpr (tmpl::list_contains_v<fixing_metric_tags, Tag>) {
      return detail::components<Tag>(*fixing_metric, range);
    } else if constexpr (tmpl::list_contains_v<geometry_tags, Tag>) {
      return detail::components<Tag>(*cell_geometry, range);
    } else if constexpr (std::is_same_v<Tag, detail::DetSpatialMetric>) {
      const View result = evaluator.scratch("DetSpatialMetric", 1);
      detail::det_spatial_metric(
          exec, result,
          evaluator.template get<gr::Tags::SpatialMetric<DataVector, 3>>());
      return result;
    } else if constexpr (tmpl::list_contains_v<detail::gauge_source_tags,
                                               Tag>) {
      const View result =
          evaluator.scratch(db::tag_name<Tag>(), Tag::type::size());
      Kokkos::deep_copy(exec, result, 0.0);
      return result;
    } else {
      static_assert(tmpl::list_contains_v<detail::gamma_tags, Tag>);
      const std::array<View, 3> gammas{
          evaluator.scratch("ConstraintGamma0", 1),
          evaluator.scratch("ConstraintGamma1", 1),
          evaluator.scratch("ConstraintGamma2", 1)};
      detail::constraint_gammas(
          exec, gammas[0], gammas[1], gammas[2],
          Kokkos::subview(grid_coordinates, Kokkos::ALL(), Kokkos::ALL(),
                          std::pair{range.first, range.first + range.size}),
          damping_functions);
      return gammas[tmpl::index_of<detail::gamma_tags, Tag>::value];
    }
  }
};
}  // namespace grmhd::GhValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
