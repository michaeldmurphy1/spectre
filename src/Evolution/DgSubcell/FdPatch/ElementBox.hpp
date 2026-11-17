// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Evolution/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd_patch {
namespace ElementBox_detail {
template <size_t Dim>
using cell_geometry_tags =
    tmpl::list<subcell::Tags::Coordinates<Dim, Frame::Grid>,
               fd::Tags::InverseJacobianLogicalToGrid<Dim>,
               fd::Tags::DetInverseJacobianLogicalToGrid>;

// The time-independent geometry at the cell centers, computed once by the patch
// (see `FdPatchElements::host_cell_geometry`)
template <size_t Dim>
struct CellGeometry : db::SimpleTag {
  using type = Variables<cell_geometry_tags<Dim>>;
};

template <size_t Dim>
using inertial_cell_geometry_tags =
    tmpl::list<subcell::Tags::Coordinates<Dim, Frame::Inertial>,
               fd::Tags::InverseJacobianLogicalToInertial<Dim>,
               fd::Tags::DetInverseJacobianLogicalToInertial>;

// The inertial geometry at the cell centers at the time of the DataBox,
// computed by the patch (see `FdPatchElements::host_inertial_cell_geometry`)
template <size_t Dim>
struct InertialCellGeometry : db::SimpleTag {
  using type = Variables<inertial_cell_geometry_tags<Dim>>;
};

// One tensor of the `Variables` of the simple tag `VariablesTag`
template <typename Tag, typename VariablesTag>
struct FromVariables : Tag, db::ReferenceTag {
  using base = Tag;
  using argument_tags = tmpl::list<VariablesTag>;

  static const typename Tag::type& get(
      const typename VariablesTag::type& variables) {
    return ::get<Tag>(variables);
  }
};

template <typename Tag, size_t Dim>
using FromCellGeometry = FromVariables<Tag, CellGeometry<Dim>>;
template <typename Tag, size_t Dim>
using FromInertialCellGeometry = FromVariables<Tag, InertialCellGeometry<Dim>>;

// Point the `DataVector`s of `result` into those of `source`, as the
// observer compute tags do
template <typename TensorType>
void refer_to(const gsl::not_null<TensorType*> result,
              const TensorType& source) {
  for (size_t i = 0; i < source.size(); ++i) {
    (*result)[i].set_data_ref(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        make_not_null(&const_cast<DataVector&>(source[i])));
  }
}

// The geometry for observations on the subcell grid, which is the active grid
// of the elements of patches. Events use these instead of their compute tags
// for the same items (see `evolution::patches::run_events_on_elements`), so
// that they don't compute the geometry again. They derive from the subcell
// compute tags, since events may also retrieve the items with those.
template <size_t Dim, typename Fr>
struct ObserverCoordinatesCompute
    : subcell::Tags::ObserverCoordinatesCompute<Dim, Fr> {
  using return_type =
      typename subcell::Tags::ObserverCoordinatesCompute<Dim, Fr>::return_type;
  using argument_tags = tmpl::list<subcell::Tags::Coordinates<Dim, Fr>>;
  static void function(const gsl::not_null<return_type*> observer_coordinates,
                       const return_type& coordinates) {
    refer_to(observer_coordinates, coordinates);
  }
};

template <size_t Dim>
struct ObserverInverseJacobianCompute
    : subcell::Tags::ObserverInverseJacobianCompute<Dim, Frame::ElementLogical,
                                                    Frame::Inertial> {
  using return_type = typename subcell::Tags::ObserverInverseJacobianCompute<
      Dim, Frame::ElementLogical, Frame::Inertial>::return_type;
  using argument_tags =
      tmpl::list<fd::Tags::InverseJacobianLogicalToInertial<Dim>>;
  static void function(
      const gsl::not_null<return_type*> observer_inverse_jacobian,
      const return_type& inverse_jacobian) {
    refer_to(observer_inverse_jacobian, inverse_jacobian);
  }
};

struct ObserverDetInvJacobianCompute
    : ::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                             Frame::Inertial>,
      db::ComputeTag {
  using base = ::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                      Frame::Inertial>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<fd::Tags::DetInverseJacobianLogicalToInertial>;
  static void function(
      const gsl::not_null<return_type*> observer_det_inverse_jacobian,
      const return_type& det_inverse_jacobian) {
    refer_to(observer_det_inverse_jacobian, det_inverse_jacobian);
  }
};

// The Jacobian, only computed if an event needs it (unlike
// `subcell::Tags::ObserverJacobianAndDetInvJacobianCompute`, which computes
// it with the determinant)
template <size_t Dim>
struct ObserverJacobianCompute
    : ::Events::Tags::ObserverJacobian<Dim, Frame::ElementLogical,
                                       Frame::Inertial>,
      db::ComputeTag {
  using base = ::Events::Tags::ObserverJacobian<Dim, Frame::ElementLogical,
                                                Frame::Inertial>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<::Events::Tags::ObserverInverseJacobian<
      Dim, Frame::ElementLogical, Frame::Inertial>>;
  static void function(
      const gsl::not_null<return_type*> jacobian,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian) {
    *jacobian = determinant_and_inverse(inverse_jacobian).second;
  }
};

template <size_t Dim>
using observer_geometry_tags =
    tmpl::list<ObserverCoordinatesCompute<Dim, Frame::ElementLogical>,
               ObserverCoordinatesCompute<Dim, Frame::Grid>,
               ObserverCoordinatesCompute<Dim, Frame::Inertial>,
               ObserverInverseJacobianCompute<Dim>,
               ObserverDetInvJacobianCompute, ObserverJacobianCompute<Dim>>;

// Default initialized, as by `Initialization::TimeStepping`. The stepper
// errors, the history on the host and the local time stepping tags are empty
// placeholders, since patches keep their history on the device, don't
// estimate errors and use global time stepping, so the step choosers that use
// them are an error for patches (see `change_slab_size`).
template <typename Metavariables>
using default_initialized_tags = tmpl::push_back<
    ::StepChoosers::step_chooser_simple_tags<Metavariables>,
    ::Tags::StepperErrors<typename Metavariables::system::variables_tag>,
    ::Tags::HistoryEvolvedVariables<
        typename Metavariables::system::variables_tag>,
    ::Tags::FixedLtsRatio>;

template <size_t Dim, typename Metavariables>
using initialized_tags = tmpl::list<
    Parallel::Tags::GlobalCache<Metavariables>,
    Parallel::Tags::MetavariablesImpl<Metavariables>, ::Tags::Time,
    ::Tags::TimeStepId, ::Tags::TimeStep, ::domain::Tags::Mesh<Dim>,
    ::domain::Tags::Element<Dim>, ::domain::Tags::ElementMap<Dim, Frame::Grid>,
    ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
    CellGeometry<Dim>, InertialCellGeometry<Dim>, subcell::Tags::ActiveGrid,
    subcell::Tags::TciDecision, subcell::Tags::ReconstructionOrder<Dim>,
    typename Metavariables::system::variables_tag,
    typename Metavariables::system::primitive_variables_tag,
    typename Metavariables::system::spacetime_variables_tag>;

template <size_t Dim, typename Metavariables>
using simple_tags = tmpl::append<initialized_tags<Dim, Metavariables>,
                                 default_initialized_tags<Metavariables>>;

// As `evolution::dg::Initialization::Domain` and
// `evolution::dg::subcell::Actions::SetSubcellGrid` for an element
template <size_t Dim, typename Metavariables>
using compute_tags = tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
    db::wrap_tags_in<
        Parallel::Tags::FromGlobalCache,
        tmpl::remove_duplicates<tmpl::append<
            Parallel::get_const_global_cache_tags<Metavariables>,
            Parallel::get_mutable_global_cache_tags<Metavariables>>>,
        Metavariables>,
    ::time_stepper_ref_tags<TimeStepper>,
    // The DG grid
    ::domain::Tags::LogicalCoordinates<Dim>,
    ::domain::Tags::MappedCoordinates<
        ::domain::Tags::ElementMap<Dim, Frame::Grid>,
        ::domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
    ::domain::Tags::InverseJacobianCompute<
        ::domain::Tags::ElementMap<Dim, Frame::Grid>,
        ::domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
    ::domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
        ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>>,
    ::domain::Tags::GridToInertialInverseJacobian<Dim>,
    ::domain::Tags::GridToInertialJacobian<Dim>,
    ::domain::Tags::InertialFromGridCoordinatesCompute<Dim>,
    ::domain::Tags::ElementToInertialInverseJacobian<Dim>,
    ::domain::Tags::DetInvJacobianCompute<Dim, Frame::ElementLogical,
                                          Frame::Inertial>,
    ::domain::Tags::InertialMeshVelocityCompute<Dim>,
    ::domain::Tags::MinimumGridSpacingCompute<Dim, Frame::Inertial>,
    // The subcell grid
    subcell::Tags::MeshCompute<Dim>,
    subcell::Tags::LogicalCoordinatesCompute<Dim>,
    // The grid coordinates and the logical to grid Jacobians at the cell
    // centers are computed once by the patch
    FromCellGeometry<subcell::Tags::Coordinates<Dim, Frame::Grid>, Dim>,
    FromCellGeometry<fd::Tags::InverseJacobianLogicalToGrid<Dim>, Dim>,
    FromCellGeometry<fd::Tags::DetInverseJacobianLogicalToGrid, Dim>,
    // The inertial geometry at the cell centers is computed by the patch at
    // the time of the DataBox
    FromInertialCellGeometry<subcell::Tags::Coordinates<Dim, Frame::Inertial>,
                             Dim>,
    FromInertialCellGeometry<fd::Tags::InverseJacobianLogicalToInertial<Dim>,
                             Dim>,
    FromInertialCellGeometry<fd::Tags::DetInverseJacobianLogicalToInertial,
                             Dim>,
    observer_geometry_tags<Dim>,
    // As for an element of the `DgElementArray`
    evolution::dg::Tags::EqualRateRegionsRef<
        Dim, typename Metavariables::equal_rate_regions>,
    // After the grid tags that they depend on
    ::StepChoosers::step_chooser_compute_tags<Metavariables>>>>;
}  // namespace ElementBox_detail

/// The type of the DataBox of `make_element_box`
template <size_t Dim, typename Metavariables>
using element_box_type = db::compute_databox_type<
    tmpl::append<ElementBox_detail::simple_tags<Dim, Metavariables>,
                 ElementBox_detail::compute_tags<Dim, Metavariables>>>;

/*!
 * \brief A DataBox on the host for an element of an FD patch, with what the
 * DataBox of an FD element of the `DgElementArray` holds for events and step
 * choosers.
 *
 * \details The DataBox holds the element, its DG mesh, maps, the time and step
 * size, the `tci_decision` of the element, and host copies of its evolved,
 * primitive and spacetime variables on the subcell grid. Its coordinates,
 * Jacobians and global cache items are the compute and reference items of an
 * element, except for the geometry at the cell centers, which the patch
 * computes:
 * - the time-independent grid coordinates and logical to grid Jacobians once,
 *   in the same way (`cell_geometry`, see
 *   `FdPatchElements::host_cell_geometry`);
 * - the inertial coordinates and logical to inertial Jacobians at the time of
 *   the DataBox, on the device if the mesh moves (`inertial_cell_geometry`,
 *   see `FdPatchElements::host_inertial_cell_geometry`).
 *
 * The geometry for observations (the `Events::Tags::ObserverCoordinates` and
 * the logical to inertial `Events::Tags::ObserverInverseJacobian`,
 * `Events::Tags::ObserverDetInvJacobian` and `Events::Tags::ObserverJacobian`)
 * is that of the subcell grid, which is the active grid of the elements.
 * Events can then run on the elements of patches as they do on the elements of
 * the `DgElementArray`, with the same results up to roundoff.
 *
 * The variables and the cell geometries may refer to memory owned elsewhere
 * (without copying), which must outlive the DataBox.
 */
template <typename Metavariables, size_t Dim>
element_box_type<Dim, Metavariables> make_element_box(
    Parallel::GlobalCache<Metavariables>& cache, Element<Dim> element,
    Mesh<Dim> dg_mesh, ElementMap<Dim, Frame::Grid> element_map,
    std::unique_ptr<
        ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>
        grid_to_inertial_map,
    const double time, const TimeStepId& time_step_id, TimeDelta time_step,
    const int tci_decision,
    Variables<ElementBox_detail::cell_geometry_tags<Dim>> cell_geometry,
    Variables<ElementBox_detail::inertial_cell_geometry_tags<Dim>>
        inertial_cell_geometry,
    typename Metavariables::system::variables_tag::type evolved_variables,
    typename Metavariables::system::primitive_variables_tag::type primitives,
    typename Metavariables::system::spacetime_variables_tag::type spacetime) {
  return tmpl::as_pack<
      ElementBox_detail::default_initialized_tags<Metavariables>>(
      [&]<typename... DefaultTags>(tmpl::type_<DefaultTags>... /*meta*/) {
        return db::create<ElementBox_detail::simple_tags<Dim, Metavariables>,
                          ElementBox_detail::compute_tags<Dim, Metavariables>>(
            &cache, Metavariables{}, time, time_step_id, std::move(time_step),
            std::move(dg_mesh), std::move(element), std::move(element_map),
            std::move(grid_to_inertial_map), std::move(cell_geometry),
            std::move(inertial_cell_geometry), ActiveGrid::Subcell,
            tci_decision,
            std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>{},
            std::move(evolved_variables), std::move(primitives),
            std::move(spacetime), typename DefaultTags::type{}...);
      });
}
}  // namespace evolution::dg::subcell::fd_patch
