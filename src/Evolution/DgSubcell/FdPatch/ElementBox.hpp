// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
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
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Evolution/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
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
// Default initialized, as by `Initialization::TimeStepping`. The stepper
// errors, the history on the host and the local time stepping tags are empty
// placeholders, since patches keep their history on the device, don't
// estimate errors and use global time stepping, so the step choosers that use
// them are an error for patches (see `change_slab_size`).
template <typename Metavariables>
using default_initialized_tags = tmpl::push_back<
    StepChoosers::step_chooser_simple_tags<Metavariables>,
    ::Tags::StepperErrors<typename Metavariables::system::variables_tag>,
    ::Tags::HistoryEvolvedVariables<
        typename Metavariables::system::variables_tag>,
    ::Tags::FixedLtsRatio>;

template <size_t Dim, typename Metavariables>
using initialized_tags =
    tmpl::list<Parallel::Tags::GlobalCache<Metavariables>,
               Parallel::Tags::MetavariablesImpl<Metavariables>, ::Tags::Time,
               ::Tags::TimeStepId, ::Tags::TimeStep, ::domain::Tags::Mesh<Dim>,
               ::domain::Tags::Element<Dim>,
               ::domain::Tags::ElementMap<Dim, Frame::Grid>,
               ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
               subcell::Tags::ActiveGrid, subcell::Tags::TciDecision,
               subcell::Tags::ReconstructionOrder<Dim>,
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
    ::domain::Tags::MappedCoordinates<
        ::domain::Tags::ElementMap<Dim, Frame::Grid>,
        subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
        subcell::Tags::Coordinates>,
    subcell::Tags::InertialCoordinatesCompute<
        ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>>,
    fd::Tags::InverseJacobianLogicalToGridCompute<
        ::domain::Tags::ElementMap<Dim, Frame::Grid>, Dim>,
    fd::Tags::DetInverseJacobianLogicalToGridCompute<Dim>,
    fd::Tags::InverseJacobianLogicalToInertialCompute<
        ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>,
        Dim>,
    fd::Tags::DetInverseJacobianLogicalToInertialCompute<
        ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>,
        Dim>,
    // As for an element of the `DgElementArray`
    evolution::dg::Tags::EqualRateRegionsRef<
        Dim, typename Metavariables::equal_rate_regions>,
    // After the grid tags that they depend on
    StepChoosers::step_chooser_compute_tags<Metavariables>>>>;
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
 * element. Events can then run on the elements of patches as they do on the
 * elements of the `DgElementArray`.
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
            std::move(grid_to_inertial_map), ActiveGrid::Subcell, tci_decision,
            std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>{},
            std::move(evolved_variables), std::move(primitives),
            std::move(spacetime), typename DefaultTags::type{}...);
      });
}
}  // namespace evolution::dg::subcell::fd_patch
