// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
/// The reconstructed primitive variables on the faces of the FD grid, in the
/// order of the ghost variables (`subcell::PrimitiveGhostVariables`) that they
/// are reconstructed from.
using reconstructed_face_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::Temperature<DataVector>,
               hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
               hydro::Tags::MagneticField<DataVector, 3>,
               hydro::Tags::DivergenceCleaningField<DataVector>>;
static_assert(
    std::is_same_v<reconstructed_face_tags,
                   subcell::PrimitiveGhostVariables::ghost_variables_tag_list>,
    "The reconstructed face variables must be the ghost variables in the same "
    "order, so that the reconstruction can write the ghost variables directly "
    "into a batch of face variables.");

/*!
 * \brief The variables on one side of the faces of the FD grid that the FD
 * time derivative computes from the reconstructed primitive variables.
 *
 * These are the variables that `subcell::TimeDerivative` holds on each side of
 * the faces in each direction (the arguments of `dg_package_data` of the
 * boundary corrections `Rusanov` and `Hll`): the reconstructed primitive
 * variables, the other primitive variables, the evolved variables, the
 * fluxes, the temporaries of the boundary corrections, the spatial metric, and
 * the normal vector and covector.
 *
 * The `reconstructed_face_tags` come first, so that the reconstruction writes
 * them directly into a batch of face variables (see
 * `reconstructed_face_views`).
 */
using face_tags = tmpl::append<
    reconstructed_face_tags,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, 3>,
               hydro::Tags::LorentzFactor<DataVector>,
               hydro::Tags::Pressure<DataVector>>,
    tmpl::list<::grmhd::ValenciaDivClean::Tags::TildeD,
               ::grmhd::ValenciaDivClean::Tags::TildeYe,
               ::grmhd::ValenciaDivClean::Tags::TildeTau,
               ::grmhd::ValenciaDivClean::Tags::TildeS<>,
               ::grmhd::ValenciaDivClean::Tags::TildeB<>,
               ::grmhd::ValenciaDivClean::Tags::TildePhi>,
    db::wrap_tags_in<::Tags::Flux,
                     tmpl::list<::grmhd::ValenciaDivClean::Tags::TildeD,
                                ::grmhd::ValenciaDivClean::Tags::TildeYe,
                                ::grmhd::ValenciaDivClean::Tags::TildeTau,
                                ::grmhd::ValenciaDivClean::Tags::TildeS<>,
                                ::grmhd::ValenciaDivClean::Tags::TildeB<>,
                                ::grmhd::ValenciaDivClean::Tags::TildePhi>,
                     tmpl::size_t<3>, Frame::Inertial>,
    tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
               hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
               gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>,
               evolution::dg::Actions::detail::NormalVector<3>,
               evolution::dg::Tags::NormalCovector<3>>>;
static_assert(tmpl::size<face_tags>::value ==
                  tmpl::size<tmpl::remove_duplicates<face_tags>>::value,
              "A face variable is listed twice.");

/*!
 * \brief The `face_tags` without the fluxes and the normal vector and
 * covector, in the same order.
 *
 * The FD time derivative on the device computes these where it needs them
 * (see `BoundaryCorrections::fluxes_and_package_data_on_device`) instead of
 * keeping them on the faces.
 */
using device_face_tags = tmpl::list_difference<
    face_tags,
    tmpl::push_back<
        db::wrap_tags_in<::Tags::Flux,
                         tmpl::list<::grmhd::ValenciaDivClean::Tags::TildeD,
                                    ::grmhd::ValenciaDivClean::Tags::TildeYe,
                                    ::grmhd::ValenciaDivClean::Tags::TildeTau,
                                    ::grmhd::ValenciaDivClean::Tags::TildeS<>,
                                    ::grmhd::ValenciaDivClean::Tags::TildeB<>,
                                    ::grmhd::ValenciaDivClean::Tags::TildePhi>,
                         tmpl::size_t<3>, Frame::Inertial>,
        evolution::dg::Actions::detail::NormalVector<3>,
        evolution::dg::Tags::NormalCovector<3>>>;

/*!
 * \brief The variables on the faces from which
 * `BoundaryCorrections::package_data_from_reconstruction_on_device` computes
 * the packaged data: the `reconstructed_face_tags` (first, see
 * `reconstructed_face_views`) and the spacetime variables.
 */
using reconstruction_face_tags = tmpl::append<
    reconstructed_face_tags,
    tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
               gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::SqrtDetSpatialMetric<DataVector>,
               gr::Tags::InverseSpatialMetric<DataVector, 3>>>;

/*!
 * \brief The variables that a `DeferredPointwiseStatus` of the fix to the
 * atmosphere on the faces saves (see `compute_conservatives_on_device` and
 * `BoundaryCorrections::package_data_from_reconstruction_on_device`): the
 * reconstructed variables that the face variables are computed from.
 */
using face_atmosphere_snapshot_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::Temperature<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>;
}  // namespace grmhd::ValenciaDivClean::fd
