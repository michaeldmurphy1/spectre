// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

// These are in their own file because the test of the device evaluation
// includes the `.tpp` files that define the functions for the device, so nvcc
// would also compile the `DataVector` instantiations there for the device (and
// warn about them).
namespace TestHelpers::GhValenciaDivClean::fd_patch {
/// The values of the observable tensors of an FD element on the host
struct HostValues {
  tuples::TaggedTuple<
      gr::Tags::Lapse<DataVector>, gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpacetimeMetric<DataVector, 3>,
      gh::Tags::GaugeConstraint<DataVector, 3>,
      gh::Tags::TwoIndexConstraint<DataVector, 3>,
      gh::Tags::ThreeIndexConstraint<DataVector, 3>,
      gh::Tags::FourIndexConstraint<DataVector, 3>,
      gh::Tags::FConstraint<DataVector, 3>,
      gh::Tags::ConstraintEnergy<DataVector, 3>,
      ::Tags::PointwiseL2Norm<gh::Tags::GaugeConstraint<DataVector, 3>>,
      ::Tags::PointwiseL2Norm<gh::Tags::ConstraintEnergy<DataVector, 3>>,
      hydro::Tags::MassWeightedInternalEnergy<DataVector>,
      hydro::Tags::MassWeightedKineticEnergy<DataVector>,
      hydro::Tags::TildeDUnboundUtCriterion<DataVector>,
      hydro::Tags::RestMassDensity<DataVector>, gh::Tags::Phi<DataVector, 3>>
      values;
};

/// \brief The values of the observable tensors of an FD element on the host,
/// from the same functions that the compute tags call
///
/// \details The gauge is harmonic, and the constraint damping parameter
/// \f$\gamma_2\f$ is evaluated at the grid coordinates.
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
    const ConstraintDamping::GaussianSum<3>& gamma2);
}  // namespace TestHelpers::GhValenciaDivClean::fd_patch
