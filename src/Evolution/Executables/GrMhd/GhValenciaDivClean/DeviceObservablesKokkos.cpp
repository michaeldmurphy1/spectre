// Distributed under the MIT License.
// See LICENSE.txt for details.

// The evaluation of the observable tensors of FD patches on the device (see
// `evolution::patches::device_compute`), instantiated for the events of the
// executable. It is in its own file so that the kernels of all observable
// tensors don't add to the compilation of the executable's main file.

#include "Evolution/Executables/GrMhd/GhValenciaDivClean/EvolveGhValenciaDivClean.hpp"
#include "Evolution/Patches/DeviceComputeTagsKokkos.tpp"

#include "DataStructures/Tensor/EagerMath/Norms.hpp"

// The definitions of the functions of the compute tags, so that the kernels
// inline them. Without them (an experiment with device link-time
// optimization, `-dlto`), the kernels call the instantiations in the files of
// the functions.
#ifndef SPECTRE_DEVICE_OBSERVABLES_WITHOUT_DEFINITIONS
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.tpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.tpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.tpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.tpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.tpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.tpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.tpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.tpp"
#endif  // SPECTRE_DEVICE_OBSERVABLES_WITHOUT_DEFINITIONS

#if defined(SPECTRE_KOKKOS) and defined(USE_FD_PATCHES) and USE_FD_PATCHES
namespace {
using metavariables = EvolutionMetavarsWithFdPatches<
    USE_CONTROL_SYSTEMS, USE_PARAMETRIZED_DELEPTONIZATION, BondiSachs>;
using sources =
    typename metavariables::base::fd_patch_tag::type::device_observable_sources;
}  // namespace

template evolution::patches::device_compute::DeviceNormStatistics
evolution::patches::device_compute::device_norm_statistics<
    metavariables, sources>(const Kokkos::DefaultExecutionSpace& exec,
                            const sources& sources,
                            const std::vector<std::string>& tensor_names);
#endif
