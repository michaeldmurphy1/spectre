// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/Executables/GrMhd/GhValenciaDivClean/GhValenciaDivCleanBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Utilities/TMPL.hpp"

template <bool UseControlSystems, bool UseParametrizedDeleptonization,
          typename... InterpolationTargetTags>
struct EvolutionMetavars
    : public GhValenciaDivCleanTemplateBase<
          EvolutionMetavars<UseControlSystems, UseParametrizedDeleptonization,
                            InterpolationTargetTags...>,
          true, UseControlSystems, UseParametrizedDeleptonization> {
  using base = GhValenciaDivCleanTemplateBase<
      EvolutionMetavars<UseControlSystems, UseParametrizedDeleptonization,
                        InterpolationTargetTags...>,
      true, UseControlSystems, UseParametrizedDeleptonization>;
  using const_global_cache_tags = typename base::const_global_cache_tags;
  using observed_reduction_data_tags =
      typename base::observed_reduction_data_tags;
  using component_list = typename base::component_list;
  using factory_creation = typename base::factory_creation;
  using registration = typename base::registration;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning, coupled to a dynamic spacetime evolved with the Generalized "
      "Harmonic formulation\n"};
};

#ifdef SPECTRE_KOKKOS
/// The metavariables of `EvolutionMetavars` with the FD elements evolved on the
/// device in FD patches (`evolution::patches::PatchArray`, see
/// `grmhd::GhValenciaDivClean::fd_patch::Patch` for what is supported). The DG
/// elements are evolved on the host as in `EvolutionMetavars`.
template <bool UseControlSystems, bool UseParametrizedDeleptonization,
          typename... InterpolationTargetTags>
struct EvolutionMetavarsWithFdPatches
    : public GhValenciaDivCleanTemplateBase<
          EvolutionMetavarsWithFdPatches<UseControlSystems,
                                         UseParametrizedDeleptonization,
                                         InterpolationTargetTags...>,
          true, UseControlSystems, UseParametrizedDeleptonization> {
  static_assert(not UseControlSystems,
                "FD patches don't check the functions of time of control "
                "systems yet.");
  static_assert(not UseParametrizedDeleptonization,
                "FD patches don't apply the parametrized deleptonization.");
  using base = GhValenciaDivCleanTemplateBase<
      EvolutionMetavarsWithFdPatches<UseControlSystems,
                                     UseParametrizedDeleptonization,
                                     InterpolationTargetTags...>,
      true, UseControlSystems, UseParametrizedDeleptonization>;
  using const_global_cache_tags = typename base::const_global_cache_tags;
  using observed_reduction_data_tags =
      typename base::observed_reduction_data_tags;
  using factory_creation = typename base::factory_creation;
  using registration = typename base::registration;

  static constexpr bool use_patches = true;
  using patch_component = typename base::fd_patch_component;
  using patch_table_tag = evolution::dg::subcell::fd_patch::Tags::PatchTable<3>;
  // The DG elements and the patches agree on the slab size
  using slab_size_reduction = evolution::patches::SlabSizeReduction<
      patch_component, typename base::dg_element_array_component>;
  using component_list =
      tmpl::push_back<typename base::component_list, patch_component>;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning, coupled to a dynamic spacetime evolved with the Generalized "
      "Harmonic formulation, with the FD elements on the device\n"};
};
#endif  // SPECTRE_KOKKOS
