# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ApparentHorizonFinder
  Charmxx::main
  CoordinateMaps
  DgSubcell
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FdPatch
  FiniteDifference
  GeneralRelativitySolutions
  GrMhdAnalyticData
  GrMhdSolutions
  Hydro
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  RelativisticEulerSolutions
  Observer
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )

function(add_grmhd_executable SUFFIX INTERPOLATION_TARGET USE_PARAMETRIZED_DELEPTONIZATION)
  set(EXECUTABLE "EvolveValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveValenciaDivClean.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    INTERPOLATION_TARGET=${INTERPOLATION_TARGET}
    USE_PARAMETRIZED_DELEPTONIZATION=${USE_PARAMETRIZED_DELEPTONIZATION}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_grmhd_executable(
  WithHorizon
  KerrHorizon
  false
)

add_grmhd_executable(
  ""
  ""
  false
)

# Core-collpase SN uses parametrized deleptonization
add_grmhd_executable(
  CoreCollapseSupernova
  ""
  true
)

# The FD elements on the device in FD patches
if (SPECTRE_KOKKOS)
  add_grmhd_executable(
    FdPatches
    ""
    false
  )
  target_compile_definitions(
    EvolveValenciaDivCleanFdPatches
    PRIVATE
    USE_FD_PATCHES=true
    )
endif()
