# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Actions
  ApparentHorizonFinder
  ApparentHorizonFinderCriteria
  Charmxx::main
  Cce
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  EvolutionTriggers
  GeneralRelativitySolutions
  GeneralizedHarmonic
  GhGrMhdAnalyticData
  GhGrMhdSolutions
  GhRelativisticEulerSolutions
  GhValenciaDivClean
  GrMhdAnalyticData
  GrMhdSolutions
  Hydro
  Informer
  LinearOperators
  MathFunctions
  MonteCarloSolutions
  Observer
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  RelativisticEulerSolutions
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )

set(LIBS_TO_LINK_WITH_CONTROL_SYSTEM
  ControlSystem
  ControlSystemMeasurements
  )

function(add_ghmhd_executable SUFFIX USE_CONTROL_SYSTEMS USE_PARAMETRIZED_DELEPTONIZATION)
  set(EXECUTABLE "EvolveGhValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhValenciaDivClean.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    USE_CONTROL_SYSTEMS=${USE_CONTROL_SYSTEMS}
    USE_PARAMETRIZED_DELEPTONIZATION=${USE_PARAMETRIZED_DELEPTONIZATION}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
  if (USE_CONTROL_SYSTEMS STREQUAL "true")
    target_link_libraries(
      ${EXECUTABLE}
      PRIVATE
      ${LIBS_TO_LINK_WITH_CONTROL_SYSTEM}
      )
  endif()
endfunction()

add_ghmhd_executable(
  ""
  "false"
  "false"
  )

add_ghmhd_executable(
  "Bns"
   "true"
   "false"
  )
add_ghmhd_executable(
  "CoreCollapseSupernova"
   "false"
   "true"
  )

# The FD elements on the device in FD patches
if (SPECTRE_KOKKOS)
  add_ghmhd_executable(
    "FdPatches"
    "false"
    "false"
    )
  target_compile_definitions(
    EvolveGhValenciaDivCleanFdPatches
    PRIVATE
    USE_FD_PATCHES=true
    )
endif()

function(add_ghmhd_with_horizon_executable SUFFIX)
  set(EXECUTABLE "EvolveGhValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhValenciaDivCleanWithHorizon.cpp
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_ghmhd_with_horizon_executable(
  "WithHorizon"
  "false"
  )
