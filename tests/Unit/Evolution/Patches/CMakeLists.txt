# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Patches")

set(LIBRARY_SOURCES
  Test_CombinedReduction.cpp
  Test_EventsOnElements.cpp
  Test_PatchTable.cpp
  Test_RunOnZerothElement.cpp
  Test_SendToPatch.cpp
  )

if (SPECTRE_KOKKOS)
  list(APPEND LIBRARY_SOURCES
    Test_ActionsKokkos.cpp
    Test_DeviceObservationsKokkos.cpp
    Test_HostCopyKokkos.cpp
    Test_SavedBatchesKokkos.cpp
    )
endif()

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DataStructuresHelpers
  Evolution
  Domain
  DomainStructure
  Events
  EventsAndTriggers
  Observer
  Parallel
  Patches
  Spectral
  Time
  Utilities
  )
