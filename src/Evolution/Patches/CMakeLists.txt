# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Patches)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActionsKokkos.hpp
  CombinedReduction.hpp
  DeviceComputeTagsKokkos.hpp
  DeviceComputeTagsKokkos.tpp
  DeviceObservationsKokkos.hpp
  ElementBoxesKokkos.hpp
  EventActionsKokkos.hpp
  EventsOnElements.hpp
  GridToInertial.hpp
  HostCopyKokkos.hpp
  InboxTags.hpp
  PatchArray.hpp
  PatchElementsKokkos.hpp
  PatchState.hpp
  PatchTable.hpp
  RunOnZerothElement.hpp
  SavedBatchesKokkos.hpp
  SelfStartKokkos.hpp
  SendToPatch.hpp
  SlabSize.hpp
  Tags.hpp
  TagsKokkos.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DeviceComputeTagsKokkos.cpp
  DeviceObservationsKokkos.cpp
  GridToInertial.cpp
  PatchElementsKokkos.cpp
  PatchTable.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  DataStructures
  Domain
  DomainStructure
  Events
  EventsAndTriggers
  Observer
  Options
  Parallel
  Serialization
  Spectral
  Time
  Utilities

  PRIVATE
  ErrorHandling
  )
