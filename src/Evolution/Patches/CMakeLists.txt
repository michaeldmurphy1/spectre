# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Patches)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActionsKokkos.hpp
  GridToInertial.hpp
  HostCopyKokkos.hpp
  InboxTags.hpp
  PatchArray.hpp
  PatchElementsKokkos.hpp
  PatchTable.hpp
  SavedBatchesKokkos.hpp
  SendToPatch.hpp
  SlabSize.hpp
  Tags.hpp
  TagsKokkos.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
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
  Options
  Parallel
  Serialization
  Spectral
  Time
  Utilities

  PRIVATE
  ErrorHandling
  )
