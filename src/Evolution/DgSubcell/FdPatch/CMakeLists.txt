# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FdPatch)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActionsKokkos.hpp
  BoundaryMessagesKokkos.hpp
  ElementBox.hpp
  FdPatchArray.hpp
  FdPatches.hpp
  GeometryKokkos.hpp
  GhostZonesKokkos.hpp
  InboxTags.hpp
  PackagedDataFromDgKokkos.hpp
  PatchElementsKokkos.hpp
  SlabSize.hpp
  Tags.hpp
  TagsKokkos.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryMessagesKokkos.cpp
  FdPatches.cpp
  GeometryKokkos.cpp
  GhostZonesKokkos.cpp
  PackagedDataFromDgKokkos.cpp
  PatchElementsKokkos.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  DataStructures
  DgSubcell
  Domain
  DomainCreators
  DomainStructure
  Evolution
  FiniteDifference
  FunctionsOfTime
  Interpolation
  Options
  Serialization
  Spectral
  Time
  Utilities

  PRIVATE
  ErrorHandling
  )
