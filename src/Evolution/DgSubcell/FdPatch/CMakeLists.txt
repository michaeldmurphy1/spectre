# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FdPatch)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GhostZonesKokkos.hpp
  PatchTable.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GhostZonesKokkos.cpp
  PatchTable.cpp
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
  Patches
  Serialization
  Spectral
  Time
  Utilities

  PRIVATE
  ErrorHandling
  )
