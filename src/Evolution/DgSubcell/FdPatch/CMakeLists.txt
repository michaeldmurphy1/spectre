# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FdPatch)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FdPatches.hpp
  GeometryKokkos.hpp
  GhostZonesKokkos.hpp
  Tags.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FdPatches.cpp
  GeometryKokkos.cpp
  GhostZonesKokkos.cpp
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
