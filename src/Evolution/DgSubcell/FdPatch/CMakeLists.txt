# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FdPatch)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FdPatches.hpp
  Tags.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FdPatches.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DgSubcell
  Domain
  DomainCreators
  DomainStructure
  Evolution
  Options
  Serialization
  Spectral
  Utilities

  PRIVATE
  ErrorHandling
  )
