# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Patches)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GridToInertial.hpp
  InboxTags.hpp
  PatchTable.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GridToInertial.cpp
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
