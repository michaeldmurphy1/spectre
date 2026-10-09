# Distributed under the MIT License.
# See LICENSE.txt for details.

if (NOT TARGET Kokkos::kokkos)
  return()
endif()

set(LIBRARY FiniteDifferenceDevice)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  MonotonisedCentral.cpp
  PartialDerivatives.cpp
  VolumeView.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  MonotonisedCentral.hpp
  PartialDerivatives.hpp
  Reconstruct.hpp
  VolumeView.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  FiniteDifference
  SpectreKokkos
  Utilities
  PRIVATE
  DomainStructure
  ErrorHandling
  Spectral
  )
