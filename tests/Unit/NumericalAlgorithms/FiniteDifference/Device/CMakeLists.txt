# Distributed under the MIT License.
# See LICENSE.txt for details.

if (NOT TARGET Kokkos::kokkos)
  return()
endif()

set(LIBRARY "Test_FiniteDifferenceDevice")

set(LIBRARY_SOURCES
  Test_MonotonisedCentral.cpp
  Test_VolumeView.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainStructure
  FiniteDifference
  FiniteDifferenceDevice
  SpectreKokkos
  Utilities
  )
