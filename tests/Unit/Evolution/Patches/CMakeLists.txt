# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Patches")

set(LIBRARY_SOURCES
  Test_PatchTable.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainStructure
  Patches
  Spectral
  Utilities
  )
