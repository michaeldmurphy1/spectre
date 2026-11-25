# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Patches")

set(LIBRARY_SOURCES
  Test_CombinedReduction.cpp
  Test_PatchTable.cpp
  Test_RunOnZerothElement.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainStructure
  Parallel
  Patches
  Spectral
  Utilities
  )
