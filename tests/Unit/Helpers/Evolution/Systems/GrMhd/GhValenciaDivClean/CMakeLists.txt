# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "GhValenciaDivCleanHelpers")

set(LIBRARY_SOURCES
  FdPatch/ObservablesHostValues.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ConstraintDamping
  DataStructures
  GeneralizedHarmonic
  GeneralRelativity
  Hydro
  LinearOperators
  Spectral
  )
