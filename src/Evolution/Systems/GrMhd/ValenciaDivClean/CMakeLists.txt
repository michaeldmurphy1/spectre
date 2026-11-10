# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ValenciaDivClean)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  ComovingMagneticFieldMagnitude.cpp
  ConservativeFromPrimitive.cpp
  FixConservatives.cpp
  FixConservativesOnDeviceKokkos.cpp
  Flattener.cpp
  Fluxes.cpp
  PrimitiveFromConservative.cpp
  PrimitiveFromConservativeOnDeviceKokkos.cpp
  PrimitiveFromConservativeOptions.cpp
  SetVariablesNeededFixingToFalse.cpp
  Sources.cpp
  SpectralFilter.cpp
  TimeDerivativeTerms.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AllSolutions.hpp
  Characteristics.hpp
  ComovingMagneticFieldMagnitude.hpp
  ComputeFluxesFromPrimitives.hpp
  ConservativeFromPrimitive.hpp
  ConservativeFromPrimitive.tpp
  FixConservatives.hpp
  FixConservativesOnDeviceKokkos.hpp
  Flattener.hpp
  Fluxes.hpp
  Fluxes.tpp
  KastaunEtAl.hpp
  KastaunEtAl.tpp
  KastaunEtAlHydro.hpp
  KastaunEtAlHydro.tpp
  NewmanHamlin.hpp
  NewmanHamlin.tpp
  PalenzuelaEtAl.hpp
  PalenzuelaEtAl.tpp
  PrimitiveFromConservative.hpp
  PrimitiveFromConservative.tpp
  PrimitiveFromConservativeOnDeviceKokkos.hpp
  PrimitiveFromConservativeOptions.hpp
  PrimitiveRecoveryData.hpp
  QuadrupoleFormula.hpp
  SetVariablesNeededFixingToFalse.hpp
  Sources.hpp
  Sources.tpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  DgSubcell
  DiscontinuousGalerkin
  ErrorHandling
  Events
  Evolution
  FdPatch
  FiniteDifference
  GeneralRelativity
  GrMhdAnalyticData
  GrMhdSolutions
  H5
  Hydro
  Options
  RelativisticEulerSolutions
  Utilities
  VariableFixing
  PRIVATE
  LinearAlgebra
  RootFinding
  Simd
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Instantiations)
add_subdirectory(Subcell)
