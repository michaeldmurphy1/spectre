# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FiniteDifference)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AoWeno.cpp
  DerivativeOrder.cpp
  FallbackReconstructorType.cpp
  Filter.cpp
  FluxDivergenceKokkos.cpp
  Minmod.cpp
  MonotonicityPreserving5.cpp
  MonotonisedCentral.cpp
  MonotonisedCentralKokkos.cpp
  NonUniform1D.cpp
  PartialDerivatives.cpp
  PartialDerivativesKokkos.cpp
  PositivityPreservingAdaptiveOrder.cpp
  PositivityPreservingAdaptiveOrderKokkos.cpp
  SecondPartialDerivatives.cpp
  TransfersKokkos.cpp
  Unlimited.cpp
  VolumeViewKokkos.cpp
  Wcns5z.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AoWeno.hpp
  DerivativeOrder.hpp
  DerivativeStencils.hpp
  FallbackReconstructorType.hpp
  Filter.hpp
  FiniteDifference.hpp
  FluxDivergenceKokkos.hpp
  HighOrderFluxCorrection.hpp
  Minmod.hpp
  MonotonicityPreserving5.hpp
  MonotonisedCentral.hpp
  MonotonisedCentralKokkos.hpp
  NeighborDataAsVariables.hpp
  NonUniform1D.hpp
  PartialDerivatives.hpp
  PartialDerivatives.tpp
  PartialDerivativesKokkos.hpp
  PositivityPreservingAdaptiveOrder.hpp
  PositivityPreservingAdaptiveOrderKokkos.hpp
  Reconstruct.hpp
  Reconstruct.tpp
  ReconstructKokkos.hpp
  SecondPartialDerivatives.hpp
  SecondPartialDerivatives.tpp
  TransfersKokkos.hpp
  Unlimited.hpp
  VolumeViewKokkos.hpp
  Wcns5z.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  SpectreKokkos
  Utilities
  PRIVATE
  DomainStructure
  ErrorHandling
  Spectral
  )
