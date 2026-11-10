// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::subcell {
/// Buffers that `rdmp_tci_data_on_device` keeps between calls
struct RdmpTciDataBuffers {
  Kokkos::View<double**> device{};
  Kokkos::View<double**>::host_mirror_type host{};
};

/*!
 * \brief The RDMP TCI data of each element of a batch whose conserved
 * variables are in device memory.
 *
 * \details The maxima and minima over the cells of each element of
 * \f$\tilde{D}\f$, \f$\tilde{Y}_e\f$, \f$\tilde{\tau}\f$ and
 * \f$|\tilde{B}|\f$, as `TciOnFdGrid` computes them. The reductions are
 * enqueued on `exec`, which is fenced to copy the result to the host.
 * `buffers`, if given, are reused between calls, so that they aren't allocated
 * every time.
 */
std::vector<evolution::dg::subcell::RdmpTciData> rdmp_tci_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const ::Tags::BatchView<Tags::TildeD>::type& tilde_d,
    const ::Tags::BatchView<Tags::TildeYe>::type& tilde_ye,
    const ::Tags::BatchView<Tags::TildeTau>::type& tilde_tau,
    const ::Tags::BatchView<Tags::TildeB<Frame::Inertial>>::type& tilde_b,
    RdmpTciDataBuffers* buffers = nullptr);
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
