// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::subcell {
/// Buffers that `rdmp_tci_data_on_device` keeps between calls. The host
/// buffer is pinned, so that copying to it doesn't wait for the device.
struct RdmpTciDataBuffers {
  Kokkos::View<double**> device{};
  Kokkos::View<double**, Kokkos::SharedHostPinnedSpace> host{};
};

/// @{
/*!
 * \brief The RDMP TCI data of each element of a batch whose conserved
 * variables are in device memory.
 *
 * \details The maxima and minima over the cells of each element of
 * \f$\tilde{D}\f$, \f$\tilde{Y}_e\f$, \f$\tilde{\tau}\f$ and
 * \f$|\tilde{B}|\f$, as `TciOnFdGrid` computes them.
 *
 * `enqueue_rdmp_tci_data_on_device` only enqueues the reductions and the copy
 * of their results into `buffers` on `exec`. Read the data with
 * `rdmp_tci_data_from_buffers` after fencing `exec`, e.g. together with other
 * copies to the host.
 *
 * `rdmp_tci_data_on_device` does both and fences `exec` in between. Its
 * `buffers`, if given, are reused between calls, so that they aren't allocated
 * every time.
 */
void enqueue_rdmp_tci_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const ::Tags::BatchView<ValenciaDivClean::Tags::TildeD>::type& tilde_d,
    const ::Tags::BatchView<ValenciaDivClean::Tags::TildeYe>::type& tilde_ye,
    const ::Tags::BatchView<ValenciaDivClean::Tags::TildeTau>::type& tilde_tau,
    const ::Tags::BatchView<
        ValenciaDivClean::Tags::TildeB<Frame::Inertial>>::type& tilde_b,
    gsl::not_null<RdmpTciDataBuffers*> buffers);

std::vector<evolution::dg::subcell::RdmpTciData> rdmp_tci_data_from_buffers(
    const RdmpTciDataBuffers& buffers);

std::vector<evolution::dg::subcell::RdmpTciData> rdmp_tci_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const ::Tags::BatchView<ValenciaDivClean::Tags::TildeD>::type& tilde_d,
    const ::Tags::BatchView<ValenciaDivClean::Tags::TildeYe>::type& tilde_ye,
    const ::Tags::BatchView<ValenciaDivClean::Tags::TildeTau>::type& tilde_tau,
    const ::Tags::BatchView<
        ValenciaDivClean::Tags::TildeB<Frame::Inertial>>::type& tilde_b,
    RdmpTciDataBuffers* buffers = nullptr);

/// The same for a batch that holds the conserved variables
/// \f$\tilde{D}\f$, \f$\tilde{Y}_e\f$, \f$\tilde{\tau}\f$ and
/// \f$\tilde{B}^i\f$ among others, e.g. the evolved variables of a system
template <typename TagsList>
void enqueue_rdmp_tci_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const BatchedVariables<TagsList>& variables,
    const gsl::not_null<RdmpTciDataBuffers*> buffers) {
  enqueue_rdmp_tci_data_on_device(
      exec, get<::Tags::BatchView<ValenciaDivClean::Tags::TildeD>>(variables),
      get<::Tags::BatchView<ValenciaDivClean::Tags::TildeYe>>(variables),
      get<::Tags::BatchView<ValenciaDivClean::Tags::TildeTau>>(variables),
      get<::Tags::BatchView<ValenciaDivClean::Tags::TildeB<>>>(variables),
      buffers);
}

template <typename TagsList>
std::vector<evolution::dg::subcell::RdmpTciData> rdmp_tci_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const BatchedVariables<TagsList>& variables,
    RdmpTciDataBuffers* const buffers = nullptr) {
  return rdmp_tci_data_on_device(
      exec, get<::Tags::BatchView<ValenciaDivClean::Tags::TildeD>>(variables),
      get<::Tags::BatchView<ValenciaDivClean::Tags::TildeYe>>(variables),
      get<::Tags::BatchView<ValenciaDivClean::Tags::TildeTau>>(variables),
      get<::Tags::BatchView<ValenciaDivClean::Tags::TildeB<>>>(variables),
      buffers);
}
/// @}
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
