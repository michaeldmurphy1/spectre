// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/RdmpTciDataKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Kokkos/ScratchViewKokkos.hpp"

namespace grmhd::ValenciaDivClean::subcell {
std::vector<evolution::dg::subcell::RdmpTciData> rdmp_tci_data_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const ::Tags::BatchView<Tags::TildeD>::type& tilde_d,
    const ::Tags::BatchView<Tags::TildeYe>::type& tilde_ye,
    const ::Tags::BatchView<Tags::TildeTau>::type& tilde_tau,
    const ::Tags::BatchView<Tags::TildeB<Frame::Inertial>>::type& tilde_b,
    RdmpTciDataBuffers* const buffers) {
  constexpr size_t number_of_rdmp_variables = 4;
  const auto& d = get(tilde_d);
  const auto& ye = get(tilde_ye);
  const auto& tau = get(tilde_tau);
  const auto& b_x = get<0>(tilde_b);
  const auto& b_y = get<1>(tilde_b);
  const auto& b_z = get<2>(tilde_b);
  const auto number_of_points = static_cast<int>(d.extent(0));
  const size_t number_of_elements = d.extent(1);
  // The maxima and then the minima of each element
  const Kokkos::View<double**> result = scratch_view(
      exec, buffers == nullptr ? nullptr : &buffers->device, "RDMP data",
      number_of_elements, 2 * number_of_rdmp_variables);

  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  Kokkos::parallel_for(
      "grmhd::ValenciaDivClean::subcell::rdmp_tci_data_on_device",
      TeamPolicy(exec, static_cast<int>(number_of_elements), Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const int e = team.league_rank();
        const auto min_max = [&team, number_of_points, e, &result](
                                 const size_t variable,
                                 const auto& value_at_point) {
          Kokkos::MinMaxScalar<double> extrema{};
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, number_of_points),
              [&value_at_point](const int i,
                                Kokkos::MinMaxScalar<double>& update) {
                const double value = value_at_point(i);
                update.min_val = Kokkos::min(update.min_val, value);
                update.max_val = Kokkos::max(update.max_val, value);
              },
              Kokkos::MinMax<double>(extrema));
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            result(e, variable) = extrema.max_val;
            result(e, number_of_rdmp_variables + variable) = extrema.min_val;
          });
        };
        min_max(0, [&d, e](const int i) { return d(i, e); });
        min_max(1, [&ye, e](const int i) { return ye(i, e); });
        min_max(2, [&tau, e](const int i) { return tau(i, e); });
        min_max(3, [&b_x, &b_y, &b_z, e](const int i) {
          return sqrt(b_x(i, e) * b_x(i, e) + b_y(i, e) * b_y(i, e) +
                      b_z(i, e) * b_z(i, e));
        });
      });

  if (buffers != nullptr and (not buffers->host.is_allocated() or
                              buffers->host.extent(0) != number_of_elements)) {
    buffers->host =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, result);
  }
  const auto host_result =
      buffers != nullptr
          ? buffers->host
          : Kokkos::create_mirror_view(Kokkos::WithoutInitializing, result);
  Kokkos::deep_copy(exec, host_result, result);
  exec.fence("grmhd::ValenciaDivClean::subcell::rdmp_tci_data_on_device");
  std::vector<evolution::dg::subcell::RdmpTciData> rdmp_tci_data(
      number_of_elements);
  for (size_t e = 0; e < number_of_elements; ++e) {
    auto& element_data = rdmp_tci_data[e];
    element_data.max_variables_values = DataVector{number_of_rdmp_variables};
    element_data.min_variables_values = DataVector{number_of_rdmp_variables};
    for (size_t variable = 0; variable < number_of_rdmp_variables; ++variable) {
      element_data.max_variables_values[variable] = host_result(e, variable);
      element_data.min_variables_values[variable] =
          host_result(e, number_of_rdmp_variables + variable);
    }
  }
  return rdmp_tci_data;
}
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
