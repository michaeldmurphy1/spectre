// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/FluxDivergenceKokkos.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device {
template <size_t Dim>
void add_cartesian_flux_divergence(
    const Kokkos::DefaultExecutionSpace& exec, const VolumeView& dt_vars,
    const std::array<ConstVolumeView, Dim>& boundary_corrections,
    const ConstVolumeView& det_inverse_jacobian,
    const std::array<double, Dim>& one_over_delta,
    const Index<Dim>& volume_extents) {
  ASSERT(detail::ghost_width(dt_vars, volume_extents) == 0,
         "The time derivatives must not be padded");
  ASSERT(detail::ghost_width(det_inverse_jacobian, volume_extents) == 0 and
             det_inverse_jacobian.extent(0) == dt_vars.extent(0) and
             det_inverse_jacobian.extent(1) == 1,
         "The determinant of the inverse Jacobian must have the extents of "
         "the volume data and one variable");
  Kokkos::Array<int, 3> extents{1, 1, 1};
  Kokkos::Array<ConstVolumeView, Dim> corrections{};
  Kokkos::Array<double, Dim> inverse_spacings{};
  for (size_t d = 0; d < Dim; ++d) {
    extents[d] = static_cast<int>(volume_extents[d]);
    corrections[d] = gsl::at(boundary_corrections, d);
    inverse_spacings[d] = gsl::at(one_over_delta, d);
#ifdef SPECTRE_DEBUG
    for (size_t i = 0; i < 3; ++i) {
      const size_t expected_extent =
          i < Dim ? volume_extents[i] + (i == d ? 1 : 0) : 1;
      ASSERT(corrections[d].extent(0) == dt_vars.extent(0) and
                 corrections[d].extent(1) == dt_vars.extent(1) and
                 corrections[d].extent(4 - i) == expected_extent,
             "The boundary corrections in direction "
                 << d << " have extent " << corrections[d].extent(4 - i)
                 << " in dimension " << i << " but expected "
                 << expected_extent);
    }
#endif  // SPECTRE_DEBUG
  }

  const detail::BatchCells cells{dt_vars.extent(0), dt_vars.extent(1), extents};
  Kokkos::parallel_for(
      "fd::device::add_cartesian_flux_divergence",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                          Kokkos::IndexType<int>>(exec, 0, cells.size()),
      KOKKOS_LAMBDA(const int index) {
        const auto [element, var, cell] = cells(index);
        const double det_inv_jacobian =
            det_inverse_jacobian(element, 0, cell[2], cell[1], cell[0]);
        double& dt_var = dt_vars(element, var, cell[2], cell[1], cell[0]);
        for (size_t d = 0; d < Dim; ++d) {
          Kokkos::Array<int, 3> upper_face = cell;
          ++upper_face[d];
          dt_var += inverse_spacings[d] * det_inv_jacobian *
                    (corrections[d](element, var, upper_face[2], upper_face[1],
                                    upper_face[0]) -
                     corrections[d](element, var, cell[2], cell[1], cell[0]));
        }
      });
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template void add_cartesian_flux_divergence(                              \
      const Kokkos::DefaultExecutionSpace& exec, const VolumeView& dt_vars, \
      const std::array<ConstVolumeView, DIM(data)>& boundary_corrections,   \
      const ConstVolumeView& det_inverse_jacobian,                          \
      const std::array<double, DIM(data)>& one_over_delta,                  \
      const Index<DIM(data)>& volume_extents);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
