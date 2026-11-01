// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/PartialDerivativesKokkos.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeStencils.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device {
namespace {
template <size_t FdOrder, size_t Dim>
void logical_partial_derivatives_impl(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<VolumeView, Dim>& logical_derivatives,
    const ConstVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents) {
  using Stencil = ::fd::detail::DerivativeStencil<FdOrder, false>;
  const size_t ghost_width =
      detail::ghost_width(padded_volume_vars, volume_extents);
  ASSERT(ghost_width >= FdOrder / 2,
         "The ghost zones of width " << ghost_width
                                     << " are too small for a derivative of "
                                        "order "
                                     << FdOrder);

  // Extents and ghost widths in (x, y, z) order
  Kokkos::Array<int, 3> extents{1, 1, 1};
  Kokkos::Array<int, 3> offsets{0, 0, 0};
  Kokkos::Array<VolumeView, Dim> derivatives{};
  Kokkos::Array<int, Dim> strides{};
  Kokkos::Array<std::array<double, FdOrder / 2>, Dim> weights{};
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(volume_extents[d] == volume_extents[0],
           "The extents must be isotropic, but got " << volume_extents);
    extents[d] = static_cast<int>(volume_extents[d]);
    offsets[d] = static_cast<int>(ghost_width);
    derivatives[d] = gsl::at(logical_derivatives, d);
    strides[d] = static_cast<int>(padded_volume_vars.stride(4 - d));
    // Compute the grid spacing like `fd::logical_partial_derivatives` so the
    // results are identical
    const auto& logical_coords =
        Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::CellCentered>(
            volume_extents[d]);
    weights[d] = Stencil::derivative_weights(
        1.0 / (logical_coords[1] - logical_coords[0]));
  }
#ifdef SPECTRE_DEBUG
  for (size_t d = 0; d < 3; ++d) {
    for (size_t i = 0; i < Dim; ++i) {
      ASSERT(derivatives[i].extent(4 - d) == static_cast<size_t>(extents[d]),
             "The derivative in direction "
                 << i << " has extent " << derivatives[i].extent(4 - d)
                 << " in dimension " << d << " but expected " << extents[d]);
    }
  }
#endif  // SPECTRE_DEBUG

  const detail::BatchCells cells{padded_volume_vars.extent(0),
                                 padded_volume_vars.extent(1), extents};
  Kokkos::parallel_for(
      "fd::device::logical_partial_derivatives",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                          Kokkos::IndexType<int>>(exec, 0, cells.size()),
      KOKKOS_LAMBDA(const int index) {
        const auto [element, var, cell] = cells(index);
        const double* const q =
            &padded_volume_vars(element, var, cell[2] + offsets[2],
                                cell[1] + offsets[1], cell[0] + offsets[0]);
        for (size_t d = 0; d < Dim; ++d) {
          derivatives[d](element, var, cell[2], cell[1], cell[0]) =
              Stencil::pointwise(q, strides[d], weights[d]);
        }
      });
}
}  // namespace

template <size_t Dim>
void logical_partial_derivatives(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<VolumeView, Dim>& logical_derivatives,
    const ConstVolumeView& padded_volume_vars, const Index<Dim>& volume_extents,
    const size_t fd_order) {
  switch (fd_order) {
    case 2:
      logical_partial_derivatives_impl<2>(exec, logical_derivatives,
                                          padded_volume_vars, volume_extents);
      break;
    case 4:
      logical_partial_derivatives_impl<4>(exec, logical_derivatives,
                                          padded_volume_vars, volume_extents);
      break;
    case 6:
      logical_partial_derivatives_impl<6>(exec, logical_derivatives,
                                          padded_volume_vars, volume_extents);
      break;
    case 8:
      logical_partial_derivatives_impl<8>(exec, logical_derivatives,
                                          padded_volume_vars, volume_extents);
      break;
    case 10:
      logical_partial_derivatives_impl<10>(exec, logical_derivatives,
                                           padded_volume_vars, volume_extents);
      break;
    default:
      ERROR("Cannot do finite difference derivative of order " << fd_order);
  };
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                      \
  template void logical_partial_derivatives(                        \
      const Kokkos::DefaultExecutionSpace& exec,                    \
      const std::array<VolumeView, DIM(data)>& logical_derivatives, \
      const ConstVolumeView& padded_volume_vars,                    \
      const Index<DIM(data)>& volume_extents, size_t fd_order);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
