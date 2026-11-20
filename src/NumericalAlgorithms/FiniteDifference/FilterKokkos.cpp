// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/FilterKokkos.hpp"

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/FiniteDifference/FilterStencils.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace fd::device {
namespace {
template <size_t FdOrder, size_t Dim>
void kreiss_oliger_filter_impl(const Kokkos::DefaultExecutionSpace& exec,
                               const StridedVolumeView& filtered_vars,
                               const ConstStridedVolumeView& padded_volume_vars,
                               const Index<Dim>& volume_extents,
                               const double epsilon) {
  using Stencil = ::fd::detail::KoDissipationStencil<FdOrder, false>;
  const size_t ghost_width =
      detail::ghost_width(padded_volume_vars, volume_extents);
  ASSERT(ghost_width >= FdOrder / 2,
         "The ghost zones of width " << ghost_width
                                     << " are too small for a filter of order "
                                     << FdOrder);
  ASSERT(filtered_vars.extent(0) == padded_volume_vars.extent(0) and
             filtered_vars.extent(1) == padded_volume_vars.extent(1),
         "The filtered variables hold "
             << filtered_vars.extent(1) << " variables of "
             << filtered_vars.extent(0)
             << " elements, but the padded variables hold "
             << padded_volume_vars.extent(1) << " variables of "
             << padded_volume_vars.extent(0) << " elements");

  // Extents and ghost widths in (x, y, z) order
  Kokkos::Array<int, 3> extents{1, 1, 1};
  Kokkos::Array<int, 3> offsets{0, 0, 0};
  Kokkos::Array<int, Dim> strides{};
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(volume_extents[d] == volume_extents[0],
           "The extents must be isotropic, but got " << volume_extents);
    ASSERT(filtered_vars.extent(4 - d) == volume_extents[d],
           "The filtered variables have extent "
               << filtered_vars.extent(4 - d) << " in dimension " << d
               << " but expected " << volume_extents[d]);
    extents[d] = static_cast<int>(volume_extents[d]);
    offsets[d] = static_cast<int>(ghost_width);
    strides[d] = static_cast<int>(padded_volume_vars.stride(4 - d));
  }

  // One thread per cell, looping over the variables
  const size_t number_of_variables = padded_volume_vars.extent(1);
  const size_t variable_stride = padded_volume_vars.stride(1);
  const size_t filtered_stride = filtered_vars.stride(1);
  const detail::BatchPoints points{padded_volume_vars.extent(0), extents};
  Kokkos::parallel_for(
      "fd::device::kreiss_oliger_filter",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                          Kokkos::IndexType<int>>(exec, 0, points.size()),
      KOKKOS_LAMBDA(const int index) {
        const auto [element, cell] = points(index);
        const double* q =
            &padded_volume_vars(element, 0, cell[2] + offsets[2],
                                cell[1] + offsets[1], cell[0] + offsets[0]);
        double* filtered_var =
            &filtered_vars(element, 0, cell[2], cell[1], cell[0]);
        for (size_t var = 0; var < number_of_variables; ++var) {
          // As `fd::kreiss_oliger_filter`: the dissipation in each
          // direction, then the value of the cell
          double filtered = Stencil::pointwise(q, strides[0], epsilon);
          for (size_t d = 1; d < Dim; ++d) {
            filtered += Stencil::pointwise(q, strides[d], epsilon);
          }
          *filtered_var = filtered + *q;
          q += variable_stride;
          filtered_var += filtered_stride;
        }
      });
}
}  // namespace

template <size_t Dim>
void kreiss_oliger_filter(const Kokkos::DefaultExecutionSpace& exec,
                          const StridedVolumeView& filtered_vars,
                          const ConstStridedVolumeView& padded_volume_vars,
                          const Index<Dim>& volume_extents,
                          const size_t fd_order, const double epsilon) {
  switch (fd_order) {
    case 2:
      kreiss_oliger_filter_impl<2>(exec, filtered_vars, padded_volume_vars,
                                   volume_extents, epsilon);
      break;
    case 4:
      kreiss_oliger_filter_impl<4>(exec, filtered_vars, padded_volume_vars,
                                   volume_extents, epsilon);
      break;
    case 6:
      kreiss_oliger_filter_impl<6>(exec, filtered_vars, padded_volume_vars,
                                   volume_extents, epsilon);
      break;
    case 8:
      kreiss_oliger_filter_impl<8>(exec, filtered_vars, padded_volume_vars,
                                   volume_extents, epsilon);
      break;
    case 10:
      kreiss_oliger_filter_impl<10>(exec, filtered_vars, padded_volume_vars,
                                    volume_extents, epsilon);
      break;
    default:
      ERROR("Cannot apply a Kreiss-Oliger filter of order " << fd_order);
  };
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                 \
  template void kreiss_oliger_filter(                          \
      const Kokkos::DefaultExecutionSpace& exec,               \
      const StridedVolumeView& filtered_vars,                  \
      const ConstStridedVolumeView& padded_volume_vars,        \
      const Index<DIM(data)>& volume_extents, size_t fd_order, \
      double epsilon);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
