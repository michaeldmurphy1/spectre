// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/PartialDerivativesKokkos.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
// The extents, ghost widths and strides of the padded data in (x, y, z) order,
// and the stencil weights in each direction
template <size_t FdOrder, size_t Dim>
struct StencilSetup {
  using Stencil = ::fd::detail::DerivativeStencil<FdOrder, false>;

  StencilSetup(const ConstStridedVolumeView& padded_volume_vars,
               const Index<Dim>& volume_extents) {
    const size_t ghost_width =
        detail::ghost_width(padded_volume_vars, volume_extents);
    ASSERT(ghost_width >= FdOrder / 2,
           "The ghost zones of width " << ghost_width
                                       << " are too small for a derivative of "
                                          "order "
                                       << FdOrder);
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(volume_extents[d] == volume_extents[0],
             "The extents must be isotropic, but got " << volume_extents);
      extents[d] = static_cast<int>(volume_extents[d]);
      offsets[d] = static_cast<int>(ghost_width);
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
  }

  template <typename View>
  void assert_extents(const View& view, const size_t number_of_variables,
                      const size_t number_of_elements) const {
    (void)view;
    (void)number_of_variables;
    (void)number_of_elements;
#ifdef SPECTRE_DEBUG
    ASSERT(view.extent(0) == number_of_elements and
               view.extent(1) == number_of_variables,
           "The result has " << view.extent(0) << " elements and "
                             << view.extent(1) << " variables but expected "
                             << number_of_elements << " and "
                             << number_of_variables);
    for (size_t d = 0; d < 3; ++d) {
      ASSERT(view.extent(4 - d) == static_cast<size_t>(extents[d]),
             "The result has extent " << view.extent(4 - d) << " in dimension "
                                      << d << " but expected " << extents[d]);
    }
#endif  // SPECTRE_DEBUG
  }

  Kokkos::Array<int, 3> extents{1, 1, 1};
  Kokkos::Array<int, 3> offsets{0, 0, 0};
  Kokkos::Array<int, Dim> strides{};
  Kokkos::Array<std::array<double, FdOrder / 2>, Dim> weights{};
};

template <size_t FdOrder, size_t Dim>
void logical_partial_derivatives_impl(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::array<VolumeView, Dim>& logical_derivatives,
    const ConstVolumeView& padded_volume_vars,
    const Index<Dim>& volume_extents) {
  using Stencil = typename StencilSetup<FdOrder, Dim>::Stencil;
  const StencilSetup<FdOrder, Dim> setup{padded_volume_vars, volume_extents};
  Kokkos::Array<VolumeView, Dim> derivatives{};
  for (size_t d = 0; d < Dim; ++d) {
    derivatives[d] = gsl::at(logical_derivatives, d);
    setup.assert_extents(derivatives[d], padded_volume_vars.extent(1),
                         padded_volume_vars.extent(0));
  }

  const auto extents = setup.extents;
  const auto offsets = setup.offsets;
  const auto strides = setup.strides;
  const auto weights = setup.weights;
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

template <size_t FdOrder, size_t Dim>
void partial_derivatives_impl(const Kokkos::DefaultExecutionSpace& exec,
                              const StridedVolumeView& partial_derivatives,
                              const ConstStridedVolumeView& padded_volume_vars,
                              const BatchInverseJacobian<Dim>& inverse_jacobian,
                              const Index<Dim>& volume_extents) {
  using Stencil = typename StencilSetup<FdOrder, Dim>::Stencil;
  const StencilSetup<FdOrder, Dim> setup{padded_volume_vars, volume_extents};
  setup.assert_extents(partial_derivatives, Dim * padded_volume_vars.extent(1),
                       padded_volume_vars.extent(0));
  // Indexed by `Dim * logical_direction + inertial_direction`
  Kokkos::Array<Kokkos::View<const double**, Kokkos::LayoutStride>, Dim * Dim>
      jacobian{};
  for (size_t j = 0; j < Dim; ++j) {
    for (size_t i = 0; i < Dim; ++i) {
      ASSERT(
          inverse_jacobian.get(j, i).extent(0) == volume_extents.product() and
              inverse_jacobian.get(j, i).extent(1) ==
                  padded_volume_vars.extent(0),
          "The inverse Jacobian has "
              << inverse_jacobian.get(j, i).extent(0) << " grid points and "
              << inverse_jacobian.get(j, i).extent(1)
              << " elements but expected " << volume_extents.product()
              << " and " << padded_volume_vars.extent(0));
      jacobian[Dim * j + i] = inverse_jacobian.get(j, i);
    }
  }

  const auto extents = setup.extents;
  const auto offsets = setup.offsets;
  const auto strides = setup.strides;
  const auto weights = setup.weights;
  const detail::BatchCells cells{padded_volume_vars.extent(0),
                                 padded_volume_vars.extent(1), extents};
  Kokkos::parallel_for(
      "fd::device::partial_derivatives",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                          Kokkos::IndexType<int>>(exec, 0, cells.size()),
      KOKKOS_LAMBDA(const int index) {
        const auto [element, var, cell] = cells(index);
        const double* const q =
            &padded_volume_vars(element, var, cell[2] + offsets[2],
                                cell[1] + offsets[1], cell[0] + offsets[0]);
        Kokkos::Array<double, Dim> logical_derivatives{};
        for (size_t d = 0; d < Dim; ++d) {
          logical_derivatives[d] =
              Stencil::pointwise(q, strides[d], weights[d]);
        }
        const int point =
            cell[0] + extents[0] * (cell[1] + extents[1] * cell[2]);
        for (size_t i = 0; i < Dim; ++i) {
          double derivative =
              jacobian[i](point, element) * logical_derivatives[0];
          for (size_t j = 1; j < Dim; ++j) {
            derivative +=
                jacobian[Dim * j + i](point, element) * logical_derivatives[j];
          }
          partial_derivatives(element,
                              static_cast<int>(Dim) * var + static_cast<int>(i),
                              cell[2], cell[1], cell[0]) = derivative;
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

template <size_t Dim>
void partial_derivatives(const Kokkos::DefaultExecutionSpace& exec,
                         const StridedVolumeView& partial_derivatives,
                         const ConstStridedVolumeView& padded_volume_vars,
                         const BatchInverseJacobian<Dim>& inverse_jacobian,
                         const Index<Dim>& volume_extents,
                         const size_t fd_order) {
  switch (fd_order) {
    case 2:
      partial_derivatives_impl<2>(exec, partial_derivatives, padded_volume_vars,
                                  inverse_jacobian, volume_extents);
      break;
    case 4:
      partial_derivatives_impl<4>(exec, partial_derivatives, padded_volume_vars,
                                  inverse_jacobian, volume_extents);
      break;
    case 6:
      partial_derivatives_impl<6>(exec, partial_derivatives, padded_volume_vars,
                                  inverse_jacobian, volume_extents);
      break;
    case 8:
      partial_derivatives_impl<8>(exec, partial_derivatives, padded_volume_vars,
                                  inverse_jacobian, volume_extents);
      break;
    case 10:
      partial_derivatives_impl<10>(exec, partial_derivatives,
                                   padded_volume_vars, inverse_jacobian,
                                   volume_extents);
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
      const Index<DIM(data)>& volume_extents, size_t fd_order);     \
  template void partial_derivatives(                                \
      const Kokkos::DefaultExecutionSpace& exec,                    \
      const StridedVolumeView& partial_derivatives,                 \
      const ConstStridedVolumeView& padded_volume_vars,             \
      const BatchInverseJacobian<DIM(data)>& inverse_jacobian,      \
      const Index<DIM(data)>& volume_extents, size_t fd_order);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace fd::device
#endif  // SPECTRE_KOKKOS
