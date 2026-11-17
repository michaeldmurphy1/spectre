// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintGammasKokkos.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianSum.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace gh {
void constraint_gammas_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const typename ::Tags::BatchView<Tags::ConstraintGamma0>::type& gamma0,
    const typename ::Tags::BatchView<Tags::ConstraintGamma1>::type& gamma1,
    const typename ::Tags::BatchView<Tags::ConstraintGamma2>::type& gamma2,
    const typename ::Tags::BatchView<
        domain::Tags::Coordinates<3, Frame::Grid>>::type& grid_coordinates,
    const Kokkos::View<const ConstraintDamping::GaussianSum<3>*>&
        damping_functions) {
  ASSERT(damping_functions.extent(0) == 3,
         "Expected the damping functions of the three gammas, but got "
             << damping_functions.extent(0));
  // The components as arrays of views, since tensors of views aren't
  // device-callable
  using View = Kokkos::View<double**, Kokkos::LayoutStride>;
  const Kokkos::Array<View, 3> gammas{get(gamma0), get(gamma1), get(gamma2)};
  const Kokkos::Array<View, 3> coordinates{get<0>(grid_coordinates),
                                           get<1>(grid_coordinates),
                                           get<2>(grid_coordinates)};
  const size_t number_of_points = coordinates[0].extent(0);
  const size_t number_of_elements = coordinates[0].extent(1);
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < 3; ++i) {
    ASSERT(gammas[i].extent(0) == number_of_points and
               gammas[i].extent(1) == number_of_elements and
               coordinates[i].extent(0) == number_of_points and
               coordinates[i].extent(1) == number_of_elements,
           "All batches must have " << number_of_points << " grid points and "
                                    << number_of_elements << " elements");
  }
#endif  // SPECTRE_DEBUG
  Kokkos::parallel_for(
      "gh::constraint_gammas_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>,
                            Kokkos::IndexType<size_t>>(
          exec, {0, 0}, {number_of_points, number_of_elements}),
      KOKKOS_LAMBDA(const size_t point, const size_t element) {
        const std::array<double, 3> x{coordinates[0](point, element),
                                      coordinates[1](point, element),
                                      coordinates[2](point, element)};
        for (size_t i = 0; i < 3; ++i) {
          gammas[i](point, element) = damping_functions(i)(x);
        }
      });
}
}  // namespace gh
#endif  // SPECTRE_KOKKOS
