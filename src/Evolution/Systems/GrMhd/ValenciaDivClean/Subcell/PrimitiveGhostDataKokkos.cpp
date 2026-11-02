// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostDataKokkos.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::subcell {
void PrimitiveGhostVariablesOnDevice::apply_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<double*>& ghost_variables,
    const Variables<hydro::grmhd_tags<DataVector>>& prims) {
  const size_t number_of_points = prims.number_of_grid_points();
  ASSERT(
      ghost_variables.extent(0) >=
          number_of_points *
              Variables<
                  ghost_variables_tag_list>::number_of_independent_components,
      "The ghost variables buffer has " << ghost_variables.extent(0)
                                        << " values, which is too small");
  using HostView =
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  const auto copy_component = [&exec, &ghost_variables, number_of_points](
                                  const size_t component,
                                  const DataVector& host_component) {
    const size_t first = component * number_of_points;
    Kokkos::deep_copy(
        exec,
        Kokkos::subview(ghost_variables,
                        Kokkos::make_pair(first, first + number_of_points)),
        HostView(host_component.data(), number_of_points));
  };

  // Copy each tensor, using the spatial velocity for W v^i
  size_t component = 0;
  size_t first_lorentz_factor_times_v = 0;
  tmpl::for_each<ghost_variables_tag_list>([&](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    if constexpr (std::is_same_v<tag,
                                 hydro::Tags::LorentzFactorTimesSpatialVelocity<
                                     DataVector, 3>>) {
      first_lorentz_factor_times_v = component;
      for (const auto& velocity_component :
           get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prims)) {
        copy_component(component, velocity_component);
        ++component;
      }
    } else {
      for (const auto& tensor_component : get<tag>(prims)) {
        copy_component(component, tensor_component);
        ++component;
      }
    }
  });

  // W v^i, computed like `PrimitiveGhostVariables::apply`
  const Kokkos::View<double*> lorentz_factor(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "Lorentz factor"),
      number_of_points);
  Kokkos::deep_copy(
      exec, lorentz_factor,
      HostView(get(get<hydro::Tags::LorentzFactor<DataVector>>(prims)).data(),
               number_of_points));
  const size_t first_value = first_lorentz_factor_times_v * number_of_points;
  Kokkos::parallel_for(
      "PrimitiveGhostVariablesOnDevice",
      Kokkos::RangePolicy<>(exec, 0, 3 * number_of_points),
      KOKKOS_LAMBDA(const size_t index) {
        ghost_variables(first_value + index) *=
            lorentz_factor(index % number_of_points);
      });
  // The temporary Lorentz factor and the host data must stay valid until the
  // kernel and copies are done
  exec.fence();
}
}  // namespace grmhd::ValenciaDivClean::subcell
#endif  // SPECTRE_KOKKOS
