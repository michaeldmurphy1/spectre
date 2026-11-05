// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <cstddef>
#include <numeric>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

namespace evolution::dg::subcell::fd_patch {
template <size_t Dim>
GhostZones<Dim>::GhostZones(const Kokkos::DefaultExecutionSpace& exec,
                            const std::vector<Element<Dim>>& elements,
                            const FdPatches<Dim>& fd_patches,
                            const size_t patch, const size_t ghost_width,
                            const size_t number_of_variables)
    : number_of_elements_(elements.size()),
      subcell_extents_(fd::mesh(fd_patches.dg_mesh(patch)).extents()),
      ghost_width_(ghost_width),
      number_of_variables_(number_of_variables) {
  const auto& element_ids = fd_patches.element_ids(patch);
  if (elements.size() != element_ids.size()) {
    ERROR("Got " << elements.size() << " elements for patch " << patch
                 << ", which has " << element_ids.size() << " elements");
  }
  const size_t number_of_cells = subcell_extents_.product();
  size_t number_of_padded_cells = 1;
  for (size_t d = 0; d < Dim; ++d) {
    if (subcell_extents_[d] < ghost_width) {
      ERROR("The ghost zones of width "
            << ghost_width << " are wider than the " << subcell_extents_[d]
            << " cells of the elements of patch " << patch << " in dimension "
            << d);
    }
    number_of_padded_cells *= subcell_extents_[d] + 2 * ghost_width;
  }
  const size_t element_stride = number_of_variables * number_of_cells;
  const size_t padded_element_stride =
      number_of_variables * number_of_padded_cells;

  std::vector<size_t> unpadded_points(number_of_cells);
  std::iota(unpadded_points.begin(), unpadded_points.end(), 0_st);
  const auto padded_points =
      ::fd::device::padded_interior_points(subcell_extents_, ghost_width);

  for (size_t slot = 0; slot < elements.size(); ++slot) {
    const Element<Dim>& element = elements[slot];
    if (element.id() != element_ids[slot]) {
      ERROR("Element " << element.id() << " is in slot " << slot
                       << " but the slot of patch " << patch << " is element "
                       << element_ids[slot]);
    }
    if (not element.external_boundaries().empty()) {
      ERROR("FD patches don't support external boundaries yet, but element "
            << element.id() << " has external boundaries in the directions "
            << element.external_boundaries()
            << ". Only periodic boundaries are supported.");
    }
    from_patch_.add_copy(slot * element_stride, number_of_cells,
                         slot * padded_element_stride, number_of_padded_cells,
                         unpadded_points, padded_points);

    for (const auto& direction : Direction<Dim>::all_directions()) {
      const auto& neighbors = element.neighbors().at(direction);
      if (neighbors.size() != 1) {
        ERROR("FD patches need conforming elements, but element "
              << element.id() << " has " << neighbors.size()
              << " neighbors in direction " << direction);
      }
      const ElementId<Dim>& neighbor_id = *neighbors.ids().begin();
      const auto location = fd_patches.location(neighbor_id);
      if (not location.has_value() or location->patch != patch) {
        external_faces_.push_back(ExternalFace{slot, direction, neighbor_id});
        continue;
      }
      // Within a block, so the neighbor is aligned and has the same extents
      ASSERT(neighbors.orientation(neighbor_id).is_aligned(),
             "The neighbor " << neighbor_id << " of element " << element.id()
                             << " in the same patch isn't aligned");
      const size_t dimension = direction.dimension();
      const size_t extent = subcell_extents_[dimension];
      const bool upper = direction.side() == Side::Upper;
      // The cells of the neighbor next to the shared face
      const auto source_points =
          upper ? ::fd::device::slab_points(subcell_extents_, dimension, 0,
                                            ghost_width)
                : ::fd::device::slab_points(subcell_extents_, dimension,
                                            extent - ghost_width, extent);
      from_patch_.add_copy(
          location->slot * element_stride, number_of_cells,
          slot * padded_element_stride, number_of_padded_cells, source_points,
          ::fd::device::padded_ghost_zone_points(subcell_extents_, ghost_width,
                                                 dimension, upper));
    }
  }
  from_patch_.copy_to_device(exec);
}

template <size_t Dim>
void GhostZones<Dim>::fill_from_patch(
    const Kokkos::DefaultExecutionSpace& exec,
    const ::fd::device::VolumeView& padded_vars,
    const Kokkos::View<const double*>& ghost_variables) const {
  ASSERT(padded_vars.extent(0) == number_of_elements_ and
             padded_vars.extent(1) == number_of_variables_ and
             ::fd::device::detail::ghost_width(padded_vars, subcell_extents_) ==
                 ghost_width_,
         "The padded view has "
             << padded_vars.extent(0) << " elements and "
             << padded_vars.extent(1) << " variables but the patch has "
             << number_of_elements_ << " elements and " << number_of_variables_
             << " variables");
  ASSERT(ghost_variables.extent(0) == number_of_elements_ *
                                          number_of_variables_ *
                                          subcell_extents_.product(),
         "The ghost variables have "
             << ghost_variables.extent(0) << " values but expected "
             << number_of_elements_ * number_of_variables_ *
                    subcell_extents_.product());
  from_patch_.apply(
      exec, Kokkos::View<double*>(padded_vars.data(), padded_vars.span()),
      ghost_variables, number_of_variables_);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class GhostZones<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
