// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Block.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

namespace evolution::dg::subcell::fd_patch {
namespace {
// The projection of the volume data of a DG neighbor in `direction` onto the
// ghost zone, as a dense matrix. The same dimension-by-dimension projection as
// in `insert_or_update_neighbor_volume_data_impl`.
template <size_t Dim>
Matrix dg_to_ghost_zone_projection(const Mesh<Dim>& dg_mesh,
                                   const Index<Dim>& subcell_extents,
                                   const size_t ghost_width,
                                   const Direction<Dim>& direction) {
  std::array<const Matrix*, Dim> matrices{};
  size_t rows = 1;
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(matrices, d) =
        d == direction.dimension()
            ? &fd::projection_matrix(
                  dg_mesh.slice_through(d), subcell_extents[d], ghost_width,
                  direction.opposite().side(), Spectral::Parity::Uninitialized)
            : &fd::projection_matrix(dg_mesh.slice_through(d),
                                     subcell_extents[d],
                                     Spectral::Quadrature::CellCentered,
                                     Spectral::Parity::Uninitialized);
    rows *= gsl::at(matrices, d)->rows();
  }
  Matrix result(rows, dg_mesh.number_of_grid_points());
  for (size_t row = 0; row < result.rows(); ++row) {
    for (size_t column = 0; column < result.columns(); ++column) {
      double value = 1.0;
      size_t row_remainder = row;
      size_t column_remainder = column;
      for (size_t d = 0; d < Dim; ++d) {
        const Matrix& matrix = *gsl::at(matrices, d);
        value *= matrix(row_remainder % matrix.rows(),
                        column_remainder % dg_mesh.extents(d));
        row_remainder /= matrix.rows();
        column_remainder /= dg_mesh.extents(d);
      }
      result(row, column) = value;
    }
  }
  return result;
}
}  // namespace

template <size_t Dim>
std::vector<size_t> cells_for_neighbor_ghost_zone(
    const Index<Dim>& extents, const size_t ghost_width,
    const Direction<Dim>& direction,
    const OrientationMap<Dim>& orientation_of_neighbor) {
  const size_t dimension = direction.dimension();
  const size_t first =
      direction.side() == Side::Lower ? 0 : extents[dimension] - ghost_width;
  auto slab =
      ::fd::device::slab_points(extents, dimension, first, first + ghost_width);
  if (orientation_of_neighbor.is_aligned()) {
    return slab;
  }
  // Reorient the positions of the cells in the slab, as the data would be
  Index<Dim> slab_extents = extents;
  slab_extents[dimension] = ghost_width;
  DataVector positions(slab.size());
  std::iota(positions.begin(), positions.end(), 0.0);
  const DataVector oriented_positions =
      orient_variables(positions, slab_extents, orientation_of_neighbor);
  std::vector<size_t> result(slab.size());
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = slab[static_cast<size_t>(oriented_positions[i])];
  }
  return result;
}

template <size_t Dim>
GhostZones<Dim>::GhostZones(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::vector<Element<Dim>>& elements,
    const std::vector<
        DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>>&
        fd_to_neighbor_fd_interpolants,
    const FdPatches<Dim>& fd_patches, const size_t patch,
    const size_t ghost_width, const size_t number_of_variables)
    : number_of_elements_(elements.size()),
      subcell_extents_(fd::mesh(fd_patches.dg_mesh(patch)).extents()),
      ghost_width_(ghost_width),
      number_of_variables_(number_of_variables) {
  const auto& element_ids = fd_patches.element_ids(patch);
  const Mesh<Dim>& dg_mesh = fd_patches.dg_mesh(patch);
  if (elements.size() != element_ids.size()) {
    ERROR("Got " << elements.size() << " elements for patch " << patch
                 << ", which has " << element_ids.size() << " elements");
  }
  if (fd_to_neighbor_fd_interpolants.size() != elements.size()) {
    ERROR("Got the interpolants of " << fd_to_neighbor_fd_interpolants.size()
                                     << " elements for the " << elements.size()
                                     << " elements of patch " << patch);
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
      const size_t dimension = direction.dimension();
      const bool upper = direction.side() == Side::Upper;
      const auto ghost_zone_points = ::fd::device::padded_ghost_zone_points(
          subcell_extents_, ghost_width, dimension, upper);
      if (not location.has_value() or location->patch != patch) {
        add_external_face(element, slot, number_of_padded_cells, direction,
                          neighbor_id, not location.has_value(),
                          ghost_zone_points, dg_mesh,
                          fd_to_neighbor_fd_interpolants[slot]);
        continue;
      }
      // The cells of the neighbor next to the shared face, in the order of the
      // ghost zone. The neighbor has the same subcell extents (all elements of
      // the patch have the same mesh), and if it is in another block, its
      // cells line up with the ghost cells (see
      // `check_cells_coincide_across_blocks`).
      const auto& orientation = neighbors.orientation(neighbor_id);
      const auto source_points = cells_for_neighbor_ghost_zone(
          subcell_extents_, ghost_width, orientation(direction.opposite()),
          orientation.inverse_map());
      from_patch_.add_copy(location->slot * element_stride, number_of_cells,
                           slot * padded_element_stride, number_of_padded_cells,
                           source_points, ghost_zone_points);
    }
  }
  from_patch_.copy_to_device(exec);
  from_external_.copy_to_device(exec);
  to_external_.copy_to_device(exec);
}

template <size_t Dim>
void GhostZones<Dim>::add_external_face(
    const Element<Dim>& element, const size_t slot,
    const size_t number_of_padded_cells, const Direction<Dim>& direction,
    const ElementId<Dim>& neighbor_id, const bool neighbor_is_dg,
    const std::vector<size_t>& ghost_zone_points, const Mesh<Dim>& dg_mesh,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        fd_to_neighbor_fd_interpolants) {
  const size_t number_of_ghost_points = ghost_zone_points.size();
  const size_t padded_element_offset =
      slot * number_of_variables_ * number_of_padded_cells;
  ExternalFace face{
      .slot = slot,
      .direction = direction,
      .neighbor = neighbor_id,
      .direction_from_neighbor = element.neighbors().at(direction).orientation(
          neighbor_id)(direction.opposite()),
      .neighbor_is_dg = neighbor_is_dg,
      .data_offset = external_data_size_,
      .data_size = number_of_variables_ * (neighbor_is_dg
                                               ? dg_mesh.number_of_grid_points()
                                               : number_of_ghost_points),
      .send_offset = send_data_size_,
      .send_size = number_of_variables_ * number_of_ghost_points};
  if (neighbor_is_dg) {
    if (neighbor_id.block_id() != element.id().block_id()) {
      ERROR(
          "FD patches don't support DG neighbors in other blocks yet, but "
          "element "
          << element.id() << " has the DG neighbor " << neighbor_id
          << " in direction " << direction);
    }
    if (dg_mesh.basis(0) == Spectral::Basis::ZernikeB1) {
      ERROR("FD patches don't support projecting DG data with the basis "
            << dg_mesh.basis(0) << " onto ghost zones");
    }
    if (not dg_projections_.contains(direction)) {
      dg_projections_[direction] =
          from_external_.store_matrix(dg_to_ghost_zone_projection(
              dg_mesh, subcell_extents_, ghost_width_, direction));
    }
    from_external_.add_matrix(face.data_offset, dg_mesh.number_of_grid_points(),
                              padded_element_offset, number_of_padded_cells,
                              dg_projections_.at(direction), ghost_zone_points);
  } else {
    std::vector<size_t> received_points(number_of_ghost_points);
    std::iota(received_points.begin(), received_points.end(), 0_st);
    from_external_.add_copy(face.data_offset, number_of_ghost_points,
                            padded_element_offset, number_of_padded_cells,
                            received_points, ghost_zone_points);
  }
  external_data_size_ += face.data_size;

  // The ghost data to send, as in `slice_data`
  const size_t number_of_cells = subcell_extents_.product();
  const size_t element_offset = slot * number_of_variables_ * number_of_cells;
  std::vector<size_t> sent_points(number_of_ghost_points);
  std::iota(sent_points.begin(), sent_points.end(), 0_st);
  if (const auto interpolant = fd_to_neighbor_fd_interpolants.find(
          DirectionalId<Dim>{direction, neighbor_id});
      interpolant != fd_to_neighbor_fd_interpolants.end()) {
    if (interpolant->second.has_value()) {
      to_external_.add_matrix(element_offset, number_of_cells, face.send_offset,
                              number_of_ghost_points,
                              interpolant->second->matrix(), sent_points);
    }
  } else {
    // The cells next to the face, in the order of the ghost zone of the
    // neighbor
    to_external_.add_copy(
        element_offset, number_of_cells, face.send_offset,
        number_of_ghost_points,
        cells_for_neighbor_ghost_zone(
            subcell_extents_, ghost_width_, direction,
            element.neighbors().at(direction).orientation(neighbor_id)),
        sent_points);
  }
  send_data_size_ += face.send_size;
  external_faces_.push_back(std::move(face));
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

template <size_t Dim>
void GhostZones<Dim>::fill_from_external(
    const Kokkos::DefaultExecutionSpace& exec,
    const ::fd::device::VolumeView& padded_vars,
    const Kokkos::View<const double*>& external_data) const {
  ASSERT(padded_vars.extent(0) == number_of_elements_ and
             padded_vars.extent(1) == number_of_variables_ and
             ::fd::device::detail::ghost_width(padded_vars, subcell_extents_) ==
                 ghost_width_,
         "The padded view has "
             << padded_vars.extent(0) << " elements and "
             << padded_vars.extent(1) << " variables but the patch has "
             << number_of_elements_ << " elements and " << number_of_variables_
             << " variables");
  ASSERT(external_data.extent(0) == external_data_size_,
         "The external data has " << external_data.extent(0)
                                  << " values but expected "
                                  << external_data_size_);
  from_external_.apply(
      exec, Kokkos::View<double*>(padded_vars.data(), padded_vars.span()),
      external_data, number_of_variables_);
}

template <size_t Dim>
void GhostZones<Dim>::slice_for_external(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<double*>& send_data,
    const Kokkos::View<const double*>& ghost_variables) const {
  ASSERT(send_data.extent(0) == send_data_size_,
         "The send buffer has " << send_data.extent(0)
                                << " values but expected " << send_data_size_);
  ASSERT(ghost_variables.extent(0) == number_of_elements_ *
                                          number_of_variables_ *
                                          subcell_extents_.product(),
         "The ghost variables have "
             << ghost_variables.extent(0) << " values but expected "
             << number_of_elements_ * number_of_variables_ *
                    subcell_extents_.product());
  to_external_.apply(exec, send_data, ghost_variables, number_of_variables_);
}

template <size_t Dim>
void check_cells_coincide_across_blocks(
    const Domain<Dim>& domain, const std::vector<Element<Dim>>& elements,
    const FdPatches<Dim>& fd_patches, const size_t patch,
    const size_t ghost_width) {
  const Mesh<Dim> subcell_mesh = fd::mesh(fd_patches.dg_mesh(patch));
  const auto logical_coords = logical_coordinates(subcell_mesh);
  for (const Element<Dim>& element : elements) {
    const ElementMap<Dim, Frame::Grid> element_map{
        element.id(), domain.blocks()[element.id().block_id()]};
    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const ElementId<Dim>& neighbor_id : neighbors.ids()) {
        const auto location = fd_patches.location(neighbor_id);
        if (neighbor_id.block_id() == element.id().block_id() or
            not location.has_value()) {
          continue;
        }
        const auto& orientation = neighbors.orientation(neighbor_id);
        const Mesh<Dim> neighbor_subcell_mesh =
            fd::mesh(fd_patches.dg_mesh(location->patch));
        if (neighbor_subcell_mesh != subcell_mesh) {
          ERROR(
              "FD patches need the same subcell mesh across block "
              "boundaries, but element "
              << element.id() << " has " << subcell_mesh << " and its neighbor "
              << neighbor_id << " has " << neighbor_subcell_mesh);
        }
        const ElementMap<Dim, Frame::Grid> neighbor_map{
            neighbor_id, domain.blocks()[neighbor_id.block_id()]};
        // The ghost cells of the element, and the cells of the neighbor that
        // are copied into them, in the same order
        const auto ghost_cells = element_map(fd::ghost_zone_logical_coordinates(
            subcell_mesh, ghost_width, direction));
        const auto neighbor_points = cells_for_neighbor_ghost_zone(
            subcell_mesh.extents(), ghost_width,
            orientation(direction.opposite()), orientation.inverse_map());
        tnsr::I<DataVector, Dim, Frame::ElementLogical> neighbor_logical_coords{
            neighbor_points.size()};
        for (size_t d = 0; d < Dim; ++d) {
          for (size_t i = 0; i < neighbor_points.size(); ++i) {
            neighbor_logical_coords.get(d)[i] =
                logical_coords.get(d)[neighbor_points[i]];
          }
        }
        const auto neighbor_cells = neighbor_map(neighbor_logical_coords);
        const auto offset = SetInterpolators_detail::periodic_offset(
            element.id(), neighbor_id, direction, orientation, element_map,
            neighbor_map);
        for (size_t d = 0; d < Dim; ++d) {
          const DataVector difference =
              ghost_cells.get(d) - neighbor_cells.get(d) - gsl::at(offset, d);
          const double scale = std::max(1.0, max(abs(ghost_cells.get(d))));
          if (max(abs(difference)) > 1.0e-10 * scale) {
            ERROR("The cells of the neighbor "
                  << neighbor_id << " of element " << element.id()
                  << " in direction " << direction
                  << " don't line up with its ghost cells (by up to "
                  << max(abs(difference)) << " in dimension " << d
                  << "), so the ghost data would have to be interpolated "
                     "between the blocks, which FD patches don't support "
                     "yet.");
          }
        }
      }
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                   \
  template std::vector<size_t> cells_for_neighbor_ghost_zone(    \
      const Index<DIM(data)>& extents, size_t ghost_width,       \
      const Direction<DIM(data)>& direction,                     \
      const OrientationMap<DIM(data)>& orientation_of_neighbor); \
  template class GhostZones<DIM(data)>;                          \
  template void check_cells_coincide_across_blocks(              \
      const Domain<DIM(data)>& domain,                           \
      const std::vector<Element<DIM(data)>>& elements,           \
      const FdPatches<DIM(data)>& fd_patches, size_t patch,      \
      size_t ghost_width);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
