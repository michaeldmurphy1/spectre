// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "NumericalAlgorithms/FiniteDifference/TransfersKokkos.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace evolution::dg::subcell::fd_patch {
namespace {
// The projection of data on the DG face `dg_face_mesh` onto the subcell face
// with `subcell_face_extents` as a dense matrix, with the columns computed by
// `fd::project` so it is the same projection as on the host.
template <size_t FaceDim>
Matrix face_projection(const Mesh<FaceDim>& dg_face_mesh,
                       const Index<FaceDim>& subcell_face_extents) {
  if constexpr (FaceDim == 0) {
    (void)dg_face_mesh;
    (void)subcell_face_extents;
    Matrix result(1, 1);
    result(0, 0) = 1.0;
    return result;
  } else {
    const size_t columns = dg_face_mesh.number_of_grid_points();
    Matrix result(subcell_face_extents.product(), columns);
    DataVector unit(columns, 0.0);
    for (size_t column = 0; column < columns; ++column) {
      unit[column] = 1.0;
      const DataVector projected =
          fd::project(unit, dg_face_mesh, subcell_face_extents);
      for (size_t row = 0; row < result.rows(); ++row) {
        result(row, column) = projected[row];
      }
      unit[column] = 0.0;
    }
    return result;
  }
}
}  // namespace

template <size_t Dim>
PackagedDataFromDg<Dim>::PackagedDataFromDg(
    const Kokkos::DefaultExecutionSpace& exec,
    const std::vector<Element<Dim>>& elements, const FdPatches<Dim>& fd_patches,
    const size_t patch, const size_t number_of_packaged_components,
    const size_t number_of_dg_components,
    const size_t variables_to_offset_in_dg_grid)
    : number_of_packaged_components_(number_of_packaged_components) {
  const auto& element_ids = fd_patches.element_ids(patch);
  if (elements.size() != element_ids.size()) {
    ERROR("Got " << elements.size() << " elements for patch " << patch
                 << ", which has " << element_ids.size() << " elements");
  }
  if (variables_to_offset_in_dg_grid + number_of_packaged_components >
      number_of_dg_components) {
    ERROR("The DG neighbors send "
          << number_of_dg_components << " components, but "
          << number_of_packaged_components << " are used after an offset of "
          << variables_to_offset_in_dg_grid);
  }
  const Mesh<Dim>& dg_mesh = fd_patches.dg_mesh(patch);
  if (dg_mesh.basis(0) == Spectral::Basis::ZernikeB1) {
    ERROR("FD patches don't support projecting DG data with the basis "
          << dg_mesh.basis(0));
  }
  const Index<Dim> subcell_extents = fd::mesh(dg_mesh).extents();
  const size_t number_of_elements = elements.size();
  for (size_t d = 1; d < Dim; ++d) {
    if (subcell_extents[d] != subcell_extents[0]) {
      ERROR("FD patches need isotropic subcell extents, but got "
            << subcell_extents);
    }
  }

  // All face sets of a batch have the same number of points
  Index<Dim> face_extents = subcell_extents;
  ++face_extents[0];
  const size_t number_of_face_points = face_extents.product();
  packaged_data_size_ = Dim * number_of_elements *
                        number_of_packaged_components * number_of_face_points;

  // The projection in each dimension, stored once for each side
  std::array<std::optional<size_t>, Dim> lower_projections{};
  std::array<std::optional<size_t>, Dim> upper_projections{};
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    const Element<Dim>& element = elements[slot];
    if (element.id() != element_ids[slot]) {
      ERROR("Element " << element.id() << " is in slot " << slot
                       << " but the slot of patch " << patch << " is element "
                       << element_ids[slot]);
    }
    for (const auto& direction : Direction<Dim>::all_directions()) {
      if (not element.neighbors().contains(direction)) {
        continue;
      }
      const auto& neighbors = element.neighbors().at(direction);
      if (neighbors.size() != 1) {
        ERROR("FD patches need conforming elements, but element "
              << element.id() << " has " << neighbors.size()
              << " neighbors in direction " << direction);
      }
      const ElementId<Dim>& neighbor_id = *neighbors.ids().begin();
      if (fd_patches.location(neighbor_id).has_value()) {
        continue;
      }
      if (neighbor_id.block_id() != element.id().block_id()) {
        ERROR(
            "FD patches don't support DG neighbors in other blocks yet, but "
            "element "
            << element.id() << " has the DG neighbor " << neighbor_id
            << " in direction " << direction);
      }
      const size_t dimension = direction.dimension();
      const Mesh<Dim - 1> dg_face_mesh = dg_mesh.slice_away(dimension);
      const size_t number_of_dg_face_points =
          dg_face_mesh.number_of_grid_points();
      const Face face{
          .slot = slot,
          .direction = direction,
          .neighbor = neighbor_id,
          .data_offset = received_data_size_,
          .data_size = number_of_dg_components * number_of_dg_face_points};

      const bool upper = direction.side() == Side::Upper;
      auto& projection = upper ? gsl::at(upper_projections, dimension)
                               : gsl::at(lower_projections, dimension);
      auto& transfers = upper ? to_upper_ : to_lower_;
      if (not projection.has_value()) {
        projection = transfers.store_matrix(face_projection(
            dg_face_mesh, subcell_extents.slice_away(dimension)));
      }
      Index<Dim> extents_with_faces = subcell_extents;
      ++extents_with_faces[dimension];
      const size_t face_index = upper ? subcell_extents[dimension] : 0;
      const size_t face_set = dimension * number_of_elements + slot;
      transfers.add_matrix(
          face.data_offset +
              variables_to_offset_in_dg_grid * number_of_dg_face_points,
          number_of_dg_face_points,
          face_set * number_of_packaged_components * number_of_face_points,
          number_of_face_points, projection.value(),
          ::fd::device::slab_points(extents_with_faces, dimension, face_index,
                                    face_index + 1));
      received_data_size_ += face.data_size;
      faces_.push_back(face);
    }
  }
  to_lower_.copy_to_device(exec);
  to_upper_.copy_to_device(exec);
}

template <size_t Dim>
void PackagedDataFromDg<Dim>::apply(
    const Kokkos::DefaultExecutionSpace& exec,
    const Kokkos::View<double*>& lower_packaged_data,
    const Kokkos::View<double*>& upper_packaged_data,
    const Kokkos::View<const double*>& received_data) const {
  if (faces_.empty()) {
    return;
  }
  ASSERT(lower_packaged_data.extent(0) == packaged_data_size_ and
             upper_packaged_data.extent(0) == packaged_data_size_,
         "The packaged data have " << lower_packaged_data.extent(0) << " and "
                                   << upper_packaged_data.extent(0)
                                   << " values but expected "
                                   << packaged_data_size_);
  ASSERT(received_data.extent(0) == received_data_size_,
         "The received data has " << received_data.extent(0)
                                  << " values but expected "
                                  << received_data_size_);
  to_lower_.apply(exec, lower_packaged_data, received_data,
                  number_of_packaged_components_);
  to_upper_.apply(exec, upper_packaged_data, received_data,
                  number_of_packaged_components_);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class PackagedDataFromDg<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
