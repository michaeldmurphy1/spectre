// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceViewsKokkos.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd {
template <typename FaceTags>
std::array<::fd::device::StridedVolumeView, 3> reconstructed_face_views(
    const BatchedVariables<FaceTags>& faces, const Index<3>& volume_extents) {
  static_assert(
      std::is_same_v<tmpl::front<FaceTags>,
                     tmpl::front<reconstructed_face_tags>>,
      "The face variables must start with the reconstructed variables.");
  constexpr size_t number_of_components =
      BatchedVariables<FaceTags>::number_of_independent_components;
  constexpr size_t number_of_reconstructed_components =
      Variables<reconstructed_face_tags>::number_of_independent_components;
  const size_t points_per_face_set = faces.number_of_grid_points();
  ASSERT(faces.number_of_elements() % 3 == 0,
         "The faces must hold one face set per direction of each element, but "
         "hold "
             << faces.number_of_elements() << " face sets.");
  const size_t number_of_elements = faces.number_of_elements() / 3;
  std::array<::fd::device::StridedVolumeView, 3> result{};
  for (size_t d = 0; d < 3; ++d) {
    Index<3> face_extents = volume_extents;
    ++face_extents[d];
    ASSERT(face_extents.product() == points_per_face_set,
           "The face sets have " << points_per_face_set
                                 << " points, but the faces in direction "
                                 << d << " of elements with " << volume_extents
                                 << " cells have " << face_extents.product()
                                 << ". The extents must be isotropic.");
    // The reconstructed variables are the first components of each face set,
    // and the points of a component are ordered with x varying fastest
    gsl::at(result, d) = ::fd::device::StridedVolumeView(
        faces.view().data() +
            d * number_of_elements * number_of_components * points_per_face_set,
        Kokkos::LayoutStride(
            number_of_elements, number_of_components * points_per_face_set,
            number_of_reconstructed_components, points_per_face_set,
            face_extents[2], face_extents[1] * face_extents[0], face_extents[1],
            face_extents[0], face_extents[0], 1));
  }
  return result;
}

template std::array<::fd::device::StridedVolumeView, 3>
reconstructed_face_views(const BatchedVariables<face_tags>& faces,
                         const Index<3>& volume_extents);
template std::array<::fd::device::StridedVolumeView, 3>
reconstructed_face_views(const BatchedVariables<device_face_tags>& faces,
                         const Index<3>& volume_extents);
template std::array<::fd::device::StridedVolumeView, 3>
reconstructed_face_views(
    const BatchedVariables<reconstruction_face_tags>& faces,
    const Index<3>& volume_extents);
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
