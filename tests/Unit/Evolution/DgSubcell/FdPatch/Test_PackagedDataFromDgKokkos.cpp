// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using evolution::dg::subcell::fd_patch::FdPatches;
using evolution::dg::subcell::fd_patch::PackagedDataFromDg;
using PackagedTags = tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, 2>>;
using PackagedVars = Variables<PackagedTags>;

// 3x3 blocks with DG-only outer blocks and 4x4 elements per block, so the
// 2x2 interior elements of the central block are FD and border DG elements of
// the same block. The DG neighbors send more components than are used, with
// an offset, as for a combined system.
void test_packaged_data_from_dg(const gsl::not_null<std::mt19937*> generator) {
  constexpr size_t number_of_packaged_components =
      PackagedVars::number_of_independent_components;
  constexpr size_t number_of_dg_components = 5;
  constexpr size_t offset = 1;
  const Kokkos::DefaultExecutionSpace exec{};
  const auto domain = rectilinear_domain<2>(
      Index<2>{3, 3}, std::array<std::vector<double>, 2>{
                          {{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 2>> refinement(9, {{2, 2}});
  const std::vector<std::array<size_t, 2>> extents(9, {{5, 5}});
  const FdPatches<2> patches{domain.blocks(),
                             refinement,
                             extents,
                             Spectral::Quadrature::GaussLobatto,
                             {0, 1, 2, 3, 5, 6, 7, 8},
                             true};
  REQUIRE(patches.number_of_patches() == 1);
  std::vector<Element<2>> elements{};
  for (const auto& id : patches.element_ids(0)) {
    elements.push_back(
        domain::create_initial_element(id, domain.blocks(), refinement));
  }
  const size_t number_of_elements = elements.size();
  const PackagedDataFromDg<2> packaged_data_from_dg{
      exec,
      elements,
      patches,
      0,
      number_of_packaged_components,
      number_of_dg_components,
      offset};
  // Each element has two DG neighbors
  const auto& faces = packaged_data_from_dg.faces();
  REQUIRE(faces.size() == 8);
  size_t expected_offset = 0;
  for (const auto& face : faces) {
    CHECK(face.data_offset == expected_offset);
    CHECK(face.data_size == number_of_dg_components * 5);
    expected_offset += face.data_size;
  }
  CHECK(packaged_data_from_dg.received_data_size() == expected_offset);

  const Mesh<2>& dg_mesh = patches.dg_mesh(0);
  const Mesh<2> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_face_points = 10_st * 9_st;
  const size_t face_set_size =
      number_of_packaged_components * number_of_face_points;
  const size_t packaged_size = 2 * number_of_elements * face_set_size;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto host_received = make_with_random_values<DataVector>(
      generator, make_not_null(&dist),
      DataVector(packaged_data_from_dg.received_data_size()));
  auto expected_lower = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), DataVector(packaged_size));
  auto expected_upper = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), DataVector(packaged_size));

  const auto to_device = [](const DataVector& host) {
    const Kokkos::View<double*> result("data", host.size());
    Kokkos::deep_copy(
        result,
        Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
            host.data(), host.size()));
    return result;
  };
  const auto lower = to_device(expected_lower);
  const auto upper = to_device(expected_upper);
  packaged_data_from_dg.apply(exec, lower, upper, to_device(host_received));
  const auto host_lower =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lower);
  const auto host_upper =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upper);

  // The host correction of each face set
  for (size_t slot = 0; slot < number_of_elements; ++slot) {
    DirectionalIdMap<2, evolution::dg::MortarDataHolder<2>> mortar_data{};
    for (const auto& face : faces) {
      if (face.slot != slot) {
        continue;
      }
      auto& neighbor_data =
          mortar_data[DirectionalId<2>{face.direction, face.neighbor}]
              .neighbor();
      neighbor_data.mortar_data = DataVector{face.data_size};
      std::copy_n(&host_received[face.data_offset], face.data_size,
                  neighbor_data.mortar_data->begin());
      neighbor_data.face_mesh = dg_mesh.slice_away(face.direction.dimension());
    }
    for (size_t dimension = 0; dimension < 2; ++dimension) {
      const size_t face_set_offset =
          (dimension * number_of_elements + slot) * face_set_size;
      PackagedVars lower_vars{&expected_lower[face_set_offset], face_set_size};
      PackagedVars upper_vars{&expected_upper[face_set_offset], face_set_size};
      evolution::dg::subcell::correct_package_data<true>(
          make_not_null(&lower_vars), make_not_null(&upper_vars), dimension,
          elements[slot], subcell_mesh, mortar_data, offset);
    }
  }
  CHECK_ITERABLE_APPROX(DataVector(host_lower.data(), host_lower.extent(0)),
                        expected_lower);
  CHECK_ITERABLE_APPROX(DataVector(host_upper.data(), host_upper.extent(0)),
                        expected_upper);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.PackagedDataFromDgKokkos",
                  "[Evolution][Unit]") {
  MAKE_GENERATOR(generator);
  test_packaged_data_from_dg(make_not_null(&generator));
}
