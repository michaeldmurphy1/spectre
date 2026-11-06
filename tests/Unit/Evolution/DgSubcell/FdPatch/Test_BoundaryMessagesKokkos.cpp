// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatches.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/VolumeViewKokkos.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace fd_patch = evolution::dg::subcell::fd_patch;
using evolution::dg::subcell::RdmpTciData;
using Inbox = fd_patch::Tags::BoundaryDataInbox<1>;
template <size_t Dim>
using Interpolants =
    DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;

template <size_t Dim>
std::vector<Element<Dim>> patch_elements(
    const fd_patch::FdPatches<Dim>& patches, const size_t patch,
    const Domain<Dim>& domain,
    const std::vector<std::array<size_t, Dim>>& refinement) {
  std::vector<Element<Dim>> result{};
  for (const auto& id : patches.element_ids(patch)) {
    result.push_back(
        domain::create_initial_element(id, domain.blocks(), refinement));
  }
  return result;
}

template <typename HostView>
Kokkos::View<double*> to_device(const HostView& host) {
  const Kokkos::View<double*> result("data", host.size());
  Kokkos::deep_copy(
      result,
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          host.data(), host.size()));
  return result;
}

RdmpTciData make_rdmp(const size_t slot) {
  const auto value = static_cast<double>(slot);
  return RdmpTciData{.max_variables_values = {value, value + 1.0},
                     .min_variables_values = {-value, -value - 1.0}};
}

// Three periodic blocks in a row, each with two elements, so each element has
// a neighbor in another patch. Each patch sends its boundary data to the other
// patches, and after receiving, every ghost zone holds the data of the cells
// next to it in the periodic domain.
void test_exchange_between_patches() {
  const Kokkos::DefaultExecutionSpace exec{};
  constexpr size_t number_of_variables = 2;
  constexpr size_t ghost_width = 2;
  // 4 DG points, so 7 cells per element and 14 per block
  constexpr long cells_per_block = 14;
  constexpr long cells_in_domain = 3 * cells_per_block;
  const auto domain = rectilinear_domain<1>(
      Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
      {}, {}, std::array{true});
  const std::vector<std::array<size_t, 1>> refinement(3, {{1}});
  const std::vector<std::array<size_t, 1>> extents(3, {{4}});
  const fd_patch::FdPatches<1> patches{domain.blocks(),
                                       refinement,
                                       extents,
                                       Spectral::Quadrature::GaussLobatto,
                                       {},
                                       false};
  REQUIRE(patches.number_of_patches() == 3);
  const Mesh<1>& dg_mesh = patches.dg_mesh(0);
  const Slab slab{0.0, 1.0};
  const TimeStepId time_step_id{true, 0, slab.start()};
  const TimeStepId next_time_step_id{true, 0, slab.end()};

  // The value of a variable in a cell, by its index in the periodic domain
  const auto value = [](const size_t variable, const long cell) {
    return 1000.0 * static_cast<double>(variable) +
           static_cast<double>((cell + cells_in_domain) % cells_in_domain);
  };
  const auto domain_cell = [](const ElementId<1>& id, const long cell) {
    return static_cast<long>(id.block_id()) * cells_per_block +
           static_cast<long>(id.segment_id(0).index()) * 7 + cell;
  };

  std::vector<std::vector<Element<1>>> elements(3);
  std::vector<fd_patch::GhostZones<1>> ghost_zones{};
  std::vector<fd_patch::PackagedDataFromDg<1>> packaged_data_from_dg{};
  std::vector<Kokkos::View<double*>> ghost_variables{};
  std::array<Inbox::type, 3> inboxes{};
  for (size_t patch = 0; patch < 3; ++patch) {
    elements[patch] = patch_elements(patches, patch, domain, refinement);
    ghost_zones.emplace_back(exec, elements[patch],
                             std::vector<Interpolants<1>>(2), patches, patch,
                             ghost_width, number_of_variables);
    packaged_data_from_dg.emplace_back(exec, elements[patch], patches, patch, 3,
                                       3, 0);
    CHECK(packaged_data_from_dg.back().faces().empty());
    const Kokkos::View<double*>::host_mirror_type host_ghost_variables(
        "ghost variables", 2 * number_of_variables * 7);
    for (size_t slot = 0; slot < 2; ++slot) {
      for (size_t v = 0; v < number_of_variables; ++v) {
        for (size_t i = 0; i < 7; ++i) {
          host_ghost_variables((slot * number_of_variables + v) * 7 + i) =
              value(v, domain_cell(patches.element_ids(patch)[slot],
                                   static_cast<long>(i)));
        }
      }
    }
    ghost_variables.push_back(to_device(host_ghost_variables));
  }

  // Send
  for (size_t patch = 0; patch < 3; ++patch) {
    const Kokkos::View<double*> send_data("send data",
                                          ghost_zones[patch].send_data_size());
    ghost_zones[patch].slice_for_external(exec, send_data,
                                          ghost_variables[patch]);
    const auto host_send_data =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, send_data);
    const auto outgoing = fd_patch::pack_boundary_data(
        ghost_zones[patch],
        gsl::span<const double>{host_send_data.data(), host_send_data.size()},
        patches.element_ids(patch), dg_mesh,
        std::vector<RdmpTciData>{make_rdmp(0), make_rdmp(1)}, next_time_step_id,
        3, 2);
    REQUIRE(outgoing.size() == 2);
    for (const auto& message : outgoing) {
      CHECK_FALSE(message.receiver_is_dg);
      CHECK(message.data.volume_mesh == dg_mesh);
      CHECK(message.data.volume_mesh_ghost_cell_data ==
            evolution::dg::subcell::fd::mesh(dg_mesh));
      CHECK(message.data.validity_range == next_time_step_id);
      CHECK(message.data.tci_status == 3);
      CHECK(message.data.integration_order == 2);
      // The sliced data, then the RDMP maxima and minima
      REQUIRE(message.data.ghost_cell_data->size() ==
              number_of_variables * ghost_width + 4);
      const size_t slot = patches.location(message.sender.id())->slot;
      const DataVector& ghost_cell_data = message.data.ghost_cell_data.value();
      DataVector rdmp(4);
      std::copy_n(&ghost_cell_data[number_of_variables * ghost_width], 4,
                  rdmp.begin());
      CHECK(rdmp == DataVector{static_cast<double>(slot),
                               static_cast<double>(slot) + 1.0,
                               -static_cast<double>(slot),
                               -static_cast<double>(slot) - 1.0});
      const size_t receiver_patch = patches.location(message.receiver)->patch;
      CHECK(receiver_patch != patch);
      Inbox::insert_into_inbox(
          make_not_null(&gsl::at(inboxes, receiver_patch)), time_step_id,
          std::tuple{message.receiver, message.sender, message.data});
    }
  }

  // Receive
  for (size_t patch = 0; patch < 3; ++patch) {
    CAPTURE(patch);
    const auto& messages = gsl::at(inboxes, patch).at(time_step_id);
    const auto& element_ids = patches.element_ids(patch);
    CHECK(fd_patch::received_all_boundary_data(ghost_zones[patch], element_ids,
                                               messages));
    std::vector<double> host_external_data(
        ghost_zones[patch].external_data_size());
    fd_patch::unpack_boundary_data(
        gsl::span<double>{host_external_data.data(), host_external_data.size()},
        gsl::span<double>{}, ghost_zones[patch], packaged_data_from_dg[patch],
        element_ids, messages, 4);
    const auto padded_vars = fd::device::make_volume_view(
        exec, "padded", 2, number_of_variables, Index<1>{7}, ghost_width);
    ghost_zones[patch].fill_from_patch(exec, padded_vars,
                                       ghost_variables[patch]);
    ghost_zones[patch].fill_from_external(exec, padded_vars,
                                          to_device(host_external_data));
    const auto host_padded =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, padded_vars);
    for (size_t slot = 0; slot < 2; ++slot) {
      for (size_t v = 0; v < number_of_variables; ++v) {
        for (long i = -static_cast<long>(ghost_width);
             i < 7 + static_cast<long>(ghost_width); ++i) {
          CHECK(host_padded(
                    slot, v, 0, 0,
                    static_cast<size_t>(i + static_cast<long>(ghost_width))) ==
                value(v, domain_cell(element_ids[slot], i)));
        }
      }
    }
  }

  // Missing data
  auto messages = gsl::at(inboxes, 1).at(time_step_id);
  std::erase_if(messages, [&patches](const auto& message) {
    return std::get<0>(message) == patches.element_ids(1)[0];
  });
  CHECK_FALSE(fd_patch::received_all_boundary_data(
      ghost_zones[1], patches.element_ids(1), messages));
  std::vector<double> host_external_data(ghost_zones[1].external_data_size());
  CHECK_THROWS_WITH(
      fd_patch::unpack_boundary_data(
          gsl::span<double>{host_external_data.data(),
                            host_external_data.size()},
          gsl::span<double>{}, ghost_zones[1], packaged_data_from_dg[1],
          patches.element_ids(1), messages, 4),
      Catch::Matchers::ContainsSubstring("didn't receive data from"));
}

// The central block of 3x3 blocks with DG-only outer blocks, so the FD
// elements border DG elements of the same block, which send their volume data
// and their packaged data on the mortar.
void test_dg_neighbors(const gsl::not_null<std::mt19937*> generator) {
  const Kokkos::DefaultExecutionSpace exec{};
  constexpr size_t number_of_variables = 2;
  constexpr size_t number_of_packaged_components = 3;
  const auto domain = rectilinear_domain<2>(
      Index<2>{3, 3}, std::array<std::vector<double>, 2>{
                          {{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 2>> refinement(9, {{2, 2}});
  const std::vector<std::array<size_t, 2>> extents(9, {{5, 5}});
  const fd_patch::FdPatches<2> patches{domain.blocks(),
                                       refinement,
                                       extents,
                                       Spectral::Quadrature::GaussLobatto,
                                       {0, 1, 2, 3, 5, 6, 7, 8},
                                       true};
  const auto elements = patch_elements(patches, 0, domain, refinement);
  const auto& element_ids = patches.element_ids(0);
  const Mesh<2>& dg_mesh = patches.dg_mesh(0);
  const fd_patch::GhostZones<2> ghost_zones{
      exec,
      elements,
      std::vector<Interpolants<2>>(elements.size()),
      patches,
      0,
      2,
      number_of_variables};
  const fd_patch::PackagedDataFromDg<2> packaged_data_from_dg{
      exec,
      elements,
      patches,
      0,
      number_of_packaged_components,
      number_of_packaged_components,
      0};
  const Slab slab{0.0, 1.0};
  const TimeStepId time_step_id{true, 0, slab.start()};

  // The messages of the DG neighbors, with random data
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  fd_patch::Tags::BoundaryDataInbox<2>::type inbox{};
  std::vector<double> expected_ghost_data(ghost_zones.external_data_size());
  std::vector<double> expected_mortar_data(
      packaged_data_from_dg.received_data_size());
  REQUIRE(ghost_zones.external_faces().size() ==
          packaged_data_from_dg.faces().size());
  for (size_t i = 0; i < ghost_zones.external_faces().size(); ++i) {
    const auto& face = ghost_zones.external_faces()[i];
    const auto& mortar_face = packaged_data_from_dg.faces()[i];
    REQUIRE(face.neighbor_is_dg);
    REQUIRE(mortar_face.neighbor == face.neighbor);
    CHECK(fd_patch::received_all_boundary_data(ghost_zones, element_ids,
                                               inbox[time_step_id]) == false);
    evolution::dg::BoundaryData<2> data{};
    data.volume_mesh = dg_mesh;
    data.volume_mesh_ghost_cell_data = dg_mesh;
    data.ghost_cell_data = make_with_random_values<DataVector>(
        generator, make_not_null(&dist), DataVector(face.data_size + 4));
    data.boundary_correction_data = make_with_random_values<DataVector>(
        generator, make_not_null(&dist), DataVector(mortar_face.data_size));
    std::copy_n(data.ghost_cell_data->begin(), face.data_size,
                &expected_ghost_data[face.data_offset]);
    std::copy_n(data.boundary_correction_data->begin(), mortar_face.data_size,
                &expected_mortar_data[mortar_face.data_offset]);
    fd_patch::Tags::BoundaryDataInbox<2>::insert_into_inbox(
        make_not_null(&inbox), time_step_id,
        std::tuple{element_ids[face.slot],
                   DirectionalId<2>{face.direction, face.neighbor},
                   std::move(data)});
  }
  const auto& messages = inbox.at(time_step_id);
  CHECK(
      fd_patch::received_all_boundary_data(ghost_zones, element_ids, messages));
  std::vector<double> ghost_data(ghost_zones.external_data_size());
  std::vector<double> mortar_data(packaged_data_from_dg.received_data_size());
  fd_patch::unpack_boundary_data(
      gsl::span<double>{ghost_data.data(), ghost_data.size()},
      gsl::span<double>{mortar_data.data(), mortar_data.size()}, ghost_zones,
      packaged_data_from_dg, element_ids, messages, 4);
  CHECK(ghost_data == expected_ghost_data);
  CHECK(mortar_data == expected_mortar_data);

  // The same data twice
  auto duplicate_messages = messages;
  duplicate_messages.push_back(messages.front());
  CHECK_THROWS_WITH(fd_patch::received_all_boundary_data(
                        ghost_zones, element_ids, duplicate_messages),
                    Catch::Matchers::ContainsSubstring("twice"));

  // A DG neighbor with a different mesh
  auto wrong_messages = messages;
  const auto& face = ghost_zones.external_faces()[0];
  auto& wrong_data = std::get<2>(*std::find_if(
      wrong_messages.begin(), wrong_messages.end(),
      [&face, &element_ids](const auto& message) {
        return std::get<0>(message) == element_ids[face.slot] and
               std::get<1>(message) ==
                   DirectionalId<2>{face.direction, face.neighbor};
      }));
  wrong_data.ghost_cell_data = DataVector(10, 0.0);
  CHECK_THROWS_WITH(
      fd_patch::unpack_boundary_data(
          gsl::span<double>{ghost_data.data(), ghost_data.size()},
          gsl::span<double>{mortar_data.data(), mortar_data.size()},
          ghost_zones, packaged_data_from_dg, element_ids, wrong_messages, 4),
      Catch::Matchers::ContainsSubstring("ghost values and 4 RDMP values"));

  // The data sent to DG neighbors is keyed by the opposite direction
  const auto outgoing = fd_patch::pack_boundary_data(
      ghost_zones,
      gsl::span<const double>{
          std::vector<double>(ghost_zones.send_data_size()).data(),
          ghost_zones.send_data_size()},
      element_ids, dg_mesh,
      std::vector<RdmpTciData>(element_ids.size(), make_rdmp(0)), time_step_id,
      0, 1);
  REQUIRE(outgoing.size() == ghost_zones.external_faces().size());
  for (size_t i = 0; i < outgoing.size(); ++i) {
    const auto& external_face = ghost_zones.external_faces()[i];
    CHECK(outgoing[i].receiver_is_dg);
    CHECK(outgoing[i].receiver == external_face.neighbor);
    CHECK(outgoing[i].sender ==
          DirectionalId<2>{external_face.direction.opposite(),
                           element_ids[external_face.slot]});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FdPatch.BoundaryMessagesKokkos",
                  "[Evolution][Unit]") {
  MAKE_GENERATOR(generator);
  test_exchange_between_patches();
  test_dg_neighbors(make_not_null(&generator));
}
