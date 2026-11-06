// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FdPatch/BoundaryMessagesKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <algorithm>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/FdPatch/GhostZonesKokkos.hpp"
#include "Evolution/DgSubcell/FdPatch/InboxTags.hpp"
#include "Evolution/DgSubcell/FdPatch/PackagedDataFromDgKokkos.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace evolution::dg::subcell::fd_patch {
namespace {
template <size_t Dim>
using MessageIndex = std::unordered_map<
    ElementId<Dim>,
    DirectionalIdMap<Dim, const evolution::dg::BoundaryData<Dim>*>>;

// The messages by receiver and sender
template <size_t Dim>
MessageIndex<Dim> index_messages(
    const typename Tags::BoundaryDataInbox<Dim>::messages_type& messages) {
  MessageIndex<Dim> result{};
  for (const auto& [receiver, sender, data] : messages) {
    if (not result[receiver].emplace(sender, &data).second) {
      ERROR("Element " << receiver << " of an FD patch received data from "
                       << sender << " twice");
    }
  }
  return result;
}

// The received data of a face, or an error if it is missing
template <size_t Dim>
const evolution::dg::BoundaryData<Dim>& received(
    const MessageIndex<Dim>& messages, const ElementId<Dim>& receiver,
    const DirectionalId<Dim>& sender) {
  const auto receiver_messages = messages.find(receiver);
  if (receiver_messages == messages.end() or
      not receiver_messages->second.contains(sender)) {
    ERROR("Element " << receiver << " of an FD patch didn't receive data from "
                     << sender);
  }
  return *receiver_messages->second.at(sender);
}
}  // namespace

template <size_t Dim>
bool received_all_boundary_data(
    const GhostZones<Dim>& ghost_zones,
    const std::vector<ElementId<Dim>>& element_ids,
    const typename Tags::BoundaryDataInbox<Dim>::messages_type& messages) {
  if (messages.size() < ghost_zones.external_faces().size()) {
    return false;
  }
  const auto index = index_messages<Dim>(messages);
  return std::all_of(
      ghost_zones.external_faces().begin(), ghost_zones.external_faces().end(),
      [&element_ids, &messages = index](const auto& face) {
        const auto receiver_messages = messages.find(element_ids[face.slot]);
        return receiver_messages != messages.end() and
               receiver_messages->second.contains(
                   DirectionalId<Dim>{face.direction, face.neighbor});
      });
}

template <size_t Dim>
void unpack_boundary_data(
    const gsl::span<double> ghost_data, const gsl::span<double> mortar_data,
    const GhostZones<Dim>& ghost_zones,
    const PackagedDataFromDg<Dim>& packaged_data_from_dg,
    const std::vector<ElementId<Dim>>& element_ids,
    const typename Tags::BoundaryDataInbox<Dim>::messages_type& messages,
    const size_t number_of_rdmp_values) {
  if (ghost_data.size() != ghost_zones.external_data_size() or
      mortar_data.size() != packaged_data_from_dg.received_data_size()) {
    ERROR("The buffers have " << ghost_data.size() << " and "
                              << mortar_data.size() << " values but expected "
                              << ghost_zones.external_data_size() << " and "
                              << packaged_data_from_dg.received_data_size());
  }
  const auto index = index_messages<Dim>(messages);
  for (const auto& face : ghost_zones.external_faces()) {
    const ElementId<Dim>& receiver = element_ids[face.slot];
    const DirectionalId<Dim> sender{face.direction, face.neighbor};
    const auto& data = received(index, receiver, sender);
    if (not data.ghost_cell_data.has_value() or
        data.ghost_cell_data->size() !=
            face.data_size + number_of_rdmp_values) {
      ERROR("Element " << receiver << " of an FD patch expected "
                       << face.data_size << " ghost values and "
                       << number_of_rdmp_values << " RDMP values from "
                       << sender << " but got "
                       << (data.ghost_cell_data.has_value()
                               ? data.ghost_cell_data->size()
                               : 0_st)
                       << " values with the volume mesh "
                       << data.volume_mesh_ghost_cell_data);
    }
    std::copy_n(data.ghost_cell_data->begin(), face.data_size,
                ghost_data.subspan(face.data_offset, face.data_size).begin());
  }
  for (const auto& face : packaged_data_from_dg.faces()) {
    const ElementId<Dim>& receiver = element_ids[face.slot];
    const DirectionalId<Dim> sender{face.direction, face.neighbor};
    const auto& data = received(index, receiver, sender);
    if (not data.boundary_correction_data.has_value() or
        data.boundary_correction_data->size() != face.data_size) {
      ERROR("Element " << receiver << " of an FD patch expected "
                       << face.data_size
                       << " mortar values from the DG element " << sender
                       << " but got "
                       << (data.boundary_correction_data.has_value()
                               ? data.boundary_correction_data->size()
                               : 0_st)
                       << " values with the mortar mesh "
                       << data.boundary_correction_mesh);
    }
    std::copy_n(data.boundary_correction_data->begin(), face.data_size,
                mortar_data.subspan(face.data_offset, face.data_size).begin());
  }
}

template <size_t Dim>
std::vector<OutgoingBoundaryData<Dim>> pack_boundary_data(
    const GhostZones<Dim>& ghost_zones, const gsl::span<const double> send_data,
    const std::vector<ElementId<Dim>>& element_ids, const Mesh<Dim>& dg_mesh,
    const std::vector<RdmpTciData>& rdmp_tci_data,
    const TimeStepId& next_time_step_id, const int tci_status,
    const size_t integration_order) {
  if (send_data.size() != ghost_zones.send_data_size() or
      rdmp_tci_data.size() != element_ids.size()) {
    ERROR("Got " << send_data.size() << " values to send and the RDMP data of "
                 << rdmp_tci_data.size() << " elements but expected "
                 << ghost_zones.send_data_size() << " values and "
                 << element_ids.size() << " elements");
  }
  const Mesh<Dim> subcell_mesh = fd::mesh(dg_mesh);
  std::vector<OutgoingBoundaryData<Dim>> result{};
  result.reserve(ghost_zones.external_faces().size());
  for (const auto& face : ghost_zones.external_faces()) {
    const RdmpTciData& rdmp = rdmp_tci_data[face.slot];
    const size_t rdmp_size =
        rdmp.max_variables_values.size() + rdmp.min_variables_values.size();
    DataVector ghost_cell_data{face.send_size + rdmp_size};
    const auto sent = send_data.subspan(face.send_offset, face.send_size);
    auto end = std::copy(sent.begin(), sent.end(), ghost_cell_data.begin());
    end = std::copy(rdmp.max_variables_values.begin(),
                    rdmp.max_variables_values.end(), end);
    std::copy(rdmp.min_variables_values.begin(),
              rdmp.min_variables_values.end(), end);
    result.push_back(OutgoingBoundaryData<Dim>{
        .receiver = face.neighbor,
        .receiver_is_dg = face.neighbor_is_dg,
        .sender = DirectionalId<Dim>{face.direction_from_neighbor,
                                     element_ids[face.slot]},
        .data = evolution::dg::BoundaryData<Dim>{
            .volume_mesh = dg_mesh,
            .volume_mesh_ghost_cell_data = subcell_mesh,
            .boundary_correction_mesh = std::nullopt,
            .ghost_cell_data = std::move(ghost_cell_data),
            .boundary_correction_data = std::nullopt,
            .validity_range = next_time_step_id,
            .tci_status = tci_status,
            .integration_order = integration_order,
            .interpolated_boundary_data = std::nullopt}});
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template bool received_all_boundary_data(                                 \
      const GhostZones<DIM(data)>& ghost_zones,                             \
      const std::vector<ElementId<DIM(data)>>& element_ids,                 \
      const typename Tags::BoundaryDataInbox<DIM(data)>::messages_type&     \
          messages);                                                        \
  template void unpack_boundary_data(                                       \
      gsl::span<double> ghost_data, gsl::span<double> mortar_data,          \
      const GhostZones<DIM(data)>& ghost_zones,                             \
      const PackagedDataFromDg<DIM(data)>& packaged_data_from_dg,           \
      const std::vector<ElementId<DIM(data)>>& element_ids,                 \
      const typename Tags::BoundaryDataInbox<DIM(data)>::messages_type&     \
          messages,                                                         \
      size_t number_of_rdmp_values);                                        \
  template std::vector<OutgoingBoundaryData<DIM(data)>> pack_boundary_data( \
      const GhostZones<DIM(data)>& ghost_zones,                             \
      gsl::span<const double> send_data,                                    \
      const std::vector<ElementId<DIM(data)>>& element_ids,                 \
      const Mesh<DIM(data)>& dg_mesh,                                       \
      const std::vector<RdmpTciData>& rdmp_tci_data,                        \
      const TimeStepId& next_time_step_id, int tci_status,                  \
      size_t integration_order);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd_patch
#endif  // SPECTRE_KOKKOS
