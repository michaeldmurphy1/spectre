// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/FiniteDifference/Device/VolumeView.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace TestHelpers::fd::device {
/*!
 * \brief Data with `number_of_variables` variables in the layout of a
 * `Variables` with `extents`, where the value at each cell is
 * `f(var, cell)`. The cell indices `cell` are offset by `first_cell`.
 */
template <size_t Dim, typename F>
DataVector make_data(const Index<Dim>& extents,
                     const size_t number_of_variables,
                     const std::array<int, Dim>& first_cell, const F& f) {
  DataVector result(extents.product() * number_of_variables);
  for (size_t var = 0; var < number_of_variables; ++var) {
    for (IndexIterator<Dim> it(extents); it; ++it) {
      std::array<int, Dim> cell{};
      for (size_t d = 0; d < Dim; ++d) {
        gsl::at(cell, d) = static_cast<int>((*it)[d]) + gsl::at(first_cell, d);
      }
      result[var * extents.product() + it.collapsed_index()] = f(var, cell);
    }
  }
  return result;
}

/// \brief Ghost data of width `ghost_width` in all directions, computed from
/// `f` as in `make_data`, so cells outside the element continue the volume
/// data.
template <size_t Dim, typename F>
DirectionMap<Dim, DataVector> make_ghost_data(const Index<Dim>& extents,
                                              const size_t number_of_variables,
                                              const size_t ghost_width,
                                              const F& f) {
  DirectionMap<Dim, DataVector> result{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const size_t dim = direction.dimension();
    Index<Dim> ghost_extents = extents;
    ghost_extents[dim] = ghost_width;
    std::array<int, Dim> first_cell{};
    gsl::at(first_cell, dim) = direction.side() == Side::Lower
                                   ? -static_cast<int>(ghost_width)
                                   : static_cast<int>(extents[dim]);
    result[direction] =
        make_data(ghost_extents, number_of_variables, first_cell, f);
  }
  return result;
}

template <size_t Dim>
DirectionMap<Dim, gsl::span<const double>> make_spans(
    const DirectionMap<Dim, DataVector>& ghost_data) {
  DirectionMap<Dim, gsl::span<const double>> result{};
  for (const auto& [direction, data] : ghost_data) {
    result[direction] = gsl::make_span(data.data(), data.size());
  }
  return result;
}

/// The volume and ghost data of an element
template <size_t Dim>
struct ElementData {
  DataVector volume_vars;
  DirectionMap<Dim, DataVector> ghost_data;
};

/// Element data with `f(element, var, cell)` at each cell, see `make_data`
template <size_t Dim, typename F>
std::vector<ElementData<Dim>> make_element_data(
    const size_t number_of_elements, const size_t number_of_variables,
    const Index<Dim>& volume_extents, const size_t ghost_width, const F& f) {
  std::vector<ElementData<Dim>> result(number_of_elements);
  for (size_t element = 0; element < number_of_elements; ++element) {
    const auto element_f = [&f, element](const size_t var,
                                         const std::array<int, Dim>& cell) {
      return f(element, var, cell);
    };
    result[element].volume_vars = make_data(volume_extents, number_of_variables,
                                            std::array<int, Dim>{}, element_f);
    result[element].ghost_data = make_ghost_data(
        volume_extents, number_of_variables, ghost_width, element_f);
  }
  return result;
}

/// Element data with random values in [-1, 1]
template <size_t Dim>
std::vector<ElementData<Dim>> make_random_element_data(
    const gsl::not_null<std::mt19937*> generator,
    const size_t number_of_elements, const size_t number_of_variables,
    const Index<Dim>& volume_extents, const size_t ghost_width) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  return make_element_data(
      number_of_elements, number_of_variables, volume_extents, ghost_width,
      [&generator, &dist](const size_t /*element*/, const size_t /*var*/,
                          const std::array<int, Dim>& /*cell*/) {
        return dist(*generator);
      });
}

/// Copy the element data to a padded view on the device
template <size_t Dim>
::fd::device::VolumeView pack(const std::vector<ElementData<Dim>>& data,
                              const size_t number_of_variables,
                              const Index<Dim>& volume_extents,
                              const size_t ghost_width) {
  const auto padded = ::fd::device::make_volume_view(
      "padded", data.size(), number_of_variables, volume_extents, ghost_width);
  const auto host_padded = Kokkos::create_mirror_view(padded);
  for (size_t element = 0; element < data.size(); ++element) {
    const auto& volume_vars = data[element].volume_vars;
    ::fd::device::pack_with_ghost_zones(
        host_padded, element,
        gsl::make_span(volume_vars.data(), volume_vars.size()),
        make_spans(data[element].ghost_data), volume_extents);
  }
  Kokkos::deep_copy(padded, host_padded);
  return padded;
}

/// Copy views to the host
template <size_t Dim>
std::array<::fd::device::VolumeView::host_mirror_type, Dim> to_host(
    const std::array<::fd::device::VolumeView, Dim>& views) {
  std::array<::fd::device::VolumeView::host_mirror_type, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, gsl::at(views, d));
  }
  return result;
}

/// The data of one element in a host view, in the layout of a `Variables`
inline DataVector element_data(
    const ::fd::device::VolumeView::host_mirror_type& view,
    const size_t element) {
  const size_t size =
      view.extent(1) * view.extent(2) * view.extent(3) * view.extent(4);
  return DataVector{&view(element, 0, 0, 0, 0), size};
}
}  // namespace TestHelpers::fd::device
