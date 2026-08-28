// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/SphericalShellPowerMonitor.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/TensorYlm/ApplyFilter.hpp"
#include "NumericalAlgorithms/TensorYlm/CartToSphere.hpp"
#include "NumericalAlgorithms/TensorYlm/Helpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave::power_monitor {
namespace {

constexpr size_t radial_monitor_index = 0;
constexpr size_t angular_monitor_index = 1;

template <typename TensorType>
DataVector tensor_radial_power_monitor(const TensorType& tensor,
                                       const Mesh<3>& mesh) {
  DataVector squared_power(mesh.extents(0), 0.0);
  DataVector component_power{};
  for (size_t component = 0; component < tensor.size(); ++component) {
    PowerMonitors::spherical_shell_radial_power_monitor(
        make_not_null(&component_power), tensor[component], mesh);
    squared_power += square(component_power);
  }
  squared_power = sqrt(squared_power / static_cast<double>(tensor.size()));
  return squared_power;
}

size_t number_of_angular_coefficients(const size_t ell, const int spin_weight,
                                      const bool zero_m_is_real,
                                      const size_t radial_extents) {
  if (ell < static_cast<size_t>(std::abs(spin_weight))) {
    return 0;
  }
  const size_t coefficients_at_ell =
      zero_m_is_real ? 2 * ell + 1 : 2 * (ell + 1);
  return radial_extents * coefficients_at_ell;
}

template <typename TensorType>
void accumulate_tensor_angular_power(
    const gsl::not_null<DataVector*> weighted_squared_power,
    const gsl::not_null<std::vector<size_t>*> counts, const TensorType& tensor,
    const Mesh<3>& mesh) {
  const size_t radial_extents = mesh.extents(0);
  const size_t ell_max = mesh.extents(1) - 1;
  constexpr bool zero_m_is_real = TensorType::rank() == 0;
  DataVector component_power{};
  for (size_t component = 0; component < tensor.size(); ++component) {
    const int spin_weight = ylm::TensorYlm::helpers::component_spin_weight<
        typename TensorType::structure>(component);
    PowerMonitors::spherical_shell_angular_power_monitor(
        make_not_null(&component_power), tensor[component], mesh, spin_weight,
        zero_m_is_real);
    for (size_t ell = 0; ell <= ell_max; ++ell) {
      const size_t component_count = number_of_angular_coefficients(
          ell, spin_weight, zero_m_is_real, radial_extents);
      (*weighted_squared_power)[ell] +=
          static_cast<double>(component_count) * square(component_power[ell]);
      (*counts)[ell] += component_count;
    }
  }
}

void normalize_angular_power(const gsl::not_null<DataVector*> power,
                             const std::vector<size_t>& counts) {
  if (power->size() != counts.size()) {
    ERROR(
        "The angular power and count buffers must have the same size, but "
        "got "
        << power->size() << " and " << counts.size() << ".");
  }
  for (size_t ell = 0; ell < power->size(); ++ell) {
    (*power)[ell] =
        counts[ell] == 0
            ? 0.0
            : sqrt((*power)[ell] / static_cast<double>(counts[ell]));
  }
}

}  // namespace

void fill_sw_cart_to_sphere_matrix(
    const gsl::not_null<SwCartToSphereMatrix*> matrix, const size_t ell_max) {
  if (not matrix->i.has_value()) {
    matrix->i.emplace();
    ylm::TensorYlm::fill_cart_to_sphere<
        typename tnsr::i<DataVector, 3>::structure>(
        make_not_null(&matrix->i.value()), ell_max,
        ylm::TensorYlm::CoefficientNormalization::Spherepack);
  }
}

SwShellPowerMonitors sw_shell_power_monitors(
    const gsl::not_null<SwCartToSphereMatrix*> cart_to_sphere_matrix,
    const Variables<
        ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>& sw_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid) {
  if (mesh.basis(0) == Spectral::Basis::SphericalHarmonic or
      mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      mesh.basis(2) != Spectral::Basis::SphericalHarmonic) {
    ERROR(
        "SW spherical-shell power monitors require the mesh dimensions to "
        "be ordered (radial, theta, phi), but the mesh is "
        << mesh);
  }
  const size_t radial_extents = mesh.extents(0);
  const size_t ell_max = mesh.extents(1) - 1;
  const size_t m_max = (mesh.extents(2) - 1) / 2;
  if (ell_max != m_max) {
    ERROR(
        "SW spherical-shell power monitors require l_max == m_max, but got "
        "l_max = "
        << ell_max << " and m_max = " << m_max << ".");
  }
  const auto& spherepack = ylm::get_spherepack_cache(ell_max);

  SwShellPowerMonitors result{};

  // Radial monitors: computed from the raw inertial variables.
  const auto& psi = get<ScalarWave::Tags::Psi>(sw_vars);
  const auto& pi = get<ScalarWave::Tags::Pi>(sw_vars);
  const auto& phi = get<ScalarWave::Tags::Phi<3>>(sw_vars);

  result.psi[radial_monitor_index] = tensor_radial_power_monitor(psi, mesh);
  result.pi[radial_monitor_index] = tensor_radial_power_monitor(pi, mesh);
  result.phi[radial_monitor_index] = tensor_radial_power_monitor(phi, mesh);

  // Angular monitors: transform variables to TensorYlm spectral basis.
  fill_sw_cart_to_sphere_matrix(cart_to_sphere_matrix, ell_max);

  const size_t n_phys = mesh.number_of_grid_points();
  const size_t n_spec = radial_extents * spherepack.spectral_size();

  // Frame-transform the variables: Psi and Pi are scalars (unchanged);
  // Phi_i is transformed from the inertial to the grid frame.
  namespace fd = ylm::TensorYlm::filter_detail;
  Variables<fd::sw_vars_list<Frame::Grid>> sw_grid_frame(n_phys);
  fd::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&sw_grid_frame), sw_vars,
                                    jac_inertial_to_grid);

  // SH analysis (nodal to modal) in the angular directions.
  Variables<fd::sw_vars_list<Frame::Grid>> sw_modal(n_spec);
  fd::nodal_to_modal_ylm(make_not_null(&sw_modal), sw_grid_frame, spherepack,
                         radial_extents);

  // Apply Cartesian-to-TensorYlm transform to the Phi SH coefficients.
  // Psi and Pi are scalars and need no further basis transformation.
  Variables<fd::sw_vars_list<Frame::Grid>> phi_tensor_ylm_vars(n_spec);
  for (auto& comp :
       get<ScalarWave::Tags::Phi<3, Frame::Grid>>(phi_tensor_ylm_vars)) {
    comp = 0.0;
  }
  {
    const gsl::span<double> src(
        get<ScalarWave::Tags::Phi<3, Frame::Grid>>(sw_modal)[0].data(),
        3 * n_spec);
    gsl::span<double> dest(
        get<ScalarWave::Tags::Phi<3, Frame::Grid>>(phi_tensor_ylm_vars)[0]
            .data(),
        3 * n_spec);
    for (size_t offset = 0; offset < radial_extents; ++offset) {
      cart_to_sphere_matrix->i->increment_multiply_on_right(
          make_not_null(&dest), offset, radial_extents, src, offset,
          radial_extents);
    }
  }

  // Accumulate angular power: Psi and Pi as real scalars, Phi in TensorYlm
  // basis.
  result.psi[angular_monitor_index] = DataVector(ell_max + 1, 0.0);
  result.pi[angular_monitor_index] = DataVector(ell_max + 1, 0.0);
  result.phi[angular_monitor_index] = DataVector(ell_max + 1, 0.0);
  std::vector<size_t> psi_counts(ell_max + 1, 0);
  std::vector<size_t> pi_counts(ell_max + 1, 0);
  std::vector<size_t> phi_counts(ell_max + 1, 0);

  accumulate_tensor_angular_power(
      make_not_null(&result.psi[angular_monitor_index]),
      make_not_null(&psi_counts), get<ScalarWave::Tags::Psi>(sw_modal), mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.pi[angular_monitor_index]),
      make_not_null(&pi_counts), get<ScalarWave::Tags::Pi>(sw_modal), mesh);
  accumulate_tensor_angular_power(
      make_not_null(&result.phi[angular_monitor_index]),
      make_not_null(&phi_counts),
      get<ScalarWave::Tags::Phi<3, Frame::Grid>>(phi_tensor_ylm_vars), mesh);

  normalize_angular_power(make_not_null(&result.psi[angular_monitor_index]),
                          psi_counts);
  normalize_angular_power(make_not_null(&result.pi[angular_monitor_index]),
                          pi_counts);
  normalize_angular_power(make_not_null(&result.phi[angular_monitor_index]),
                          phi_counts);
  return result;
}

}  // namespace ScalarWave::power_monitor
