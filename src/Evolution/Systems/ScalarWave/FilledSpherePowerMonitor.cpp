// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/FilledSpherePowerMonitor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/TensorYlm/ApplyFilter.hpp"
#include "NumericalAlgorithms/TensorYlm/Helpers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"

namespace ScalarWave::power_monitor {
namespace {

constexpr size_t radial_monitor_index = 0;
constexpr size_t angular_monitor_index = 1;

// Accumulate squared B3 Jacobi spectral coefficients from one TensorYlm
// component into both radial and angular sums in a single pass.
//
// `spec_buf` has layout spec_buf[s * n_r + i_r]: SPHEREPACK offset s at
// radial collocation point i_r. Groups by radial mode = (n_total+1)/2 and by
// angular degree l, where n_total = l + 2*k_spec. Modes with l < |spin_weight|
// are skipped.
//
// `offsets_by_l` must be pre-grouped for the correct zero_m_is_real value.
// `gathered` and `modal_buf` are caller-owned scratch buffers of size
// max_n_modes_l * n_r each (max_n_modes_l = 2*(l_max+1)).
void accumulate_b3_tensor_component_sums(
    const gsl::not_null<DataVector*> sum_sq_radial,
    const gsl::not_null<DataVector*> counts_radial,
    const gsl::not_null<DataVector*> sum_sq_angular,
    const gsl::not_null<DataVector*> counts_angular,
    const double* const spec_buf, const size_t n_r, const size_t n_r_max,
    const int spin_weight, const std::vector<std::vector<size_t>>& offsets_by_l,
    double* const gathered, double* const modal_buf) {
  const size_t l_max = offsets_by_l.size() - 1;
  const auto abs_spin_weight = static_cast<size_t>(std::abs(spin_weight));

  for (size_t l = 0; l <= l_max; ++l) {
    if (l < abs_spin_weight) {
      // Spin-weighted spherical harmonics vanish for l < |s|; skip.
      continue;
    }
    const size_t spectral_size_l = (n_r_max - l + 2) / 2;
    const auto& offsets = offsets_by_l[l];
    const size_t n_modes_l = offsets.size();

    // Gather radial profiles for all same-l SH modes into a contiguous buffer.
    // Layout: gathered[k * n_r + i_r] for the k-th offset at this l.
    for (size_t k = 0; k < n_modes_l; ++k) {
      std::copy(spec_buf + offsets[k] * n_r,        // NOLINT
                spec_buf + (offsets[k] + 1) * n_r,  // NOLINT
                gathered + k * n_r);                // NOLINT
    }

    // Apply the Jacobi NTM for angular degree l:
    //   modal_buf[spectral_size_l x n_modes_l]
    //     = NTM_l[spectral_size_l x n_r] * gathered[n_r x n_modes_l].
    const auto& ntm =
        Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB3,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, l, n_r_max);
    dgemm_<true>('N', 'N', spectral_size_l, n_modes_l, n_r, 1.0, ntm.data(),
                 ntm.spacing(), gathered, n_r, 0.0, modal_buf, spectral_size_l);

    // Accumulate squared Jacobi coefficients into radial and angular bins.
    for (size_t k_spec = 0; k_spec < spectral_size_l; ++k_spec) {
      const size_t n_total = l + 2 * k_spec;
      const size_t radial_mode = (n_total + 1) / 2;
      for (size_t col = 0; col < n_modes_l; ++col) {
        const double coeff_sq =
            square(modal_buf[k_spec + spectral_size_l * col]);  // NOLINT
        (*sum_sq_radial)[radial_mode] += coeff_sq;
        (*counts_radial)[radial_mode] += 1.0;
        (*sum_sq_angular)[l] += coeff_sq;
        (*counts_angular)[l] += 1.0;
      }
    }
  }
}

// Accumulate B3 spectral sums for all components of a TensorYlm tensor.
// Selects offsets_by_l_real (for scalars) or offsets_by_l_complex (rank > 0)
// based on the tensor rank.
template <typename TensorType>
void accumulate_b3_tensor_sums(
    const gsl::not_null<DataVector*> sum_sq_radial,
    const gsl::not_null<DataVector*> counts_radial,
    const gsl::not_null<DataVector*> sum_sq_angular,
    const gsl::not_null<DataVector*> counts_angular, const TensorType& tensor,
    const size_t n_r, const size_t n_r_max,
    const std::vector<std::vector<size_t>>& offsets_by_l_real,
    const std::vector<std::vector<size_t>>& offsets_by_l_complex,
    double* const gathered, double* const modal_buf) {
  constexpr bool zero_m_is_real = TensorType::rank() == 0;
  const auto& offsets_by_l =
      zero_m_is_real ? offsets_by_l_real : offsets_by_l_complex;
  for (size_t component = 0; component < tensor.size(); ++component) {
    const int spin_weight = ylm::TensorYlm::helpers::component_spin_weight<
        typename TensorType::structure>(component);
    accumulate_b3_tensor_component_sums(
        sum_sq_radial, counts_radial, sum_sq_angular, counts_angular,
        tensor[component].data(), n_r, n_r_max, spin_weight, offsets_by_l,
        gathered, modal_buf);
  }
}

// Normalize: result[i] = sqrt(sum_sq[i] / counts[i]), or 0 if counts[i] == 0.
void normalize_b3_power(const gsl::not_null<DataVector*> result,
                        const DataVector& sum_sq, const DataVector& counts) {
  result->destructive_resize(sum_sq.size());
  for (size_t i = 0; i < sum_sq.size(); ++i) {
    (*result)[i] = counts[i] == 0.0 ? 0.0 : sqrt(sum_sq[i] / counts[i]);
  }
}

}  // namespace

SwFilledSpherePowerMonitors sw_filled_sphere_power_monitors(
    const gsl::not_null<SwCartToSphereMatrix*> cart_to_sphere_matrix,
    const Variables<
        ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>& sw_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid) {
  if (mesh.basis(0) != Spectral::Basis::ZernikeB3 or
      mesh.quadrature(0) != Spectral::Quadrature::GaussRadauUpper or
      mesh.quadrature(1) != Spectral::Quadrature::Gauss or
      mesh.quadrature(2) != Spectral::Quadrature::Equiangular) {
    ERROR(
        "SW filled-sphere power monitors require a ZernikeB3 mesh with "
        "quadratures (GaussRadauUpper, Gauss, Equiangular), but the mesh is "
        << mesh);
  }
  const size_t n_r = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  const size_t m_max = (mesh.extents(2) - 1) / 2;
  if (l_max != m_max) {
    ERROR(
        "SW filled-sphere power monitors require l_max == m_max, but got "
        "l_max = "
        << l_max << " and m_max = " << m_max << ".");
  }
  ASSERT(n_r > 1,
         "At least 2 radial grid points required for B3 power monitors, got "
             << n_r);
  ASSERT(
      l_max >= 1,
      "At least l_max=1 required for B3 power monitors, got l_max=" << l_max);
  const size_t n_r_max = 2 * n_r - 2;
  ASSERT(l_max <= n_r_max,
         "ZernikeB3 constraint l_max <= 2*n_r-2 violated: l_max="
             << l_max << ", n_r=" << n_r);
  const auto& spherepack = ylm::get_spherepack_cache(l_max);

  // Precompute SPHEREPACK offsets grouped by l for both zero_m_is_real cases.
  // zero_m_is_real=true: scalars Psi and Pi (rank-0); false: co-vector Phi.
  std::vector<std::vector<size_t>> offsets_by_l_real(l_max + 1);
  for (ylm::SpherepackIterator it{l_max, l_max, 1, true}; it; ++it) {
    offsets_by_l_real[it.l()].push_back(it());
  }
  std::vector<std::vector<size_t>> offsets_by_l_complex(l_max + 1);
  for (ylm::SpherepackIterator it{l_max, l_max, 1, false}; it; ++it) {
    offsets_by_l_complex[it.l()].push_back(it());
  }

  // Single allocation split into two regions:
  //   [scratch region] gathered + modal_buf for gather/Jacobi NTM steps
  //   [accum region]   12 DataVector accumulator views (zero-initialised)
  //
  // Scratch layout: 2 * max_n_modes_l * n_r  (max_n_modes_l = 2*(l_max+1))
  // Accum layout (per SW variable, 3 variables total):
  //   sum_sq_radial | counts_radial | sum_sq_angular | counts_angular
  const size_t max_n_modes_l = 2 * (l_max + 1);
  const size_t n_angular = l_max + 1;
  const size_t buf_per_var = 2 * n_r + 2 * n_angular;
  const size_t scratch_size = 2 * max_n_modes_l * n_r;
  const size_t accum_size = 3 * buf_per_var;
  const auto buf =
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      cpp20::make_unique_for_overwrite<double[]>(scratch_size + accum_size);
  double* const gathered = buf.get();
  double* const modal_buf = gathered + max_n_modes_l * n_r;  // NOLINT
  double* const accum_start = buf.get() + scratch_size;      // NOLINT
  std::fill(accum_start, accum_start + accum_size, 0.0);     // NOLINT

  // Transform SW variables to TensorYlm spectral coefficients.
  // Frame-transform Phi to the grid frame; Psi and Pi are scalars (unchanged).
  namespace fd = ylm::TensorYlm::filter_detail;

  const size_t n_phys = mesh.number_of_grid_points();
  const size_t n_spec = n_r * spherepack.spectral_size();

  Variables<fd::sw_vars_list<Frame::Grid>> sw_grid_frame(n_phys);
  fd::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&sw_grid_frame), sw_vars,
                                    jac_inertial_to_grid);

  // SH analysis (nodal to modal) in the angular directions.
  Variables<fd::sw_vars_list<Frame::Grid>> sw_modal(n_spec);
  fd::nodal_to_modal_ylm(make_not_null(&sw_modal), sw_grid_frame, spherepack,
                         n_r);

  // Apply Cartesian-to-TensorYlm transform to Phi SH coefficients.
  fill_sw_cart_to_sphere_matrix(cart_to_sphere_matrix, l_max);
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
    for (size_t offset = 0; offset < n_r; ++offset) {
      cart_to_sphere_matrix->i->increment_multiply_on_right(
          make_not_null(&dest), offset, n_r, src, offset, n_r);
    }
  }

  // Set up accumulator DataVector views into the flat buffer.
  double* ap = accum_start;
  DataVector psi_sum_sq_radial{};
  psi_sum_sq_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector psi_counts_radial{};
  psi_counts_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector psi_sum_sq_angular{};
  psi_sum_sq_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT
  DataVector psi_counts_angular{};
  psi_counts_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT

  DataVector pi_sum_sq_radial{};
  pi_sum_sq_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector pi_counts_radial{};
  pi_counts_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector pi_sum_sq_angular{};
  pi_sum_sq_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT
  DataVector pi_counts_angular{};
  pi_counts_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT

  DataVector phi_sum_sq_radial{};
  phi_sum_sq_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector phi_counts_radial{};
  phi_counts_radial.set_data_ref(ap, n_r);
  ap += n_r;  // NOLINT
  DataVector phi_sum_sq_angular{};
  phi_sum_sq_angular.set_data_ref(ap, n_angular);
  ap += n_angular;  // NOLINT
  DataVector phi_counts_angular{};
  phi_counts_angular.set_data_ref(ap, n_angular);

  // Psi: scalar (rank-0), zero_m_is_real=true, spin_weight=0.
  accumulate_b3_tensor_sums(
      make_not_null(&psi_sum_sq_radial), make_not_null(&psi_counts_radial),
      make_not_null(&psi_sum_sq_angular), make_not_null(&psi_counts_angular),
      get<ScalarWave::Tags::Psi>(sw_modal), n_r, n_r_max, offsets_by_l_real,
      offsets_by_l_complex, gathered, modal_buf);

  // Pi: scalar (rank-0), zero_m_is_real=true, spin_weight=0.
  accumulate_b3_tensor_sums(
      make_not_null(&pi_sum_sq_radial), make_not_null(&pi_counts_radial),
      make_not_null(&pi_sum_sq_angular), make_not_null(&pi_counts_angular),
      get<ScalarWave::Tags::Pi>(sw_modal), n_r, n_r_max, offsets_by_l_real,
      offsets_by_l_complex, gathered, modal_buf);

  // Phi: co-vector (rank-1), zero_m_is_real=false, spin_weight from TensorYlm.
  accumulate_b3_tensor_sums(
      make_not_null(&phi_sum_sq_radial), make_not_null(&phi_counts_radial),
      make_not_null(&phi_sum_sq_angular), make_not_null(&phi_counts_angular),
      get<ScalarWave::Tags::Phi<3, Frame::Grid>>(phi_tensor_ylm_vars), n_r,
      n_r_max, offsets_by_l_real, offsets_by_l_complex, gathered, modal_buf);

  SwFilledSpherePowerMonitors result{};
  normalize_b3_power(make_not_null(&result.psi[radial_monitor_index]),
                     psi_sum_sq_radial, psi_counts_radial);
  normalize_b3_power(make_not_null(&result.psi[angular_monitor_index]),
                     psi_sum_sq_angular, psi_counts_angular);
  normalize_b3_power(make_not_null(&result.pi[radial_monitor_index]),
                     pi_sum_sq_radial, pi_counts_radial);
  normalize_b3_power(make_not_null(&result.pi[angular_monitor_index]),
                     pi_sum_sq_angular, pi_counts_angular);
  normalize_b3_power(make_not_null(&result.phi[radial_monitor_index]),
                     phi_sum_sq_radial, phi_counts_radial);
  normalize_b3_power(make_not_null(&result.phi[angular_monitor_index]),
                     phi_sum_sq_angular, phi_counts_angular);
  return result;
}

}  // namespace ScalarWave::power_monitor
