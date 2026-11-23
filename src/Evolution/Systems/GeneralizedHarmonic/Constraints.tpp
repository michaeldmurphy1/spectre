// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

// Device kernels that call these functions inline the functions they call
// too, so they need those definitions (see "Definitions that kernels call" in
// the GPU guide). Builds without Kokkos call their instantiations instead.
#ifdef SPECTRE_KOKKOS
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.tpp"
#endif  // SPECTRE_KOKKOS

namespace gh::Constraints_detail {
// Sets `result` to zero, with the number of grid points of `pattern`
template <typename T, typename Pattern>
KOKKOS_FUNCTION void set_to_zero(const gsl::not_null<T*> result,
                                 const Pattern& pattern) {
  set_number_of_grid_points(result, pattern);
  for (auto& component : *result) {
    component = 0.0;
  }
}

// Contractions of the variables that the terms of the two-index and F
// constraints (Eqs. (43) and (44) of https://arXiv.org/abs/gr-qc/0512093v3)
// share. With them each term is a few small contractions instead of nested
// loops over all its indices (up to 9216 iterations per component for a term
// of the F constraint).
template <typename DataType, size_t SpatialDim, typename Frame>
struct Contractions {
  // g^{cd} \partial_i \Phi_{jcd}
  tnsr::ij<DataType, SpatialDim, Frame> trace_d_phi{};
  // g^{cd} \partial_i \Pi_{cd}
  tnsr::i<DataType, SpatialDim, Frame> trace_d_pi{};
  // g^{cd} \Phi_{icd} and \gamma^{ij} g^{cd} \Phi_{jcd}
  tnsr::i<DataType, SpatialDim, Frame> trace_phi{};
  tnsr::I<DataType, SpatialDim, Frame> trace_phi_up{};
  // g^{cd} C_{icd} of the three-index constraint
  tnsr::i<DataType, SpatialDim, Frame> trace_three_index_constraint{};
  // n^c \Phi_{ica}, n^c n^d \Phi_{icd} and g^{ab} n^c \Phi_{icb}
  tnsr::ia<DataType, SpatialDim, Frame> phi_n{};
  tnsr::i<DataType, SpatialDim, Frame> phi_nn{};
  tnsr::iA<DataType, SpatialDim, Frame> phi_n_up{};
  // g^{ac} g^{bd} \Phi_{icd} and \Phi_{icd} g^{ce} g^{df} \Phi_{jef}
  tnsr::iAA<DataType, SpatialDim, Frame> phi_up{};
  tnsr::ii<DataType, SpatialDim, Frame> phi_phi{};
  // \gamma^{ij} \Phi_{jab}
  tnsr::Iaa<DataType, SpatialDim, Frame> phi_spatial_up{};
  // \Pi_{ab} n^b, g^{ab} \Pi_{ab} and g^{ac} g^{bd} \Pi_{cd}
  tnsr::a<DataType, SpatialDim, Frame> pi_n{};
  Scalar<DataType> trace_pi{};
  tnsr::AA<DataType, SpatialDim, Frame> pi_up{};
};

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void compute_contractions(
    const gsl::not_null<Contractions<DataType, SpatialDim, Frame>*> result,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint) {
  set_to_zero(make_not_null(&result->trace_d_phi), pi);
  set_to_zero(make_not_null(&result->trace_d_pi), pi);
  set_to_zero(make_not_null(&result->trace_phi), pi);
  set_to_zero(make_not_null(&result->trace_phi_up), pi);
  set_to_zero(make_not_null(&result->trace_three_index_constraint), pi);
  set_to_zero(make_not_null(&result->phi_n), pi);
  set_to_zero(make_not_null(&result->phi_nn), pi);
  set_to_zero(make_not_null(&result->phi_n_up), pi);
  set_to_zero(make_not_null(&result->phi_up), pi);
  set_to_zero(make_not_null(&result->phi_phi), pi);
  set_to_zero(make_not_null(&result->phi_spatial_up), pi);
  set_to_zero(make_not_null(&result->pi_n), pi);
  set_to_zero(make_not_null(&result->trace_pi), pi);
  set_to_zero(make_not_null(&result->pi_up), pi);

  for (size_t c = 0; c < SpatialDim + 1; ++c) {
    for (size_t d = 0; d < SpatialDim + 1; ++d) {
      get(result->trace_pi) +=
          inverse_spacetime_metric.get(c, d) * pi.get(c, d);
      result->pi_n.get(c) += pi.get(c, d) * spacetime_normal_vector.get(d);
      for (size_t i = 0; i < SpatialDim; ++i) {
        result->trace_d_pi.get(i) +=
            inverse_spacetime_metric.get(c, d) * d_pi.get(i, c, d);
        result->trace_phi.get(i) +=
            inverse_spacetime_metric.get(c, d) * phi.get(i, c, d);
        result->trace_three_index_constraint.get(i) +=
            inverse_spacetime_metric.get(c, d) *
            three_index_constraint.get(i, c, d);
        result->phi_n.get(i, d) +=
            spacetime_normal_vector.get(c) * phi.get(i, c, d);
        for (size_t j = 0; j < SpatialDim; ++j) {
          result->trace_d_phi.get(i, j) +=
              inverse_spacetime_metric.get(c, d) * d_phi.get(i, j, c, d);
        }
      }
    }
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      result->trace_phi_up.get(i) +=
          inverse_spatial_metric.get(i, j) * result->trace_phi.get(j);
    }
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      result->phi_nn.get(i) +=
          spacetime_normal_vector.get(a) * result->phi_n.get(i, a);
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        result->phi_n_up.get(i, b) +=
            inverse_spacetime_metric.get(b, a) * result->phi_n.get(i, a);
      }
    }
  }
  // The symmetric components only
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = a; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          result->pi_up.get(a, b) += inverse_spacetime_metric.get(a, c) *
                                     inverse_spacetime_metric.get(b, d) *
                                     pi.get(c, d);
          for (size_t i = 0; i < SpatialDim; ++i) {
            result->phi_up.get(i, a, b) += inverse_spacetime_metric.get(a, c) *
                                           inverse_spacetime_metric.get(b, d) *
                                           phi.get(i, c, d);
          }
        }
      }
      for (size_t k = 0; k < SpatialDim; ++k) {
        for (size_t j = 0; j < SpatialDim; ++j) {
          result->phi_spatial_up.get(k, a, b) +=
              inverse_spatial_metric.get(k, j) * phi.get(j, a, b);
        }
      }
    }
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          result->phi_phi.get(i, j) +=
              phi.get(i, c, d) * result->phi_up.get(j, c, d);
        }
      }
    }
  }
}
}  // namespace gh::Constraints_detail

namespace gh {
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  auto constraint =
      make_with_value<tnsr::iaa<DataType, SpatialDim, Frame>>(phi, 0.0);
  three_index_constraint<DataType, SpatialDim, Frame>(&constraint,
                                                      d_spacetime_metric, phi);
  return constraint;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void three_index_constraint(
    const gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  // Declare iterators for d_spacetime_metric and phi outside the for loop,
  // because they are const but constraint is not
  // clang-tidy: llvm-qualified-auto
  auto d_spacetime_metric_it = d_spacetime_metric.begin();  // NOLINT
  auto phi_it = phi.begin();                                // NOLINT

  for (auto constraint_it = (*constraint).begin();  // NOLINT
       constraint_it != (*constraint).end();
       ++constraint_it, (void)++d_spacetime_metric_it, (void)++phi_it) {
    // clang-tidy: cppcoreguidelines-pro-bounds-pointer-arithmetic
    *constraint_it = *d_spacetime_metric_it - *phi_it;  // NOLINT
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> gauge_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  auto constraint =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(pi, 0.0);
  gauge_constraint<DataType, SpatialDim, Frame>(
      &constraint, gauge_function, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi);
  return constraint;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void gauge_constraint(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  trace_christoffel(constraint, spacetime_normal_one_form,
                    spacetime_normal_vector, inverse_spatial_metric,
                    inverse_spacetime_metric, pi, phi);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    (*constraint).get(a) += gauge_function.get(a);
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ia<DataType, SpatialDim, Frame> two_index_constraint(
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint) {
  auto constraint =
      make_with_value<tnsr::ia<DataType, SpatialDim, Frame>>(pi, 0.0);
  two_index_constraint<DataType, SpatialDim, Frame>(
      &constraint, spacetime_d_gauge_function, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi, d_pi, d_phi, gamma2, three_index_constraint);
  return constraint;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint) {
  set_number_of_grid_points(constraint, gamma2);
  for (auto& component : *constraint) {
    component = 0.0;
  }
  Constraints_detail::Contractions<DataType, SpatialDim, Frame> sums{};
  Constraints_detail::compute_contractions(
      make_not_null(&sums), spacetime_normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi, d_pi, d_phi, three_index_constraint);

  // The terms of Eq. (44) of https://arXiv.org/abs/gr-qc/0512093v3 (numbered
  // as there) with a spacetime index `a` of a spatial tensor `Y_{ji}`, i.e.
  // `(a > 0) Y_{(a-1)i} + n_a n^j Y_{ji}`: terms 2 and 6
  tnsr::ij<DataType, SpatialDim, Frame> y{};
  Constraints_detail::set_to_zero(make_not_null(&y), pi);
  // The coefficient of `n_a`: terms 4, 7, 9, half of 11, and `n^j Y_{ji}`
  tnsr::i<DataType, SpatialDim, Frame> normal_coefficient{};
  Constraints_detail::set_to_zero(make_not_null(&normal_coefficient), pi);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      y.get(j, i) = 0.5 * (sums.phi_phi.get(j, i) - sums.trace_d_phi.get(j, i));
    }
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      normal_coefficient.get(i) +=
          spacetime_normal_vector.get(j + 1) * y.get(j, i) +
          0.5 * sums.phi_n.get(i, j + 1) * sums.trace_phi_up.get(j);
    }
    normal_coefficient.get(i) +=
        -0.5 * sums.trace_d_pi.get(i) +
        0.25 * get(sums.trace_pi) * sums.phi_nn.get(i) +
        0.5 * get(gamma2) * sums.trace_three_index_constraint.get(i);
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        normal_coefficient.get(i) +=
            0.5 * phi.get(i, c, d) * sums.pi_up.get(c, d);
      }
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // Terms 4, 7, 9, half of 11, and terms 2 and 6
      constraint->get(i, a) +=
          spacetime_normal_one_form.get(a) * normal_coefficient.get(i);
      if (a > 0) {
        constraint->get(i, a) += y.get(a - 1, i);
      }
      // Term 5
      constraint->get(i, a) += spacetime_d_gauge_function.get(i + 1, a);
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        // Terms 3, 10 and the other half of 11
        constraint->get(i, a) +=
            spacetime_normal_vector.get(b) *
                (d_pi.get(i, b, a) -
                 get(gamma2) * three_index_constraint.get(i, a, b)) -
            pi.get(b, a) *
                (sums.phi_n_up.get(i, b) +
                 0.5 * spacetime_normal_vector.get(b) * sums.phi_nn.get(i));
      }
      // Term 1
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          constraint->get(i, a) +=
              inverse_spatial_metric.get(j, k) * d_phi.get(j, i, k + 1, a);
        }
      }
    }
    // Term 8
    for (size_t k = 0; k < SpatialDim; ++k) {
      for (size_t m = 0; m < SpatialDim; ++m) {
        DataType phi_ikm =
            inverse_spatial_metric.get(m, 0) * phi.get(i, k + 1, 1);
        for (size_t n = 1; n < SpatialDim; ++n) {
          phi_ikm +=
              inverse_spatial_metric.get(m, n) * phi.get(i, k + 1, n + 1);
        }
        for (size_t a = 0; a < SpatialDim + 1; ++a) {
          constraint->get(i, a) -=
              sums.phi_spatial_up.get(k, m + 1, a) * phi_ikm;
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::iaa<DataType, SpatialDim, Frame> four_index_constraint(
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  static_assert(
      SpatialDim == 3,
      "four_index_constraint() currently only supports 3 spatial dimensions");
  auto constraint =
      make_with_value<tnsr::iaa<DataType, SpatialDim, Frame>>(d_phi, 0.0);
  four_index_constraint<DataType, SpatialDim, Frame>(&constraint, d_phi);
  return constraint;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void four_index_constraint(
    const gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  static_assert(
      SpatialDim == 3,
      "four_index_constraint() currently only supports 3 spatial dimensions");
  set_number_of_grid_points(constraint, d_phi);
  for (auto& component : *constraint) {
    component = 0.0;
  }

  // The permutations of (0, 1, 2) in lexicographic order, with their signs,
  // as `LeviCivitaIterator` (which can't be used on the device)
  const size_t permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                     {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  const int signs[6] = {1, -1, -1, 1, 1, -1};
  for (size_t p = 0; p < 6; ++p) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // Constraint is symmetric on a,b. Avoid double-counting in summation
      // by starting the inner loop at b = a.
      for (size_t b = a; b < SpatialDim + 1; ++b) {
        constraint->get(permutations[p][0], a, b) +=
            signs[p] * d_phi.get(permutations[p][1], permutations[p][2], a, b);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> f_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint) {
  auto constraint =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(pi, 0.0);
  f_constraint<DataType, SpatialDim, Frame>(
      &constraint, gauge_function, spacetime_d_gauge_function,
      spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi, d_pi, d_phi,
      gamma2, three_index_constraint);
  return constraint;
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> f_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::aa<DataType, SpatialDim, Frame>& trace_reversed_stress_energy) {
  auto constraint =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(pi, 0.0);
  f_constraint<DataType, SpatialDim, Frame>(
      &constraint, gauge_function, spacetime_d_gauge_function,
      spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi, d_pi, d_phi,
      gamma2, three_index_constraint, trace_reversed_stress_energy);
  return constraint;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::aa<DataType, SpatialDim, Frame>& trace_reversed_stress_energy) {
  f_constraint<DataType, SpatialDim, Frame>(
      constraint, gauge_function, spacetime_d_gauge_function,
      spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi, d_pi, d_phi,
      gamma2, three_index_constraint);
  Scalar<DataType> trace_stress_energy{};
  Constraints_detail::set_to_zero(make_not_null(&trace_stress_energy), pi);
  for (size_t b = 0; b < SpatialDim + 1; ++b) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      get(trace_stress_energy) += inverse_spacetime_metric.get(b, c) *
                                  trace_reversed_stress_energy.get(b, c);
    }
  }
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    constraint->get(a) += 8.0 * M_PI * spacetime_normal_one_form.get(a) *
                          get(trace_stress_energy);
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      constraint->get(a) -= 16.0 * M_PI * spacetime_normal_vector.get(b) *
                            trace_reversed_stress_energy.get(a, b);
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint) {
  set_number_of_grid_points(constraint, pi);
  for (auto& component : *constraint) {
    component = 0.0;
  }
  Constraints_detail::Contractions<DataType, SpatialDim, Frame> sums{};
  Constraints_detail::compute_contractions(
      make_not_null(&sums), spacetime_normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi, d_pi, d_phi, three_index_constraint);
  // g^{ab} H_b, \gamma^{ij} H_j, H_a n^a, \gamma^{ij} \Pi_{jb} n^b and
  // \gamma^{ij} n^c n^d \Phi_{jcd}
  tnsr::A<DataType, SpatialDim, Frame> gauge_up{};
  tnsr::I<DataType, SpatialDim, Frame> gauge_spatial_up{};
  Scalar<DataType> gauge_n{};
  tnsr::I<DataType, SpatialDim, Frame> pi_n_spatial_up{};
  tnsr::I<DataType, SpatialDim, Frame> phi_nn_up{};
  Constraints_detail::set_to_zero(make_not_null(&gauge_up), pi);
  Constraints_detail::set_to_zero(make_not_null(&gauge_spatial_up), pi);
  Constraints_detail::set_to_zero(make_not_null(&gauge_n), pi);
  Constraints_detail::set_to_zero(make_not_null(&pi_n_spatial_up), pi);
  Constraints_detail::set_to_zero(make_not_null(&phi_nn_up), pi);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    get(gauge_n) += gauge_function.get(a) * spacetime_normal_vector.get(a);
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      gauge_up.get(a) +=
          inverse_spacetime_metric.get(a, b) * gauge_function.get(b);
    }
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      gauge_spatial_up.get(i) +=
          inverse_spatial_metric.get(i, j) * gauge_function.get(j + 1);
      pi_n_spatial_up.get(i) +=
          inverse_spatial_metric.get(i, j) * sums.pi_n.get(j + 1);
      phi_nn_up.get(i) += inverse_spatial_metric.get(i, j) * sums.phi_nn.get(j);
    }
  }

  // The terms of Eq. (43) of https://arXiv.org/abs/gr-qc/0512093v3 (numbered
  // as there) with a spacetime index `a` of a spatial covector `X_i`, i.e.
  // `(a > 0) X_{a-1} + n_a n^i X_i`: terms 1, 6, 7, 8, 15, 17, 21 and half of
  // 22
  tnsr::i<DataType, SpatialDim, Frame> x{};
  Constraints_detail::set_to_zero(make_not_null(&x), pi);
  for (size_t i = 0; i < SpatialDim; ++i) {
    x.get(i) = 0.5 * sums.trace_d_pi.get(i) -
               0.25 * get(sums.trace_pi) * sums.phi_nn.get(i) -
               0.5 * get(gamma2) * sums.trace_three_index_constraint.get(i);
    for (size_t j = 0; j < SpatialDim; ++j) {
      x.get(i) -= 0.5 * sums.phi_n.get(i, j + 1) * sums.trace_phi_up.get(j);
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          x.get(i) += phi.get(i, j + 1, b) * inverse_spatial_metric.get(j, k) *
                      sums.phi_n_up.get(k, b);
        }
      }
    }
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      x.get(i) += sums.phi_n_up.get(i, b) * sums.pi_n.get(b) +
                  sums.phi_n.get(i, b) * gauge_up.get(b) -
                  spacetime_normal_vector.get(b) *
                      spacetime_d_gauge_function.get(i + 1, b);
    }
  }

  // The coefficient of `n_a`: terms 4, 5, 10, 11, 12, 16, 23, 24, 25 and
  // `n^i X_i`
  Scalar<DataType> normal_coefficient{};
  Constraints_detail::set_to_zero(make_not_null(&normal_coefficient), pi);
  get(normal_coefficient) = 0.5 * get(sums.trace_pi) * get(gauge_n);
  for (size_t i = 0; i < SpatialDim; ++i) {
    get(normal_coefficient) +=
        spacetime_normal_vector.get(i + 1) * x.get(i) +
        0.5 * gauge_spatial_up.get(i) * sums.trace_phi.get(i);
    for (size_t j = 0; j < SpatialDim; ++j) {
      get(normal_coefficient) += inverse_spatial_metric.get(i, j) *
                                 (0.5 * sums.trace_d_phi.get(i, j) +
                                  spacetime_d_gauge_function.get(i + 1, j + 1) -
                                  0.25 * sums.phi_phi.get(i, j));
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          get(normal_coefficient) -= 0.5 *
                                     sums.phi_spatial_up.get(i, j + 1, c) *
                                     inverse_spacetime_metric.get(c, d) *
                                     sums.phi_spatial_up.get(j, i + 1, d);
        }
      }
    }
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      get(normal_coefficient) -=
          sums.phi_spatial_up.get(i, i + 1, c) * gauge_up.get(c);
    }
  }
  for (size_t c = 0; c < SpatialDim + 1; ++c) {
    for (size_t d = 0; d < SpatialDim + 1; ++d) {
      get(normal_coefficient) += 0.25 * pi.get(c, d) * sums.pi_up.get(c, d) +
                                 0.5 * sums.pi_n.get(c) *
                                     inverse_spacetime_metric.get(c, d) *
                                     sums.pi_n.get(d);
    }
  }

  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    // Terms 1, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17, 21, 23, 24, 25 and half
    // of 22
    constraint->get(a) +=
        spacetime_normal_one_form.get(a) * get(normal_coefficient);
    if (a > 0) {
      constraint->get(a) += x.get(a - 1);
    }
    for (size_t i = 0; i < SpatialDim; ++i) {
      // Terms 13, 14, 19, 18 and 20
      constraint->get(a) -= (gauge_spatial_up.get(i) + pi_n_spatial_up.get(i) +
                             0.5 * phi_nn_up.get(i)) *
                                pi.get(i + 1, a) +
                            (pi_n_spatial_up.get(i) + gauge_spatial_up.get(i)) *
                                sums.phi_n.get(i, a);
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        // Term 9 and the other half of 22
        constraint->get(a) +=
            sums.phi_n_up.get(i, b) * sums.phi_spatial_up.get(i, b, a) +
            get(gamma2) *
                (inverse_spacetime_metric.get(i + 1, b) +
                 spacetime_normal_vector.get(i + 1) *
                     spacetime_normal_vector.get(b)) *
                three_index_constraint.get(i, b, a);
      }
      for (size_t j = 0; j < SpatialDim; ++j) {
        // Terms 2 and 3
        constraint->get(a) -=
            inverse_spatial_metric.get(i, j) * d_pi.get(i, j + 1, a);
        for (size_t b = 0; b < SpatialDim + 1; ++b) {
          constraint->get(a) -= inverse_spatial_metric.get(i, j) *
                                spacetime_normal_vector.get(b) *
                                d_phi.get(i, j, b, a);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> constraint_energy(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& f_constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& two_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& four_index_constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& spatial_metric_determinant,
    double gauge_constraint_multiplier, double two_index_constraint_multiplier,
    double three_index_constraint_multiplier,
    double four_index_constraint_multiplier) {
  auto energy = make_with_value<Scalar<DataType>>(gauge_constraint, 0.0);
  constraint_energy<DataType, SpatialDim, Frame>(
      &energy, gauge_constraint, f_constraint, two_index_constraint,
      three_index_constraint, four_index_constraint, inverse_spatial_metric,
      spatial_metric_determinant, gauge_constraint_multiplier,
      two_index_constraint_multiplier, three_index_constraint_multiplier,
      four_index_constraint_multiplier);
  return energy;
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void constraint_energy(
    const gsl::not_null<Scalar<DataType>*> energy,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& f_constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& two_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& four_index_constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& spatial_metric_determinant,
    double gauge_constraint_multiplier, double two_index_constraint_multiplier,
    double three_index_constraint_multiplier,
    double four_index_constraint_multiplier) {
  get(*energy) = gauge_constraint_multiplier * square(gauge_constraint.get(0)) +
                 two_index_constraint_multiplier * square(f_constraint.get(0));
  for (size_t a = 1; a < SpatialDim + 1; ++a) {
    get(*energy) +=
        gauge_constraint_multiplier * square(gauge_constraint.get(a)) +
        two_index_constraint_multiplier * square(f_constraint.get(a));
  }

  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        get(*energy) +=
            two_index_constraint_multiplier * two_index_constraint.get(i, a) *
            two_index_constraint.get(j, a) * inverse_spatial_metric.get(i, j);

        for (size_t b = 0; b < SpatialDim + 1; ++b) {
          get(*energy) += three_index_constraint_multiplier *
                              three_index_constraint.get(i, a, b) *
                              three_index_constraint.get(j, a, b) *
                              inverse_spatial_metric.get(i, j) +
                          2.0 * four_index_constraint_multiplier *
                              get(spatial_metric_determinant) *
                              four_index_constraint.get(i, a, b) *
                              four_index_constraint.get(j, a, b) *
                              inverse_spatial_metric.get(i, j);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> constraint_energy_normalization(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_spatial_metric_determinant,
    const double dimensional_constant) {
  Scalar<DataType> energy_norm{get_size(get(sqrt_spatial_metric_determinant))};
  constraint_energy_normalization<DataType, SpatialDim, Frame>(
      make_not_null(&energy_norm), d_spacetime_metric, d_pi, d_phi,
      inverse_spatial_metric, sqrt_spatial_metric_determinant,
      dimensional_constant);
  return energy_norm;
}

template <typename DataType, size_t SpatialDim, typename Frame>
void constraint_energy_normalization(
    const gsl::not_null<Scalar<DataType>*> energy_norm,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_spatial_metric_determinant,
    const double dimensional_constant) {
  const double square_dimensional_constant = square(dimensional_constant);
  get(*energy_norm) = 0.0;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        for (size_t j = 0; j < SpatialDim; ++j) {
          get(*energy_norm) += inverse_spatial_metric.get(i, j) *
                                   square_dimensional_constant *
                                   d_spacetime_metric.get(i, a, c) *
                                   d_spacetime_metric.get(j, a, c) +
                               inverse_spatial_metric.get(i, j) *
                                   d_pi.get(i, a, c) * d_pi.get(j, a, c);

          for (size_t k = 0; k < SpatialDim; ++k) {
            for (size_t l = 0; l < SpatialDim; ++l) {
              get(*energy_norm) += inverse_spatial_metric.get(i, j) *
                                   inverse_spatial_metric.get(k, l) *
                                   d_phi.get(i, k, a, c) *
                                   d_phi.get(j, l, a, c);
            }
          }
        }
      }
    }
  }
  get(*energy_norm) *= get(sqrt_spatial_metric_determinant);
}

template <typename DataType>
Scalar<DataType> normalized_constraint_energy(
    const Scalar<DataType>& constraint_energy,
    const Scalar<DataType>& constraint_energy_normalization) {
  auto normalized_energy =
      make_with_value<Scalar<DataType>>(constraint_energy, 0.0);
  normalized_constraint_energy<DataType>(make_not_null(&normalized_energy),
                                         constraint_energy,
                                         constraint_energy_normalization);
  return normalized_energy;
}

template <typename DataType>
void normalized_constraint_energy(
    const gsl::not_null<Scalar<DataType>*> normalized_energy,
    const Scalar<DataType>& constraint_energy,
    const Scalar<DataType>& constraint_energy_normalization) {
  get(*normalized_energy) =
      get(constraint_energy) / (1.e-10 + get(constraint_energy_normalization));
}

}  // namespace gh
