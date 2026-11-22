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
// Functions to compute generalized-harmonic 2-index constraint, where
// function arguments are in the order that each quantity first
// appears in the corresponding term in Eq. (44) of
// https://arXiv.org/abs/gr-qc/0512093v3
template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_1_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          constraint->get(i, a) +=
              inverse_spatial_metric.get(j, k) * d_phi.get(j, i, k + 1, a);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_2_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          if (a > 0) {
            constraint->get(i, a) -= 0.5 * inverse_spacetime_metric.get(c, d) *
                                     d_phi.get(a - 1, i, c, d);
          }
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(i, a) -= 0.5 * spacetime_normal_vector.get(j + 1) *
                                     spacetime_normal_one_form.get(a) *
                                     inverse_spacetime_metric.get(c, d) *
                                     d_phi.get(j, i, c, d);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_3_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        constraint->get(i, a) +=
            spacetime_normal_vector.get(b) * d_pi.get(i, b, a);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_4_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          constraint->get(i, a) -= 0.5 * spacetime_normal_one_form.get(a) *
                                   inverse_spacetime_metric.get(c, d) *
                                   d_pi.get(i, c, d);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_5_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      constraint->get(i, a) += spacetime_d_gauge_function.get(i + 1, a);
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_6_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            for (size_t f = 0; f < SpatialDim + 1; ++f) {
              if (a > 0) {
                constraint->get(i, a) += 0.5 * phi.get(a - 1, c, d) *
                                         phi.get(i, e, f) *
                                         inverse_spacetime_metric.get(c, e) *
                                         inverse_spacetime_metric.get(d, f);
              }
              for (size_t j = 0; j < SpatialDim; ++j) {
                constraint->get(i, a) +=
                    0.5 * spacetime_normal_vector.get(j + 1) *
                    spacetime_normal_one_form.get(a) * phi.get(j, c, d) *
                    phi.get(i, e, f) * inverse_spacetime_metric.get(c, e) *
                    inverse_spacetime_metric.get(d, f);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_7_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          for (size_t c = 0; c < SpatialDim + 1; ++c) {
            for (size_t d = 0; d < SpatialDim + 1; ++d) {
              for (size_t e = 0; e < SpatialDim + 1; ++e) {
                constraint->get(i, a) +=
                    0.5 * inverse_spatial_metric.get(j, k) * phi.get(j, c, d) *
                    phi.get(i, k + 1, e) * inverse_spacetime_metric.get(c, d) *
                    spacetime_normal_vector.get(e) *
                    spacetime_normal_one_form.get(a);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_8_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          for (size_t m = 0; m < SpatialDim; ++m) {
            for (size_t n = 0; n < SpatialDim; ++n) {
              constraint->get(i, a) -= inverse_spatial_metric.get(j, k) *
                                       inverse_spatial_metric.get(m, n) *
                                       phi.get(j, m + 1, a) *
                                       phi.get(i, k + 1, n + 1);
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_9_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        for (size_t c = 0; c < SpatialDim + 1; ++c) {
          for (size_t d = 0; d < SpatialDim + 1; ++d) {
            for (size_t e = 0; e < SpatialDim + 1; ++e) {
              constraint->get(i, a) +=
                  0.5 * phi.get(i, c, d) * pi.get(b, e) *
                  spacetime_normal_one_form.get(a) *
                  (inverse_spacetime_metric.get(c, b) *
                       inverse_spacetime_metric.get(d, e) +
                   0.5 * inverse_spacetime_metric.get(b, e) *
                       spacetime_normal_vector.get(c) *
                       spacetime_normal_vector.get(d));
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_10_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        for (size_t c = 0; c < SpatialDim + 1; ++c) {
          for (size_t d = 0; d < SpatialDim + 1; ++d) {
            constraint->get(i, a) -= phi.get(i, c, d) * pi.get(b, a) *
                                     spacetime_normal_vector.get(c) *
                                     (inverse_spacetime_metric.get(b, d) +
                                      0.5 * spacetime_normal_vector.get(b) *
                                          spacetime_normal_vector.get(d));
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void two_index_constraint_add_term_11_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const Scalar<DataType>& gamma2,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint) {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t c = 0; c < SpatialDim + 1; ++c) {
          constraint->get(i, a) += 0.5 * get(gamma2) *
                                   spacetime_normal_one_form.get(a) *
                                   inverse_spacetime_metric.get(c, d) *
                                   three_index_constraint.get(i, c, d);
        }
        constraint->get(i, a) -= get(gamma2) * spacetime_normal_vector.get(d) *
                                 three_index_constraint.get(i, a, d);
      }
    }
  }
}

// Functions to compute generalized-harmonic F constraint, where
// function arguments are in the order that each quantity first
// appears in the corresponding term in Eq. (43) of
// https://arXiv.org/abs/gr-qc/0512093v3
template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_1_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        if (a > 0) {
          constraint->get(a) +=
              0.5 * inverse_spacetime_metric.get(b, c) * d_pi.get(a - 1, b, c);
        }
        for (size_t i = 0; i < SpatialDim; ++i) {
          constraint->get(a) += 0.5 * spacetime_normal_vector.get(i + 1) *
                                spacetime_normal_one_form.get(a) *
                                inverse_spacetime_metric.get(b, c) *
                                d_pi.get(i, b, c);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_2_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        constraint->get(a) -=
            inverse_spatial_metric.get(i, j) * d_pi.get(i, j + 1, a);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_3_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
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
KOKKOS_FUNCTION void f_constraint_add_term_4_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) += 0.5 * spacetime_normal_one_form.get(a) *
                                  inverse_spacetime_metric.get(b, c) *
                                  inverse_spatial_metric.get(i, j) *
                                  d_phi.get(i, j, b, c);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_5_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        constraint->get(a) += spacetime_normal_one_form.get(a) *
                              inverse_spatial_metric.get(i, j) *
                              spacetime_d_gauge_function.get(i + 1, j + 1);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_6_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            for (size_t k = 0; k < SpatialDim; ++k) {
              if (a > 0) {
                constraint->get(a) += phi.get(a - 1, j + 1, b) *
                                      inverse_spatial_metric.get(j, k) *
                                      phi.get(k, c, d) *
                                      inverse_spacetime_metric.get(b, d) *
                                      spacetime_normal_vector.get(c);
              }
              for (size_t i = 0; i < SpatialDim; ++i) {
                constraint->get(a) +=
                    spacetime_normal_one_form.get(a) *
                    spacetime_normal_vector.get(i + 1) * phi.get(i, j + 1, b) *
                    inverse_spatial_metric.get(j, k) * phi.get(k, c, d) *
                    inverse_spacetime_metric.get(b, d) *
                    spacetime_normal_vector.get(c);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_7_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            for (size_t k = 0; k < SpatialDim; ++k) {
              if (a > 0) {
                constraint->get(a) -= 0.5 * phi.get(a - 1, j + 1, b) *
                                      inverse_spatial_metric.get(j, k) *
                                      phi.get(k, c, d) *
                                      inverse_spacetime_metric.get(c, d) *
                                      spacetime_normal_vector.get(b);
              }
              for (size_t i = 0; i < SpatialDim; ++i) {
                constraint->get(a) -=
                    0.5 * spacetime_normal_one_form.get(a) *
                    spacetime_normal_vector.get(i + 1) * phi.get(i, j + 1, b) *
                    inverse_spatial_metric.get(j, k) * phi.get(k, c, d) *
                    inverse_spacetime_metric.get(c, d) *
                    spacetime_normal_vector.get(b);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_8_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::ab<DataType, SpatialDim, Frame>& spacetime_d_gauge_function) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      if (a > 0) {
        constraint->get(a) -= spacetime_normal_vector.get(b) *
                              spacetime_d_gauge_function.get(a, b);
      }
      for (size_t i = 0; i < SpatialDim; ++i) {
        constraint->get(a) -= spacetime_normal_one_form.get(a) *
                              spacetime_normal_vector.get(i + 1) *
                              spacetime_normal_vector.get(b) *
                              spacetime_d_gauge_function.get(i + 1, b);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_9_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t i = 0; i < SpatialDim; ++i) {
            for (size_t j = 0; j < SpatialDim; ++j) {
              constraint->get(a) += inverse_spatial_metric.get(i, j) *
                                    phi.get(i, c, d) * phi.get(j, b, a) *
                                    inverse_spacetime_metric.get(b, c) *
                                    spacetime_normal_vector.get(d);
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_10_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            for (size_t m = 0; m < SpatialDim; ++m) {
              for (size_t n = 0; n < SpatialDim; ++n) {
                constraint->get(a) -=
                    0.5 * spacetime_normal_one_form.get(a) *
                    inverse_spatial_metric.get(i, j) *
                    inverse_spatial_metric.get(m, n) * phi.get(i, m + 1, c) *
                    phi.get(n, j + 1, d) * inverse_spacetime_metric.get(c, d);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_11_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            for (size_t i = 0; i < SpatialDim; ++i) {
              for (size_t j = 0; j < SpatialDim; ++j) {
                constraint->get(a) -= 0.25 * spacetime_normal_one_form.get(a) *
                                      inverse_spatial_metric.get(i, j) *
                                      phi.get(i, c, d) * phi.get(j, b, e) *
                                      inverse_spacetime_metric.get(c, b) *
                                      inverse_spacetime_metric.get(d, e);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_12_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            constraint->get(a) += 0.25 * spacetime_normal_one_form.get(a) *
                                  pi.get(c, d) * pi.get(b, e) *
                                  inverse_spacetime_metric.get(c, b) *
                                  inverse_spacetime_metric.get(d, e);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_13_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        constraint->get(a) -= inverse_spatial_metric.get(i, j) *
                              gauge_function.get(i + 1) * pi.get(j + 1, a);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_14_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        for (size_t j = 0; j < SpatialDim; ++j) {
          constraint->get(a) -= spacetime_normal_vector.get(b) *
                                inverse_spatial_metric.get(i, j) *
                                pi.get(b, i + 1) * pi.get(j + 1, a);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_15_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            if (a > 0) {
              constraint->get(a) -=
                  0.25 * phi.get(a - 1, c, d) * spacetime_normal_vector.get(c) *
                  spacetime_normal_vector.get(d) * pi.get(b, e) *
                  inverse_spacetime_metric.get(b, e);
            }
            for (size_t i = 0; i < SpatialDim; ++i) {
              constraint->get(a) -=
                  0.25 * spacetime_normal_one_form.get(a) *
                  spacetime_normal_vector.get(i + 1) * phi.get(i, c, d) *
                  spacetime_normal_vector.get(c) *
                  spacetime_normal_vector.get(d) * pi.get(b, e) *
                  inverse_spacetime_metric.get(b, e);
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_16_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            constraint->get(a) +=
                0.5 * spacetime_normal_one_form.get(a) * pi.get(c, d) *
                pi.get(b, e) * inverse_spacetime_metric.get(c, e) *
                spacetime_normal_vector.get(d) * spacetime_normal_vector.get(b);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_17_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            if (a > 0) {
              constraint->get(a) += phi.get(a - 1, c, d) * pi.get(b, e) *
                                    spacetime_normal_vector.get(c) *
                                    spacetime_normal_vector.get(b) *
                                    inverse_spacetime_metric.get(d, e);
            }
            for (size_t i = 0; i < SpatialDim; ++i) {
              constraint->get(a) += spacetime_normal_one_form.get(a) *
                                    spacetime_normal_vector.get(i + 1) *
                                    phi.get(i, c, d) * pi.get(b, e) *
                                    spacetime_normal_vector.get(c) *
                                    spacetime_normal_vector.get(b) *
                                    inverse_spacetime_metric.get(d, e);
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_18_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t e = 0; e < SpatialDim + 1; ++e) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) -=
                inverse_spatial_metric.get(i, j) * phi.get(i, b, a) *
                spacetime_normal_vector.get(b) * pi.get(j + 1, e) *
                spacetime_normal_vector.get(e);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_19_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) -=
                0.5 * inverse_spatial_metric.get(i, j) * phi.get(i, c, d) *
                spacetime_normal_vector.get(c) *
                spacetime_normal_vector.get(d) * pi.get(j + 1, a);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_20_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        for (size_t j = 0; j < SpatialDim; ++j) {
          constraint->get(a) -= inverse_spatial_metric.get(i, j) *
                                gauge_function.get(i + 1) * phi.get(j, b, a) *
                                spacetime_normal_vector.get(b);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_21_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          if (a > 0) {
            constraint->get(a) += phi.get(a - 1, c, d) * gauge_function.get(b) *
                                  inverse_spacetime_metric.get(b, c) *
                                  spacetime_normal_vector.get(d);
          }
          for (size_t i = 0; i < SpatialDim; ++i) {
            constraint->get(a) += spacetime_normal_one_form.get(a) *
                                  spacetime_normal_vector.get(i + 1) *
                                  phi.get(i, c, d) * gauge_function.get(b) *
                                  inverse_spacetime_metric.get(b, c) *
                                  spacetime_normal_vector.get(d);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_22_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const Scalar<DataType>& gamma2,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t d = 0; d < SpatialDim + 1; ++d) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        constraint->get(a) += get(gamma2) *
                              inverse_spacetime_metric.get(i + 1, d) *
                              three_index_constraint.get(i, d, a);
        constraint->get(a) += get(gamma2) * spacetime_normal_vector.get(i + 1) *
                              spacetime_normal_vector.get(d) *
                              three_index_constraint.get(i, d, a);
      }

      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        if (a > 0) {
          constraint->get(a) -= 0.5 * get(gamma2) *
                                inverse_spacetime_metric.get(c, d) *
                                three_index_constraint.get(a - 1, c, d);
        }
        for (size_t i = 0; i < SpatialDim; ++i) {
          constraint->get(a) -= 0.5 * get(gamma2) *
                                spacetime_normal_one_form.get(a) *
                                spacetime_normal_vector.get(i + 1) *
                                inverse_spacetime_metric.get(c, d) *
                                three_index_constraint.get(i, c, d);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_23_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          constraint->get(a) +=
              0.5 * spacetime_normal_one_form.get(a) * pi.get(c, d) *
              inverse_spacetime_metric.get(c, d) * gauge_function.get(b) *
              spacetime_normal_vector.get(b);
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_24_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) -= spacetime_normal_one_form.get(a) *
                                  inverse_spatial_metric.get(i, j) *
                                  phi.get(i, j + 1, c) * gauge_function.get(d) *
                                  inverse_spacetime_metric.get(c, d);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_term_25_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric) {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) += 0.5 * spacetime_normal_one_form.get(a) *
                                  inverse_spatial_metric.get(i, j) *
                                  gauge_function.get(i + 1) * phi.get(j, c, d) *
                                  inverse_spacetime_metric.get(c, d);
          }
        }
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
KOKKOS_FUNCTION void f_constraint_add_stress_energy_term(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::aa<DataType, SpatialDim, Frame>& trace_reversed_stress_energy) {
  // This term, like many terms in the f constraint, may benefit from
  // allocating a temporary for the trace. However, once we apply that
  // optimization it should be applied to all terms that can benefit from
  // temporary storage, and the allocation shared among all of the terms.
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      constraint->get(a) -= 16.0 * M_PI * spacetime_normal_vector.get(b) *
                            trace_reversed_stress_energy.get(a, b);
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        constraint->get(a) += 8.0 * M_PI * spacetime_normal_one_form.get(a) *
                              inverse_spacetime_metric.get(b, c) *
                              trace_reversed_stress_energy.get(b, c);
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

  Constraints_detail::two_index_constraint_add_term_1_of_11(
      constraint, inverse_spatial_metric, d_phi);
  Constraints_detail::two_index_constraint_add_term_2_of_11(
      constraint, spacetime_normal_vector, spacetime_normal_one_form,
      inverse_spacetime_metric, d_phi);
  Constraints_detail::two_index_constraint_add_term_3_of_11(
      constraint, spacetime_normal_vector, d_pi);
  Constraints_detail::two_index_constraint_add_term_4_of_11(
      constraint, spacetime_normal_one_form, inverse_spacetime_metric, d_pi);
  Constraints_detail::two_index_constraint_add_term_5_of_11(
      constraint, spacetime_d_gauge_function);
  Constraints_detail::two_index_constraint_add_term_6_of_11(
      constraint, spacetime_normal_vector, spacetime_normal_one_form, phi,
      inverse_spacetime_metric);
  Constraints_detail::two_index_constraint_add_term_7_of_11(
      constraint, inverse_spatial_metric, phi, inverse_spacetime_metric,
      spacetime_normal_vector, spacetime_normal_one_form);
  Constraints_detail::two_index_constraint_add_term_8_of_11(
      constraint, inverse_spatial_metric, phi);
  Constraints_detail::two_index_constraint_add_term_9_of_11(
      constraint, phi, pi, spacetime_normal_one_form, inverse_spacetime_metric,
      spacetime_normal_vector);
  Constraints_detail::two_index_constraint_add_term_10_of_11(
      constraint, phi, pi, spacetime_normal_vector, inverse_spacetime_metric);
  Constraints_detail::two_index_constraint_add_term_11_of_11(
      constraint, gamma2, spacetime_normal_one_form, inverse_spacetime_metric,
      spacetime_normal_vector, three_index_constraint);
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
  Constraints_detail::f_constraint_add_stress_energy_term(
      constraint, inverse_spacetime_metric, spacetime_normal_vector,
      spacetime_normal_one_form, trace_reversed_stress_energy);
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

  Constraints_detail::f_constraint_add_term_1_of_25(
      constraint, spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spacetime_metric, d_pi);
  Constraints_detail::f_constraint_add_term_2_of_25(
      constraint, inverse_spatial_metric, d_pi);
  Constraints_detail::f_constraint_add_term_3_of_25(
      constraint, inverse_spatial_metric, spacetime_normal_vector, d_phi);
  Constraints_detail::f_constraint_add_term_4_of_25(
      constraint, spacetime_normal_one_form, inverse_spacetime_metric,
      inverse_spatial_metric, d_phi);
  Constraints_detail::f_constraint_add_term_5_of_25(
      constraint, spacetime_normal_one_form, inverse_spatial_metric,
      spacetime_d_gauge_function);
  Constraints_detail::f_constraint_add_term_6_of_25(
      constraint, spacetime_normal_one_form, spacetime_normal_vector, phi,
      inverse_spatial_metric, inverse_spacetime_metric);
  Constraints_detail::f_constraint_add_term_7_of_25(
      constraint, spacetime_normal_one_form, spacetime_normal_vector, phi,
      inverse_spatial_metric, inverse_spacetime_metric);
  Constraints_detail::f_constraint_add_term_8_of_25(
      constraint, spacetime_normal_one_form, spacetime_normal_vector,
      spacetime_d_gauge_function);
  Constraints_detail::f_constraint_add_term_9_of_25(
      constraint, inverse_spatial_metric, phi, inverse_spacetime_metric,
      spacetime_normal_vector);
  Constraints_detail::f_constraint_add_term_10_of_25(
      constraint, spacetime_normal_one_form, inverse_spatial_metric, phi,
      inverse_spacetime_metric);
  Constraints_detail::f_constraint_add_term_11_of_25(
      constraint, spacetime_normal_one_form, inverse_spatial_metric, phi,
      inverse_spacetime_metric);
  Constraints_detail::f_constraint_add_term_12_of_25(
      constraint, spacetime_normal_one_form, pi, inverse_spacetime_metric);
  Constraints_detail::f_constraint_add_term_13_of_25(
      constraint, inverse_spatial_metric, gauge_function, pi);
  Constraints_detail::f_constraint_add_term_14_of_25(
      constraint, spacetime_normal_vector, inverse_spatial_metric, pi);
  Constraints_detail::f_constraint_add_term_15_of_25(
      constraint, inverse_spacetime_metric, spacetime_normal_one_form,
      spacetime_normal_vector, phi, pi);
  Constraints_detail::f_constraint_add_term_16_of_25(
      constraint, spacetime_normal_one_form, pi, inverse_spacetime_metric,
      spacetime_normal_vector);
  Constraints_detail::f_constraint_add_term_17_of_25(
      constraint, inverse_spacetime_metric, spacetime_normal_one_form,
      spacetime_normal_vector, phi, pi);
  Constraints_detail::f_constraint_add_term_18_of_25(
      constraint, inverse_spatial_metric, phi, spacetime_normal_vector, pi);
  Constraints_detail::f_constraint_add_term_19_of_25(
      constraint, inverse_spatial_metric, phi, spacetime_normal_vector, pi);
  Constraints_detail::f_constraint_add_term_20_of_25(
      constraint, inverse_spatial_metric, gauge_function, phi,
      spacetime_normal_vector);
  Constraints_detail::f_constraint_add_term_21_of_25(
      constraint, spacetime_normal_one_form, spacetime_normal_vector, phi,
      gauge_function, inverse_spacetime_metric);
  Constraints_detail::f_constraint_add_term_22_of_25(
      constraint, gamma2, inverse_spacetime_metric, spacetime_normal_vector,
      three_index_constraint, spacetime_normal_one_form);
  Constraints_detail::f_constraint_add_term_23_of_25(
      constraint, spacetime_normal_one_form, pi, inverse_spacetime_metric,
      gauge_function, spacetime_normal_vector);
  Constraints_detail::f_constraint_add_term_24_of_25(
      constraint, spacetime_normal_one_form, inverse_spatial_metric, phi,
      gauge_function, inverse_spacetime_metric);
  Constraints_detail::f_constraint_add_term_25_of_25(
      constraint, spacetime_normal_one_form, inverse_spatial_metric,
      gauge_function, phi, inverse_spacetime_metric);
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
