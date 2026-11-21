// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivativeKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace gh {
namespace {
constexpr size_t dim = TimeDerivativeOnDevice::dim;

// Calls `f(std::integral_constant<size_t, I>{})` for `I` in `[0, N)`, so that
// the tensor indices in `f` are compile-time constants
template <typename F, size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION constexpr void static_for_impl(
    const F& f, std::index_sequence<Is...> /*meta*/) {
  (f(std::integral_constant<size_t, Is>{}), ...);
}

template <size_t N, typename F>
KOKKOS_FORCEINLINE_FUNCTION constexpr void static_for(const F& f) {
  static_for_impl(f, std::make_index_sequence<N>{});
}

// The batches are indexed by grid point and element
struct PointIndex {
  size_t point;
  size_t element;
};

KOKKOS_FORCEINLINE_FUNCTION PointIndex
point_index(const size_t k, const size_t number_of_points) {
  return {.point = k % number_of_points, .element = k / number_of_points};
}

// The indices of the independent component `Component` of a symmetric
// spacetime tensor, (0, 0), (0, 1), ..., (dim, dim)
template <size_t Component>
constexpr std::array<size_t, 2> symmetric_indices() {
  size_t component = Component;
  for (size_t mu = 0; mu < dim + 1; ++mu) {
    if (component < dim + 1 - mu) {
      return {mu, mu + component};
    }
    component -= dim + 1 - mu;
  }
  return {0, 0};
}

// Component `Is...` of the batched tensor at `index`
template <size_t... Is, typename BatchTensor>
KOKKOS_FORCEINLINE_FUNCTION double& at(const BatchTensor& tensor,
                                       const PointIndex& index) {
  return get<Is...>(tensor)(index.point, index.element);
}

template <typename BatchScalar>
KOKKOS_FORCEINLINE_FUNCTION double& scalar_at(const BatchScalar& scalar,
                                              const PointIndex& index) {
  return get(scalar)(index.point, index.element);
}

// The inverse spacetime metric from the lapse, shift and inverse spatial
// metric, as `gr::inverse_spacetime_metric`
KOKKOS_FORCEINLINE_FUNCTION tnsr::AA<double, dim> inverse_spacetime_metric(
    const double lapse, const tnsr::I<double, dim>& shift,
    const tnsr::II<double, dim>& inverse_spatial_metric) {
  tnsr::AA<double, dim> result{};
  get<0, 0>(result) = -1.0 / (lapse * lapse);
  const double minus_one_over_lapse_sqrd = get<0, 0>(result);
  static_for<dim>([&](auto i_c) {
    constexpr size_t i = decltype(i_c)::value;
    get<0, i + 1>(result) = -get<i>(shift) * minus_one_over_lapse_sqrd;
  });
  static_for<dim>([&](auto i_c) {
    constexpr size_t i = decltype(i_c)::value;
    static_for<dim - i>([&](auto offset_c) {
      constexpr size_t j = i + decltype(offset_c)::value;
      get<i + 1, j + 1>(result) =
          get<i, j>(inverse_spatial_metric) +
          get<i>(shift) * get<j>(shift) * minus_one_over_lapse_sqrd;
    });
  });
  return result;
}

// The lapse, shift and inverse spatial metric from the cache
struct ThreePlusOne {
  double lapse;
  tnsr::I<double, dim> shift;
  tnsr::II<double, dim> inverse_spatial_metric;
};

template <typename Tag>
using Batch = TimeDerivativeOnDevice::Batch<Tag>;

KOKKOS_FORCEINLINE_FUNCTION ThreePlusOne read_three_plus_one(
    const Batch<gr::Tags::Lapse<DataVector>>& lapse,
    const Batch<gr::Tags::Shift<DataVector, dim>>& shift,
    const Batch<gr::Tags::InverseSpatialMetric<DataVector, dim>>&
        inverse_spatial_metric,
    const PointIndex& index) {
  ThreePlusOne result{};
  result.lapse = scalar_at(lapse, index);
  static_for<dim>([&](auto i_c) {
    constexpr size_t i = decltype(i_c)::value;
    get<i>(result.shift) = at<i>(shift, index);
    static_for<dim - i>([&](auto offset_c) {
      constexpr size_t j = i + decltype(offset_c)::value;
      get<i, j>(result.inverse_spatial_metric) =
          at<i, j>(inverse_spatial_metric, index);
    });
  });
  return result;
}

// The spacetime normal vector, as `gr::spacetime_normal_vector`
KOKKOS_FORCEINLINE_FUNCTION tnsr::A<double, dim> normal_vector(
    const double lapse, const tnsr::I<double, dim>& shift) {
  tnsr::A<double, dim> result{};
  get<0>(result) = 1. / lapse;
  static_for<dim>([&](auto i_c) {
    constexpr size_t i = decltype(i_c)::value;
    get<i + 1>(result) = -get<i>(shift) * get<0>(result);
  });
  return result;
}
// The arguments of the kernels
struct Arguments {
  using Class = TimeDerivativeOnDevice;
  Batch<::Tags::dt<Class::spacetime_metric_tag>> dt_spacetime_metric;
  Batch<::Tags::dt<Class::pi_tag>> dt_pi;
  Batch<::Tags::dt<Class::phi_tag>> dt_phi;
  Batch<Class::deriv<Class::spacetime_metric_tag>> d_spacetime_metric;
  Batch<Class::deriv<Class::pi_tag>> d_pi;
  Batch<Class::deriv<Class::phi_tag>> d_phi;
  Batch<Class::spacetime_metric_tag> spacetime_metric;
  Batch<Class::pi_tag> pi;
  Batch<Class::phi_tag> phi;
  Batch<Tags::ConstraintGamma0> gamma0;
  Batch<Tags::ConstraintGamma1> gamma1;
  Batch<Tags::ConstraintGamma2> gamma2;
  Batch<Class::MeshVelocity> mesh_velocity;
  // The cache
  Batch<gr::Tags::Lapse<DataVector>> lapse;
  Batch<gr::Tags::Shift<DataVector, dim>> shift;
  Batch<gr::Tags::InverseSpatialMetric<DataVector, dim>> inverse_spatial_metric;
  Batch<Class::ChristoffelContraction> christoffel_contraction;
  Batch<Tags::GaugeConstraint<DataVector, dim, Frame::Inertial>>
      gauge_constraint;
  size_t number_of_points;
};

// The mesh velocity at a point, or zero if the mesh doesn't move
template <bool HasMeshVelocity>
KOKKOS_FORCEINLINE_FUNCTION tnsr::I<double, dim> read_mesh_velocity(
    const Arguments& args, const PointIndex& index) {
  tnsr::I<double, dim> result{};
  if constexpr (HasMeshVelocity) {
    static_for<dim>([&](auto i_c) {
      constexpr size_t i = decltype(i_c)::value;
      get<i>(result) = at<i>(args.mesh_velocity, index);
    });
  }
  return result;
}

// 1. The lapse, shift and inverse spatial metric, as
// `determinant_and_inverse`, `gr::shift` and `gr::lapse`
KOKKOS_FORCEINLINE_FUNCTION ThreePlusOne
three_plus_one(const Arguments& args, const PointIndex& index) {
  const auto& spacetime_metric = args.spacetime_metric;
  const double t00 = at<1, 1>(spacetime_metric, index);
  const double t01 = at<1, 2>(spacetime_metric, index);
  const double t02 = at<1, 3>(spacetime_metric, index);
  const double t11 = at<2, 2>(spacetime_metric, index);
  const double t12 = at<2, 3>(spacetime_metric, index);
  const double t22 = at<3, 3>(spacetime_metric, index);
  const double a = t11 * t22 - t12 * t12;
  const double b = t12 * t02 - t01 * t22;
  const double c = t01 * t12 - t11 * t02;
  const double det = t00 * a + t01 * b + t02 * c;
  const double one_over_det = 1.0 / det;
  ThreePlusOne result{};
  auto& inverse = result.inverse_spatial_metric;
  get<0, 0>(inverse) = (t11 * t22 - t12 * t12) * one_over_det;
  get<0, 1>(inverse) = (t12 * t02 - t22 * t01) * one_over_det;
  get<0, 2>(inverse) = (t01 * t12 - t02 * t11) * one_over_det;
  get<1, 1>(inverse) = (t22 * t00 - t02 * t02) * one_over_det;
  get<1, 2>(inverse) = (t02 * t01 - t00 * t12) * one_over_det;
  get<2, 2>(inverse) = (t00 * t11 - t01 * t01) * one_over_det;
  auto& shift = result.shift;
  static_for<dim>([&](auto i_c) {
    constexpr size_t i = decltype(i_c)::value;
    get<i>(shift) = get<i, 0>(inverse) * at<1, 0>(spacetime_metric, index);
    static_for<dim - 1>([&](auto j_c) {
      constexpr size_t j = decltype(j_c)::value + 1;
      get<i>(shift) +=
          get<i, j>(inverse) * at<j + 1, 0>(spacetime_metric, index);
    });
  });
  double lapse = -at<0, 0>(spacetime_metric, index);
  static_for<dim>([&](auto i_c) {
    constexpr size_t i = decltype(i_c)::value;
    lapse += get<i>(shift) * at<i + 1, 0>(spacetime_metric, index);
  });
  result.lapse = sqrt(lapse);
  return result;
}

// 2. The time derivative of the spacetime metric
template <bool HasMeshVelocity>
KOKKOS_FORCEINLINE_FUNCTION void dt_spacetime_metric(
    const Arguments& args, const ThreePlusOne& three_plus_one,
    const PointIndex& index) {
  const auto& pi = args.pi;
  const auto& phi = args.phi;
  const auto& d_spacetime_metric = args.d_spacetime_metric;
  const double lapse = three_plus_one.lapse;
  const auto& shift = three_plus_one.shift;
  const double gamma1_value = scalar_at(args.gamma1, index);
  const double gamma1_plus_1 = 1.0 + gamma1_value;
  const auto velocity = read_mesh_velocity<HasMeshVelocity>(args, index);
  static_for<dim + 1>([&](auto mu_c) {
    constexpr size_t mu = decltype(mu_c)::value;
    static_for<dim + 1 - mu>([&](auto offset_c) {
      constexpr size_t nu = mu + decltype(offset_c)::value;
      double result = -lapse * at<mu, nu>(pi, index);
      static_for<dim>([&](auto m_c) {
        constexpr size_t m = decltype(m_c)::value;
        result += get<m>(shift) * at<m, mu, nu>(phi, index);
      });
      const auto constraint = [&](auto m_c) -> double {
        constexpr size_t m = decltype(m_c)::value;
        return at<m, mu, nu>(d_spacetime_metric, index) -
               at<m, mu, nu>(phi, index);
      };
      double shift_dot_constraint =
          get<0>(shift) * constraint(std::integral_constant<size_t, 0>{});
      static_for<dim - 1>([&](auto m_c) {
        constexpr size_t m = decltype(m_c)::value + 1;
        shift_dot_constraint +=
            get<m>(shift) * constraint(std::integral_constant<size_t, m>{});
      });
      result += gamma1_plus_1 * shift_dot_constraint;
      if constexpr (HasMeshVelocity) {
        double mesh_velocity_dot_constraint =
            get<0>(velocity) * constraint(std::integral_constant<size_t, 0>{});
        static_for<dim - 1>([&](auto m_c) {
          constexpr size_t m = decltype(m_c)::value + 1;
          mesh_velocity_dot_constraint +=
              get<m>(velocity) *
              constraint(std::integral_constant<size_t, m>{});
        });
        result += gamma1_value * mesh_velocity_dot_constraint;
        static_for<dim>([&](auto m_c) {
          constexpr size_t m = decltype(m_c)::value;
          result += get<m>(velocity) * at<m, mu, nu>(d_spacetime_metric, index);
        });
      }
      at<mu, nu>(args.dt_spacetime_metric, index) = result;
    });
  });
}

// The Christoffel symbols of the first kind with the last index raised,
// \Gamma_{mu nu}^alpha for all nu and alpha (as `gr::christoffel_first_kind`
// and `gh::TimeDerivative`), from \partial_c g_{ab} and the inverse metric
template <size_t Mu, typename DMetric>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<double, (dim + 1) * (dim + 1)>
christoffel_row(const DMetric& d_metric,
                const tnsr::AA<double, dim>& inverse_metric) {
  constexpr size_t mu = Mu;
  const auto mu_c = std::integral_constant<size_t, mu>{};
  Kokkos::Array<double, (dim + 1) * (dim + 1)> result{};
  static_for<dim + 1>([&](auto nu_c) {
    constexpr size_t nu = decltype(nu_c)::value;
    // \Gamma_{mu nu beta}
    tnsr::a<double, dim> christoffel{};
    static_for<dim + 1>([&](auto beta_c) {
      constexpr size_t beta = decltype(beta_c)::value;
      get<beta>(christoffel) =
          0.5 * (d_metric(nu_c, beta_c, mu_c) + d_metric(beta_c, nu_c, mu_c) -
                 d_metric(mu_c, nu_c, beta_c));
    });
    static_for<dim + 1>([&](auto alpha_c) {
      constexpr size_t alpha = decltype(alpha_c)::value;
      double raised = get<alpha, 0>(inverse_metric) * get<0>(christoffel);
      static_for<dim>([&](auto beta_c) {
        constexpr size_t beta = decltype(beta_c)::value + 1;
        raised += get<alpha, beta>(inverse_metric) * get<beta>(christoffel);
      });
      result[nu * (dim + 1) + alpha] = raised;
    });
  });
  return result;
}

// 3. The gauge constraint C_mu = \Gamma_{mu nu}^nu and the contraction
// \Gamma_{mu alpha}^delta \Gamma_{nu delta}^alpha of the Christoffel symbols,
// the only parts of the Christoffel symbols that d_t Pi uses. Each row of the
// Christoffel symbols is recomputed where it is needed, so that only two rows
// are held at a time.
KOKKOS_FORCEINLINE_FUNCTION void christoffel_contractions(
    const Arguments& args, const ThreePlusOne& three_plus_one,
    const PointIndex& index) {
  const auto& pi = args.pi;
  const auto& phi = args.phi;
  const auto inverse_metric =
      inverse_spacetime_metric(three_plus_one.lapse, three_plus_one.shift,
                               three_plus_one.inverse_spatial_metric);
  // The time derivative of the spacetime metric without the constraint terms,
  // i.e. the time component of the spacetime derivative
  tnsr::aa<double, dim> dt_metric{};
  static_for<dim + 1>([&](auto a_c) {
    constexpr size_t a = decltype(a_c)::value;
    static_for<dim + 1 - a>([&](auto offset_c) {
      constexpr size_t b = a + decltype(offset_c)::value;
      get<a, b>(dt_metric) = -three_plus_one.lapse * at<a, b>(pi, index);
      static_for<dim>([&](auto m_c) {
        constexpr size_t m = decltype(m_c)::value;
        get<a, b>(dt_metric) +=
            get<m>(three_plus_one.shift) * at<m, a, b>(phi, index);
      });
    });
  });
  // \partial_c g_{ab}
  const auto d_metric = [&](auto derivative_index, auto first_index,
                            auto second_index) -> double {
    constexpr size_t c = decltype(derivative_index)::value;
    constexpr size_t a = decltype(first_index)::value;
    constexpr size_t b = decltype(second_index)::value;
    if constexpr (c == 0) {
      return get<a, b>(dt_metric);
    } else {
      return at<c - 1, a, b>(phi, index);
    }
  };
  static_for<dim + 1>([&](auto mu_c) {
    constexpr size_t mu = decltype(mu_c)::value;
    const auto row_mu = christoffel_row<mu>(d_metric, inverse_metric);
    double trace = row_mu[0];
    static_for<dim>([&](auto nu_c) {
      constexpr size_t nu = decltype(nu_c)::value + 1;
      trace += row_mu[nu * (dim + 1) + nu];
    });
    at<mu>(args.gauge_constraint, index) = trace;
    const auto contraction = [&](const auto& row_nu) {
      double result = 0.0;
      static_for<dim + 1>([&](auto delta_c) {
        constexpr size_t delta = decltype(delta_c)::value;
        static_for<dim + 1>([&](auto alpha_c) {
          constexpr size_t alpha = decltype(alpha_c)::value;
          result += row_mu[alpha * (dim + 1) + delta] *
                    row_nu[delta * (dim + 1) + alpha];
        });
      });
      return result;
    };
    at<mu, mu>(args.christoffel_contraction, index) = contraction(row_mu);
    static_for<dim - mu>([&](auto offset_c) {
      constexpr size_t nu = mu + 1 + decltype(offset_c)::value;
      at<mu, nu>(args.christoffel_contraction, index) =
          contraction(christoffel_row<nu>(d_metric, inverse_metric));
    });
  });
}

// Steps 1 to 3, with one thread per grid point. It has its own name, so that
// profilers tell it apart.
template <bool HasMeshVelocity>
struct MetricKernel {
  Arguments args;
  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const PointIndex index = point_index(k, args.number_of_points);
    const ThreePlusOne values = three_plus_one(args, index);
    scalar_at(args.lapse, index) = values.lapse;
    static_for<dim>([&](auto i_c) {
      constexpr size_t i = decltype(i_c)::value;
      at<i>(args.shift, index) = get<i>(values.shift);
      static_for<dim - i>([&](auto offset_c) {
        constexpr size_t j = i + decltype(offset_c)::value;
        at<i, j>(args.inverse_spatial_metric, index) =
            get<i, j>(values.inverse_spatial_metric);
      });
    });
    dt_spacetime_metric<HasMeshVelocity>(args, values, index);
    christoffel_contractions(args, values, index);
  }
};

// The values at a grid point that all components of d_t Pi use
struct DtPiPointValues {
  ThreePlusOne three_plus_one;
  tnsr::AA<double, dim> inverse_metric;
  tnsr::a<double, dim> gauge_constraint;
  double gamma0;
  double gamma1gamma2;
  tnsr::a<double, dim> pi_one_normal;
  double half_pi_two_normals;
  double normal_dot_gauge_constraint;
  tnsr::I<double, dim> mesh_velocity;
};

template <bool HasMeshVelocity>
KOKKOS_FORCEINLINE_FUNCTION DtPiPointValues
dt_pi_point_values(const Arguments& args, const PointIndex& index) {
  DtPiPointValues values{};
  values.three_plus_one = read_three_plus_one(
      args.lapse, args.shift, args.inverse_spatial_metric, index);
  const double lapse = values.three_plus_one.lapse;
  const auto& shift = values.three_plus_one.shift;
  values.inverse_metric = inverse_spacetime_metric(
      lapse, shift, values.three_plus_one.inverse_spatial_metric);
  const auto normal = normal_vector(lapse, shift);
  static_for<dim + 1>([&](auto a_c) {
    constexpr size_t a = decltype(a_c)::value;
    get<a>(values.gauge_constraint) = at<a>(args.gauge_constraint, index);
  });
  values.gamma0 = scalar_at(args.gamma0, index);
  values.mesh_velocity = read_mesh_velocity<HasMeshVelocity>(args, index);
  values.gamma1gamma2 =
      scalar_at(args.gamma1, index) * scalar_at(args.gamma2, index);

  // n^a Pi_{ab}
  static_for<dim + 1>([&](auto mu_c) {
    constexpr size_t mu = decltype(mu_c)::value;
    get<mu>(values.pi_one_normal) = get<0>(normal) * at<0, mu>(args.pi, index);
    static_for<dim>([&](auto nu_c) {
      constexpr size_t nu = decltype(nu_c)::value + 1;
      get<mu>(values.pi_one_normal) +=
          get<nu>(normal) * at<nu, mu>(args.pi, index);
    });
  });
  values.half_pi_two_normals = get<0>(normal) * get<0>(values.pi_one_normal);
  static_for<dim>([&](auto mu_c) {
    constexpr size_t mu = decltype(mu_c)::value + 1;
    values.half_pi_two_normals +=
        get<mu>(normal) * get<mu>(values.pi_one_normal);
  });
  values.half_pi_two_normals *= 0.5;
  values.normal_dot_gauge_constraint =
      get<0>(normal) * get<0>(values.gauge_constraint);
  static_for<dim>([&](auto mu_c) {
    constexpr size_t mu = decltype(mu_c)::value + 1;
    values.normal_dot_gauge_constraint +=
        get<mu>(normal) * get<mu>(values.gauge_constraint);
  });
  // Rescaled by gamma0, as on the host
  values.normal_dot_gauge_constraint *= values.gamma0;
  return values;
}

// The component `(Mu, Nu)` of d_t Pi, with the operations in the same order as
// on the host, except that the contraction of the Christoffel symbols is
// summed separately
template <bool HasMeshVelocity, size_t Mu, size_t Nu>
KOKKOS_FORCEINLINE_FUNCTION void dt_pi_component(const Arguments& args,
                                                 const DtPiPointValues& values,
                                                 const PointIndex& index) {
  constexpr size_t mu = Mu;
  constexpr size_t nu = Nu;
  const double lapse = values.three_plus_one.lapse;
  const auto& shift = values.three_plus_one.shift;
  const auto& inverse_spatial_metric =
      values.three_plus_one.inverse_spatial_metric;
  const auto& inverse_metric = values.inverse_metric;
  const auto& pi = args.pi;
  const auto& phi = args.phi;

  // The contractions, recomputed where they are used
  // Pi_{nu beta} g^{delta beta}
  const auto pi_2_up = [&](auto delta_c) -> double {
    constexpr size_t delta = decltype(delta_c)::value;
    double result = get<delta, 0>(inverse_metric) * at<nu, 0>(pi, index);
    static_for<dim>([&](auto beta_c) {
      constexpr size_t beta = decltype(beta_c)::value + 1;
      result += get<delta, beta>(inverse_metric) * at<nu, beta>(pi, index);
    });
    return result;
  };
  // g^{m n} Phi_{n a b}
  const auto phi_1_up = [&](auto derivative, auto row, auto column) -> double {
    constexpr size_t m = decltype(derivative)::value;
    constexpr size_t a = decltype(row)::value;
    constexpr size_t b = decltype(column)::value;
    double result = get<m, 0>(inverse_spatial_metric) * at<0, a, b>(phi, index);
    static_for<dim - 1>([&](auto n_c) {
      constexpr size_t n = decltype(n_c)::value + 1;
      result += get<m, n>(inverse_spatial_metric) * at<n, a, b>(phi, index);
    });
    return result;
  };
  // g^{alpha beta} Phi_{m nu beta}
  const auto phi_3_up = [&](auto m_c, auto alpha_c) -> double {
    constexpr size_t m = decltype(m_c)::value;
    constexpr size_t alpha = decltype(alpha_c)::value;
    double result = get<alpha, 0>(inverse_metric) * at<m, nu, 0>(phi, index);
    static_for<dim>([&](auto beta_c) {
      constexpr size_t beta = decltype(beta_c)::value + 1;
      result += get<alpha, beta>(inverse_metric) * at<m, nu, beta>(phi, index);
    });
    return result;
  };
  const auto mu_c = std::integral_constant<size_t, mu>{};
  const auto nu_c = std::integral_constant<size_t, nu>{};

  // The terms proportional to n_a
  const double metric = at<mu, nu>(args.spacetime_metric, index);
  double result = 0.0;
  if constexpr (mu == 0) {
    const double minus_gamma0_lapse = -values.gamma0 * lapse;
    if constexpr (nu == 0) {
      result = 2.0 * minus_gamma0_lapse * get<0>(values.gauge_constraint) -
               values.normal_dot_gauge_constraint * metric;
    } else {
      result = minus_gamma0_lapse * get<nu>(values.gauge_constraint) -
               values.normal_dot_gauge_constraint * metric;
    }
  } else {
    result = -values.normal_dot_gauge_constraint * metric;
  }

  result -= values.half_pi_two_normals * at<mu, nu>(pi, index);
  static_for<dim + 1>([&](auto delta_c) {
    constexpr size_t delta = decltype(delta_c)::value;
    result -= 2 * at<mu, delta>(pi, index) * pi_2_up(delta_c);
    static_for<dim>([&](auto n_c) {
      result += 2 * phi_1_up(n_c, mu_c, delta_c) * phi_3_up(n_c, delta_c);
    });
  });
  result -= 2. * at<mu, nu>(args.christoffel_contraction, index);
  static_for<dim>([&](auto m_c) {
    constexpr size_t m = decltype(m_c)::value;
    result -= get<m + 1>(values.pi_one_normal) * phi_1_up(m_c, mu_c, nu_c);
    static_for<dim>([&](auto n_c) {
      constexpr size_t n = decltype(n_c)::value;
      result -= get<m, n>(inverse_spatial_metric) *
                at<m, n, mu, nu>(args.d_phi, index);
    });
  });
  result *= lapse;

  const auto constraint = [&](auto m_c) -> double {
    constexpr size_t m = decltype(m_c)::value;
    return at<m, mu, nu>(args.d_spacetime_metric, index) -
           at<m, mu, nu>(phi, index);
  };
  double shift_dot_constraint =
      get<0>(shift) * constraint(std::integral_constant<size_t, 0>{});
  static_for<dim - 1>([&](auto m_c) {
    constexpr size_t m = decltype(m_c)::value + 1;
    shift_dot_constraint +=
        get<m>(shift) * constraint(std::integral_constant<size_t, m>{});
  });
  result += values.gamma1gamma2 * shift_dot_constraint;
  if constexpr (HasMeshVelocity) {
    double mesh_velocity_dot_constraint =
        get<0>(values.mesh_velocity) *
        constraint(std::integral_constant<size_t, 0>{});
    static_for<dim - 1>([&](auto m_c) {
      constexpr size_t m = decltype(m_c)::value + 1;
      mesh_velocity_dot_constraint +=
          get<m>(values.mesh_velocity) *
          constraint(std::integral_constant<size_t, m>{});
    });
    result += values.gamma1gamma2 * mesh_velocity_dot_constraint;
  }
  static_for<dim>([&](auto m_c) {
    constexpr size_t m = decltype(m_c)::value;
    result += get<m>(shift) * at<m, mu, nu>(args.d_pi, index);
  });
  if constexpr (HasMeshVelocity) {
    static_for<dim>([&](auto m_c) {
      constexpr size_t m = decltype(m_c)::value;
      result += get<m>(values.mesh_velocity) * at<m, mu, nu>(args.d_pi, index);
    });
  }
  at<mu, nu>(args.dt_pi, index) = result;
}

// The independent components `Components...` (see `symmetric_indices()`) at a
// grid point
template <bool HasMeshVelocity, size_t... Components>
KOKKOS_FORCEINLINE_FUNCTION void dt_pi_components(
    const Arguments& args, const PointIndex& index,
    std::index_sequence<Components...> /*meta*/) {
  const DtPiPointValues values =
      dt_pi_point_values<HasMeshVelocity>(args, index);
  (dt_pi_component<HasMeshVelocity, symmetric_indices<Components>()[0],
                   symmetric_indices<Components>()[1]>(args, values, index),
   ...);
}

// 4. The time derivative of Pi, with one thread per grid point that computes
// all components
template <bool HasMeshVelocity>
struct DtPiKernel {
  Arguments args;
  KOKKOS_FUNCTION void operator()(const size_t k) const {
    dt_pi_components<HasMeshVelocity>(
        args, point_index(k, args.number_of_points),
        std::make_index_sequence<(dim + 1) * (dim + 2) / 2>{});
  }
};

// 5. The time derivative of Phi, with one thread per grid point
template <bool HasMeshVelocity>
struct DtPhiKernel {
  Arguments args;
  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const PointIndex index = point_index(k, args.number_of_points);
    const auto& pi = args.pi;
    const auto& phi = args.phi;
    const auto& d_pi = args.d_pi;
    const auto& d_phi = args.d_phi;
    const auto& d_spacetime_metric = args.d_spacetime_metric;
    const ThreePlusOne three_plus_one = read_three_plus_one(
        args.lapse, args.shift, args.inverse_spatial_metric, index);
    const double lapse = three_plus_one.lapse;
    const auto& shift = three_plus_one.shift;
    const auto& inverse_spatial_metric = three_plus_one.inverse_spatial_metric;
    const auto normal = normal_vector(lapse, shift);
    const double gamma2_value = scalar_at(args.gamma2, index);
    const auto velocity = read_mesh_velocity<HasMeshVelocity>(args, index);
    static_for<dim>([&](auto i_c) {
      constexpr size_t i = decltype(i_c)::value;
      // n^a Phi_{i a b}
      tnsr::a<double, dim> phi_one_normal{};
      static_for<dim + 1>([&](auto nu_c) {
        constexpr size_t nu = decltype(nu_c)::value;
        get<nu>(phi_one_normal) = get<0>(normal) * at<i, 0, nu>(phi, index);
        static_for<dim>([&](auto mu_c) {
          constexpr size_t mu = decltype(mu_c)::value + 1;
          get<nu>(phi_one_normal) +=
              get<mu>(normal) * at<i, mu, nu>(phi, index);
        });
      });
      double half_phi_two_normals = get<0>(normal) * get<0>(phi_one_normal);
      static_for<dim>([&](auto mu_c) {
        constexpr size_t mu = decltype(mu_c)::value + 1;
        half_phi_two_normals += get<mu>(normal) * get<mu>(phi_one_normal);
      });
      half_phi_two_normals *= 0.5;

      static_for<dim + 1>([&](auto mu_c) {
        constexpr size_t mu = decltype(mu_c)::value;
        static_for<dim + 1 - mu>([&](auto offset_c) {
          constexpr size_t nu = mu + decltype(offset_c)::value;
          double result =
              at<mu, nu>(pi, index) * half_phi_two_normals -
              at<i, mu, nu>(d_pi, index) +
              gamma2_value * (at<i, mu, nu>(d_spacetime_metric, index) -
                              at<i, mu, nu>(phi, index));
          static_for<dim>([&](auto n_c) {
            constexpr size_t n = decltype(n_c)::value;
            double phi_1_up =
                get<n, 0>(inverse_spatial_metric) * at<0, mu, nu>(phi, index);
            static_for<dim - 1>([&](auto m_c) {
              constexpr size_t m = decltype(m_c)::value + 1;
              phi_1_up +=
                  get<n, m>(inverse_spatial_metric) * at<m, mu, nu>(phi, index);
            });
            result += get<n + 1>(phi_one_normal) * phi_1_up;
          });
          result *= lapse;
          static_for<dim>([&](auto m_c) {
            constexpr size_t m = decltype(m_c)::value;
            result += get<m>(shift) * at<m, i, mu, nu>(d_phi, index);
          });
          if constexpr (HasMeshVelocity) {
            static_for<dim>([&](auto m_c) {
              constexpr size_t m = decltype(m_c)::value;
              result += get<m>(velocity) * at<m, i, mu, nu>(d_phi, index);
            });
          }
          at<i, mu, nu>(args.dt_phi, index) = result;
        });
      });
    });
  }
};

template <bool HasMeshVelocity>
void launch_kernels(const Kokkos::DefaultExecutionSpace& exec,
                    const Arguments& args, const size_t total) {
  Kokkos::parallel_for("gh::TimeDerivativeOnDevice::metric",
                       Kokkos::RangePolicy<>(exec, 0, total),
                       MetricKernel<HasMeshVelocity>{.args = args});
  Kokkos::parallel_for("gh::TimeDerivativeOnDevice::dt_pi",
                       Kokkos::RangePolicy<>(exec, 0, total),
                       DtPiKernel<HasMeshVelocity>{.args = args});
  Kokkos::parallel_for("gh::TimeDerivativeOnDevice::dt_phi",
                       Kokkos::RangePolicy<>(exec, 0, total),
                       DtPhiKernel<HasMeshVelocity>{.args = args});
}
}  // namespace

void TimeDerivativeOnDevice::apply(
    const Kokkos::DefaultExecutionSpace& exec,
    const Batch<::Tags::dt<spacetime_metric_tag>>& dt_spacetime_metric,
    const Batch<::Tags::dt<pi_tag>>& dt_pi,
    const Batch<::Tags::dt<phi_tag>>& dt_phi,
    const Batch<deriv<spacetime_metric_tag>>& d_spacetime_metric,
    const Batch<deriv<pi_tag>>& d_pi, const Batch<deriv<phi_tag>>& d_phi,
    const Batch<spacetime_metric_tag>& spacetime_metric,
    const Batch<pi_tag>& pi, const Batch<phi_tag>& phi,
    const Batch<Tags::ConstraintGamma0>& gamma0,
    const Batch<Tags::ConstraintGamma1>& gamma1,
    const Batch<Tags::ConstraintGamma2>& gamma2,
    const std::optional<Batch<MeshVelocity>>& mesh_velocity) {
  const Kokkos::Profiling::ScopedRegion region{
      "gh::TimeDerivativeOnDevice::apply"};
  const size_t number_of_points = get<0, 0>(spacetime_metric).extent(0);
  const size_t number_of_elements = get<0, 0>(spacetime_metric).extent(1);
  if (cache_.number_of_grid_points() != number_of_points or
      cache_.number_of_elements() != number_of_elements) {
    cache_ = BatchedVariables<cache_tags>{exec, number_of_points,
                                          number_of_elements};
  }
  const Arguments args{
      .dt_spacetime_metric = dt_spacetime_metric,
      .dt_pi = dt_pi,
      .dt_phi = dt_phi,
      .d_spacetime_metric = d_spacetime_metric,
      .d_pi = d_pi,
      .d_phi = d_phi,
      .spacetime_metric = spacetime_metric,
      .pi = pi,
      .phi = phi,
      .gamma0 = gamma0,
      .gamma1 = gamma1,
      .gamma2 = gamma2,
      .mesh_velocity = mesh_velocity.value_or(Batch<MeshVelocity>{}),
      .lapse = get<::Tags::BatchView<gr::Tags::Lapse<DataVector>>>(cache_),
      .shift = get<::Tags::BatchView<gr::Tags::Shift<DataVector, dim>>>(cache_),
      .inverse_spatial_metric = get<
          ::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, dim>>>(
          cache_),
      .christoffel_contraction =
          get<::Tags::BatchView<ChristoffelContraction>>(cache_),
      .gauge_constraint = get<::Tags::BatchView<
          Tags::GaugeConstraint<DataVector, dim, Frame::Inertial>>>(cache_),
      .number_of_points = number_of_points};
  const size_t total = number_of_points * number_of_elements;
  if (mesh_velocity.has_value()) {
    launch_kernels<true>(exec, args, total);
  } else {
    launch_kernels<false>(exec, args, total);
  }
}
}  // namespace gh
#endif  // SPECTRE_KOKKOS
