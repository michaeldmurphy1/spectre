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

// The threads of kernels in which each thread computes one of
// `number_of_components` components at a grid point. The consecutive warps of
// threads compute the components at the same `points_per_group` grid points,
// so that the values they all read are cached.
constexpr size_t points_per_group = 32;

KOKKOS_FORCEINLINE_FUNCTION size_t number_of_component_threads(
    const size_t number_of_points, const size_t number_of_components) {
  return (number_of_points + points_per_group - 1) / points_per_group *
         points_per_group * number_of_components;
}

struct ComponentAtPoint {
  size_t point;
  size_t component;
};

KOKKOS_FORCEINLINE_FUNCTION ComponentAtPoint
component_at_point(const size_t k, const size_t number_of_components) {
  const size_t group_and_component = k / points_per_group;
  return {
      .point = (group_and_component / number_of_components) * points_per_group +
               k % points_per_group,
      .component = group_and_component % number_of_components};
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
// The time derivative of Pi, as `gh::TimeDerivative` with the harmonic gauge.
// The kernels of the layouts (see `TimeDerivativeOnDevice::DtPiLayout`) compute
// some of its components at a grid point.
struct DtPiArguments {
  using Class = TimeDerivativeOnDevice;
  Batch<::Tags::dt<Class::pi_tag>> dt_pi;
  Batch<Class::deriv<Class::spacetime_metric_tag>> d_spacetime_metric;
  Batch<Class::deriv<Class::pi_tag>> d_pi;
  Batch<Class::deriv<Class::phi_tag>> d_phi;
  Batch<Class::spacetime_metric_tag> spacetime_metric;
  Batch<Class::pi_tag> pi;
  Batch<Class::phi_tag> phi;
  Batch<Tags::ConstraintGamma0> gamma0;
  Batch<Tags::ConstraintGamma1> gamma1;
  Batch<Tags::ConstraintGamma2> gamma2;
  bool has_mesh_velocity;
  Batch<Class::MeshVelocity> mesh_velocity;
  Batch<gr::Tags::Lapse<DataVector>> lapse;
  Batch<gr::Tags::Shift<DataVector, dim>> shift;
  Batch<gr::Tags::InverseSpatialMetric<DataVector, dim>> inverse_spatial_metric;
  Batch<::Tags::TempTensor<0, tnsr::abC<DataVector, dim>>> christoffel_3_up;
  Batch<Tags::GaugeConstraint<DataVector, dim, Frame::Inertial>>
      gauge_constraint;
};

// The values at a grid point that all components use
struct DtPiPointValues {
  ThreePlusOne three_plus_one;
  tnsr::AA<double, dim> inverse_metric;
  tnsr::a<double, dim> gauge_constraint;
  double gamma0;
  double gamma1gamma2;
  tnsr::a<double, dim> pi_one_normal;
  double half_pi_two_normals;
  double normal_dot_gauge_constraint;
};

KOKKOS_FORCEINLINE_FUNCTION DtPiPointValues
dt_pi_point_values(const DtPiArguments& args, const PointIndex& index) {
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

// The component `(Mu, Nu)`, with the operations in the same order as on the
// host
template <size_t Mu, size_t Nu>
KOKKOS_FORCEINLINE_FUNCTION void dt_pi_component(const DtPiArguments& args,
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
  const auto phi_1_up = [&](auto m_c, auto a_c, auto b_c) -> double {
    constexpr size_t m = decltype(m_c)::value;
    constexpr size_t a = decltype(a_c)::value;
    constexpr size_t b = decltype(b_c)::value;
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
    static_for<dim + 1>([&](auto alpha_c) {
      constexpr size_t alpha = decltype(alpha_c)::value;
      result -= 2. * at<mu, alpha, delta>(args.christoffel_3_up, index) *
                at<nu, delta, alpha>(args.christoffel_3_up, index);
    });
  });
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
  if (args.has_mesh_velocity) {
    double mesh_velocity_dot_constraint =
        at<0>(args.mesh_velocity, index) *
        constraint(std::integral_constant<size_t, 0>{});
    static_for<dim - 1>([&](auto m_c) {
      constexpr size_t m = decltype(m_c)::value + 1;
      mesh_velocity_dot_constraint +=
          at<m>(args.mesh_velocity, index) *
          constraint(std::integral_constant<size_t, m>{});
    });
    result += values.gamma1gamma2 * mesh_velocity_dot_constraint;
  }
  static_for<dim>([&](auto m_c) {
    constexpr size_t m = decltype(m_c)::value;
    result += get<m>(shift) * at<m, mu, nu>(args.d_pi, index);
  });
  at<mu, nu>(args.dt_pi, index) = result;
}

// The independent components `Components...` (see `symmetric_indices()`) at a
// grid point
template <size_t... Components>
KOKKOS_FORCEINLINE_FUNCTION void dt_pi_components(
    const DtPiArguments& args, const PointIndex& index,
    std::index_sequence<Components...> /*meta*/) {
  const DtPiPointValues values = dt_pi_point_values(args, index);
  (dt_pi_component<symmetric_indices<Components>()[0],
                   symmetric_indices<Components>()[1]>(args, values, index),
   ...);
}

// The kernels of the layouts, with one thread per grid point that computes
// the components `Components...`. Each layout has its own name, so that
// profilers tell them apart.
template <size_t... Components>
struct DtPiComponents {
  DtPiArguments args;
  size_t number_of_points;
  KOKKOS_FUNCTION void operator()(const size_t k) const {
    dt_pi_components(args, point_index(k, number_of_points),
                     std::index_sequence<Components...>{});
  }
};
template <size_t... Components>
struct DtPiPointKernel : DtPiComponents<Components...> {};
template <size_t... Components>
struct DtPiRowKernel : DtPiComponents<Components...> {};
template <size_t... Components>
struct DtPiComponentKernel : DtPiComponents<Components...> {};

// The kernel of `DtPiLayout::WarpComponent`
struct DtPiWarpComponentKernel {
  static constexpr size_t number_of_components = (dim + 1) * (dim + 2) / 2;
  DtPiArguments args;
  size_t number_of_points;
  size_t total;
  KOKKOS_FUNCTION void operator()(const size_t k) const {
    const ComponentAtPoint thread = component_at_point(k, number_of_components);
    if (thread.point >= total) {
      return;
    }
    const PointIndex index = point_index(thread.point, number_of_points);
    static_for<number_of_components>([&](auto component_c) {
      constexpr size_t component = decltype(component_c)::value;
      if (component == thread.component) {
        dt_pi_components(args, index, std::index_sequence<component>{});
      }
    });
  }
};

template <template <size_t...> typename Kernel, size_t... Components>
void launch_dt_pi(const Kokkos::DefaultExecutionSpace& exec,
                  const DtPiArguments& args, const size_t number_of_points,
                  const size_t total,
                  std::index_sequence<Components...> /*meta*/) {
  Kokkos::parallel_for("gh::TimeDerivativeOnDevice::dt_pi",
                       Kokkos::RangePolicy<>(exec, 0, total),
                       Kernel<Components...>{{args, number_of_points}});
}

template <size_t Offset, size_t... Is>
constexpr std::index_sequence<Offset + Is...> offset_sequence(
    std::index_sequence<Is...> /*meta*/) {
  return {};
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
  const size_t total = number_of_points * number_of_elements;
  constexpr size_t number_of_symmetric_components = (dim + 1) * (dim + 2) / 2;
  if (cache_.number_of_grid_points() != number_of_points or
      cache_.number_of_elements() != number_of_elements) {
    cache_ = BatchedVariables<cache_tags>{exec, number_of_points,
                                          number_of_elements};
  }
  const auto lapse_batch =
      get<::Tags::BatchView<gr::Tags::Lapse<DataVector>>>(cache_);
  const auto shift_batch =
      get<::Tags::BatchView<gr::Tags::Shift<DataVector, dim>>>(cache_);
  const auto inverse_spatial_metric_batch =
      get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, dim>>>(
          cache_);
  const auto christoffel_3_up_batch =
      get<::Tags::BatchView<::Tags::TempTensor<0, tnsr::abC<DataVector, dim>>>>(
          cache_);
  const auto gauge_constraint_batch = get<::Tags::BatchView<
      Tags::GaugeConstraint<DataVector, dim, Frame::Inertial>>>(cache_);

  // 1. The lapse, shift and inverse spatial metric, as
  // `determinant_and_inverse`, `gr::shift` and `gr::lapse`
  Kokkos::parallel_for(
      "gh::TimeDerivativeOnDevice::three_plus_one",
      Kokkos::RangePolicy<>(exec, 0, total), KOKKOS_LAMBDA(const size_t k) {
        const PointIndex index = point_index(k, number_of_points);
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
        tnsr::II<double, dim> inverse{};
        get<0, 0>(inverse) = (t11 * t22 - t12 * t12) * one_over_det;
        get<0, 1>(inverse) = (t12 * t02 - t22 * t01) * one_over_det;
        get<0, 2>(inverse) = (t01 * t12 - t02 * t11) * one_over_det;
        get<1, 1>(inverse) = (t22 * t00 - t02 * t02) * one_over_det;
        get<1, 2>(inverse) = (t02 * t01 - t00 * t12) * one_over_det;
        get<2, 2>(inverse) = (t00 * t11 - t01 * t01) * one_over_det;
        tnsr::I<double, dim> shift{};
        static_for<dim>([&](auto i_c) {
          constexpr size_t i = decltype(i_c)::value;
          get<i>(shift) =
              get<i, 0>(inverse) * at<1, 0>(spacetime_metric, index);
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
        lapse = sqrt(lapse);
        scalar_at(lapse_batch, index) = lapse;
        static_for<dim>([&](auto i_c) {
          constexpr size_t i = decltype(i_c)::value;
          at<i>(shift_batch, index) = get<i>(shift);
          static_for<dim - i>([&](auto offset_c) {
            constexpr size_t j = i + decltype(offset_c)::value;
            at<i, j>(inverse_spatial_metric_batch, index) = get<i, j>(inverse);
          });
        });
      });

  // 2. The Christoffel symbols with the last index raised and the gauge
  // constraint, as `gr::christoffel_first_kind` and `gh::TimeDerivative`
  Kokkos::parallel_for(
      "gh::TimeDerivativeOnDevice::christoffel",
      Kokkos::RangePolicy<>(exec, 0, total), KOKKOS_LAMBDA(const size_t k) {
        const PointIndex index = point_index(k, number_of_points);
        const ThreePlusOne three_plus_one = read_three_plus_one(
            lapse_batch, shift_batch, inverse_spatial_metric_batch, index);
        const auto inverse_metric =
            inverse_spacetime_metric(three_plus_one.lapse, three_plus_one.shift,
                                     three_plus_one.inverse_spatial_metric);
        // The time derivative of the spacetime metric without the constraint
        // terms, i.e. the time component of the spacetime derivative
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
          double trace = 0.0;
          static_for<dim + 1>([&](auto nu_c) {
            constexpr size_t nu = decltype(nu_c)::value;
            // \Gamma_{mu nu beta}
            tnsr::a<double, dim> christoffel{};
            static_for<dim + 1>([&](auto beta_c) {
              constexpr size_t beta = decltype(beta_c)::value;
              get<beta>(christoffel) = 0.5 * (d_metric(nu_c, beta_c, mu_c) +
                                              d_metric(beta_c, nu_c, mu_c) -
                                              d_metric(mu_c, nu_c, beta_c));
            });
            static_for<dim + 1>([&](auto alpha_c) {
              constexpr size_t alpha = decltype(alpha_c)::value;
              double raised =
                  get<alpha, 0>(inverse_metric) * get<0>(christoffel);
              static_for<dim>([&](auto beta_c) {
                constexpr size_t beta = decltype(beta_c)::value + 1;
                raised +=
                    get<alpha, beta>(inverse_metric) * get<beta>(christoffel);
              });
              at<mu, nu, alpha>(christoffel_3_up_batch, index) = raised;
              if constexpr (alpha == nu) {
                if constexpr (nu == 0) {
                  trace = raised;
                } else {
                  trace += raised;
                }
              }
            });
          });
          at<mu>(gauge_constraint_batch, index) = trace;
        });
      });

  const bool has_mesh_velocity = mesh_velocity.has_value();
  const Batch<MeshVelocity> mesh_velocity_batch =
      has_mesh_velocity ? *mesh_velocity : Batch<MeshVelocity>{};

  // 3. The time derivative of the spacetime metric
  Kokkos::parallel_for(
      "gh::TimeDerivativeOnDevice::dt_spacetime_metric",
      Kokkos::RangePolicy<>(exec, 0, total), KOKKOS_LAMBDA(const size_t k) {
        const PointIndex index = point_index(k, number_of_points);
        const double lapse = scalar_at(lapse_batch, index);
        tnsr::I<double, dim> shift{};
        static_for<dim>([&](auto i_c) {
          constexpr size_t i = decltype(i_c)::value;
          get<i>(shift) = at<i>(shift_batch, index);
        });
        const double gamma1_value = scalar_at(gamma1, index);
        const double gamma1_plus_1 = 1.0 + gamma1_value;
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
                  get<m>(shift) *
                  constraint(std::integral_constant<size_t, m>{});
            });
            result += gamma1_plus_1 * shift_dot_constraint;
            if (has_mesh_velocity) {
              double mesh_velocity_dot_constraint =
                  at<0>(mesh_velocity_batch, index) *
                  constraint(std::integral_constant<size_t, 0>{});
              static_for<dim - 1>([&](auto m_c) {
                constexpr size_t m = decltype(m_c)::value + 1;
                mesh_velocity_dot_constraint +=
                    at<m>(mesh_velocity_batch, index) *
                    constraint(std::integral_constant<size_t, m>{});
              });
              result += gamma1_value * mesh_velocity_dot_constraint;
            }
            at<mu, nu>(dt_spacetime_metric, index) = result;
          });
        });
      });

  // 4. The time derivative of Pi
  const DtPiArguments dt_pi_arguments{dt_pi,
                                      d_spacetime_metric,
                                      d_pi,
                                      d_phi,
                                      spacetime_metric,
                                      pi,
                                      phi,
                                      gamma0,
                                      gamma1,
                                      gamma2,
                                      has_mesh_velocity,
                                      mesh_velocity_batch,
                                      lapse_batch,
                                      shift_batch,
                                      inverse_spatial_metric_batch,
                                      christoffel_3_up_batch,
                                      gauge_constraint_batch};
  switch (dt_pi_layout_) {
    case DtPiLayout::Point:
      launch_dt_pi<DtPiPointKernel>(
          exec, dt_pi_arguments, number_of_points, total,
          std::make_index_sequence<number_of_symmetric_components>{});
      break;
    case DtPiLayout::Row:
      [&]<size_t... Rows>(std::index_sequence<Rows...> /*meta*/) {
        // The first component of row `a` is `a (2 dim + 3 - a) / 2`
        (launch_dt_pi<DtPiRowKernel>(
             exec, dt_pi_arguments, number_of_points, total,
             offset_sequence<Rows * (2 * dim + 3 - Rows) / 2>(
                 std::make_index_sequence<dim + 1 - Rows>{})),
         ...);
      }(std::make_index_sequence<dim + 1>{});
      break;
    case DtPiLayout::Component:
      [&]<size_t... Components>(std::index_sequence<Components...> /*meta*/) {
        (launch_dt_pi<DtPiComponentKernel>(exec, dt_pi_arguments,
                                           number_of_points, total,
                                           std::index_sequence<Components>{}),
         ...);
      }(std::make_index_sequence<number_of_symmetric_components>{});
      break;
    case DtPiLayout::WarpComponent:
      Kokkos::parallel_for(
          "gh::TimeDerivativeOnDevice::dt_pi",
          Kokkos::RangePolicy<>(exec, 0,
                                number_of_component_threads(
                                    total, number_of_symmetric_components)),
          DtPiWarpComponentKernel{dt_pi_arguments, number_of_points, total});
      break;
    default:
      ERROR("Unknown layout of the dt_pi kernels: "
            << static_cast<int>(dt_pi_layout_));
  }

  // 5. The time derivative of Phi
  Kokkos::parallel_for(
      "gh::TimeDerivativeOnDevice::dt_phi",
      Kokkos::RangePolicy<>(exec, 0, total), KOKKOS_LAMBDA(const size_t k) {
        const PointIndex index = point_index(k, number_of_points);
        const ThreePlusOne three_plus_one = read_three_plus_one(
            lapse_batch, shift_batch, inverse_spatial_metric_batch, index);
        const double lapse = three_plus_one.lapse;
        const auto& shift = three_plus_one.shift;
        const auto& inverse_spatial_metric =
            three_plus_one.inverse_spatial_metric;
        const auto normal = normal_vector(lapse, shift);
        const double gamma2_value = scalar_at(gamma2, index);
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
                double phi_1_up = get<n, 0>(inverse_spatial_metric) *
                                  at<0, mu, nu>(phi, index);
                static_for<dim - 1>([&](auto m_c) {
                  constexpr size_t m = decltype(m_c)::value + 1;
                  phi_1_up += get<n, m>(inverse_spatial_metric) *
                              at<m, mu, nu>(phi, index);
                });
                result += get<n + 1>(phi_one_normal) * phi_1_up;
              });
              result *= lapse;
              static_for<dim>([&](auto m_c) {
                constexpr size_t m = decltype(m_c)::value;
                result += get<m>(shift) * at<m, i, mu, nu>(d_phi, index);
              });
              at<i, mu, nu>(dt_phi, index) = result;
            });
          });
        });
      });
}
}  // namespace gh
#endif  // SPECTRE_KOKKOS
