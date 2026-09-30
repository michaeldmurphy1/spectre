// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"

#include <cstddef>
#include <limits>
#include <type_traits>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
template <typename DataType>
KOKKOS_FUNCTION void densitized_stress(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*> result,
    const tnsr::II<DataType, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& one_over_w_squared,
    const Scalar<DataType>& pressure_star,
    const Scalar<DataType>& h_rho_w_squared_plus_b_squared,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataType, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataType>& sqrt_det_spatial_metric) {
  *result = inv_spatial_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) *= get(pressure_star);
      result->get(i, j) +=
          get(h_rho_w_squared_plus_b_squared) * spatial_velocity.get(i) *
              spatial_velocity.get(j) -
          get(magnetic_field_dot_spatial_velocity) *
              (magnetic_field.get(i) * spatial_velocity.get(j) +
               magnetic_field.get(j) * spatial_velocity.get(i)) -
          magnetic_field.get(i) * magnetic_field.get(j) *
              get(one_over_w_squared);
      result->get(i, j) *= get(sqrt_det_spatial_metric);
    }
  }
}

template <typename DataType>
struct MagneticFieldOneForm : db::SimpleTag {
  using type = tnsr::i<DataType, 3, Frame::Inertial>;
};
template <typename DataType>
struct TildeSUp : db::SimpleTag {
  using type = tnsr::I<DataType, 3, Frame::Inertial>;
};
template <typename DataType>
struct DensitizedStress : db::SimpleTag {
  using type = tnsr::II<DataType, 3, Frame::Inertial>;
};
template <typename DataType>
struct OneOverLorentzFactorSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct PressureStar : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct EnthalpyTimesDensityWSquaredPlusBSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace

namespace grmhd::ValenciaDivClean {
namespace detail {
template <typename DataType>
KOKKOS_FUNCTION void sources_impl(
    const gsl::not_null<Scalar<DataType>*> source_tilde_tau,
    const gsl::not_null<tnsr::i<DataType, 3, Frame::Inertial>*> source_tilde_s,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> source_tilde_b,
    const gsl::not_null<Scalar<DataType>*> source_tilde_phi,

    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> tilde_s_up,
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        densitized_stress,
    const gsl::not_null<Scalar<DataType>*> h_rho_w_squared_plus_b_squared,

    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& magnetic_field_squared,
    const Scalar<DataType>& one_over_w_squared,
    const Scalar<DataType>& pressure_star,
    const tnsr::I<DataType, 3, Frame::Inertial>&
        trace_spatial_christoffel_second,

    const Scalar<DataType>& tilde_d, const Scalar<DataType>& /* tilde_ye */,
    const Scalar<DataType>& tilde_tau,
    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataType>& tilde_phi, const Scalar<DataType>& lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataType, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataType, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataType, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, 3, Frame::Inertial>& magnetic_field,

    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /* electron_fraction */,
    const Scalar<DataType>& pressure,
    const Scalar<DataType>& specific_internal_energy,
    const tnsr::ii<DataType, 3, Frame::Inertial>& extrinsic_curvature,
    const double constraint_damping_parameter) {
  get(*h_rho_w_squared_plus_b_squared) =
      get(magnetic_field_squared) +
      (get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
       get(pressure)) *
          square(get(lorentz_factor));
  ::densitized_stress<DataType>(
      densitized_stress, inv_spatial_metric,
      magnetic_field_dot_spatial_velocity, one_over_w_squared, pressure_star,
      *h_rho_w_squared_plus_b_squared, spatial_velocity, magnetic_field,
      sqrt_det_spatial_metric);
  raise_or_lower_index(tilde_s_up, tilde_s, inv_spatial_metric);

  get(*source_tilde_tau) = get(lapse) * get<0, 0>(extrinsic_curvature) *
                               get<0, 0>(*densitized_stress) -
                           get<0>(*tilde_s_up) * get<0>(d_lapse);
  for (size_t m = 1; m < 3; ++m) {
    get(*source_tilde_tau) +=
        get(lapse) *
            (extrinsic_curvature.get(m, 0) * densitized_stress->get(m, 0) +
             extrinsic_curvature.get(0, m) * densitized_stress->get(0, m)) -
        tilde_s_up->get(m) * d_lapse.get(m);
    for (size_t n = 1; n < 3; ++n) {
      get(*source_tilde_tau) += get(lapse) * extrinsic_curvature.get(m, n) *
                                densitized_stress->get(m, n);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    source_tilde_s->get(i) = -(get(tilde_d) + get(tilde_tau)) * d_lapse.get(i);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_s->get(i) += tilde_s.get(m) * d_shift.get(i, m);
      for (size_t n = 0; n < 3; ++n) {
        source_tilde_s->get(i) += 0.5 * get(lapse) *
                                  d_spatial_metric.get(i, m, n) *
                                  densitized_stress->get(m, n);
      }
    }
  }

  raise_or_lower_index(source_tilde_b, d_lapse, inv_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    source_tilde_b->get(i) *= get(tilde_phi);
    source_tilde_b->get(i) -=
        get(lapse) * get(tilde_phi) * trace_spatial_christoffel_second.get(i);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_b->get(i) -= tilde_b.get(m) * d_shift.get(m, i);
    }
  }

  trace(source_tilde_phi, extrinsic_curvature, inv_spatial_metric);
  get(*source_tilde_phi) += constraint_damping_parameter;
  get(*source_tilde_phi) *= -1.0 * get(lapse) * get(tilde_phi);
  for (size_t m = 0; m < 3; ++m) {
    get(*source_tilde_phi) += tilde_b.get(m) * d_lapse.get(m);
  }
}

template <typename DataType>
KOKKOS_FUNCTION void cartoon_sources_impl(
    const gsl::not_null<tnsr::i<DataType, 3, Frame::Inertial>*> source_tilde_s,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> source_tilde_b,

    const Scalar<DataType>& pressure_star,
    const tnsr::i<DataType, 3, Frame::Inertial>& magnetic_field_one_form,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,

    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataType>& tilde_phi,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::I<DataType, 3, Frame::Inertial>& inertial_coords,
    const Spectral::Quadrature cartoon_quadrature) {
#ifdef SPECTRE_DEBUG
  if constexpr (std::is_same_v<DataType, double>) {
    // A single grid point, possibly on the device where we can't use `ASSERT`
    KOKKOS_IF_ON_HOST(
        (ASSERT(get<0>(inertial_coords) != 0.0,
                "Cannot compute the Cartoon source terms with x=0 in the "
                "domain but we got inertial_coords = "
                    << inertial_coords);))
    KOKKOS_IF_ON_DEVICE((if (get<0>(inertial_coords) == 0.0) {
      Kokkos::abort(
          "Cannot compute the Cartoon source terms with x=0 in the domain");
    }))
  } else {
    for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
      ASSERT(not equal_within_roundoff(
                 0.0, get<0>(inertial_coords)[i],
                 std::numeric_limits<double>::epsilon() * 100.0,
                 max(get<0>(inertial_coords))),
             "Cannot compute the Cartoon source terms with x=0 in the domain "
             "but we got inertial_coords = "
                 << inertial_coords);
    }
  }
#endif  // SPECTRE_DEBUG
  if (cartoon_quadrature == Spectral::Quadrature::SphericalSymmetry) {
    get<0>(*source_tilde_s) += 2.0 * get(lapse) * get(sqrt_det_spatial_metric) *
                               get(pressure_star) / get<0>(inertial_coords);
    get<0>(*source_tilde_b) +=
        get(lapse) *
        (get<1, 1>(inv_spatial_metric) + get<2, 2>(inv_spatial_metric)) *
        get(tilde_phi) / get<0>(inertial_coords);
  } else {
    KOKKOS_IF_ON_HOST(
        (ASSERT(cartoon_quadrature == Spectral::Quadrature::AxialSymmetry,
                "Got unexpected quadrature for Cartoon basis: "
                    << cartoon_quadrature);))

    TempBuffer<tmpl::list<
        hydro::Tags::SpatialVelocityOneForm<DataType, 3, Frame::Inertial>,
        ::Tags::Tempa<0, 3, Frame::Inertial, DataType>,
        hydro::Tags::TransportVelocity<DataType, 3, Frame::Inertial>>>
        temp_tensors{get_size(get(lapse))};

    auto& spatial_velocity_one_form =
        get<hydro::Tags::SpatialVelocityOneForm<DataType, 3, Frame::Inertial>>(
            temp_tensors);
    raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                         spatial_velocity, spatial_metric);

    auto& comoving_magnetic_field_one_form =
        get<::Tags::Tempa<0, 3, Frame::Inertial, DataType>>(temp_tensors);
    hydro::comoving_magnetic_field_one_form(
        make_not_null(&comoving_magnetic_field_one_form),
        spatial_velocity_one_form, magnetic_field_one_form,
        magnetic_field_dot_spatial_velocity, lorentz_factor, shift, lapse);

    auto& transport_velocity =
        get<hydro::Tags::TransportVelocity<DataType, 3, Frame::Inertial>>(
            temp_tensors);
    hydro::transport_velocity(make_not_null(&transport_velocity),
                              spatial_velocity, lapse, shift);

    get<0>(*source_tilde_s) +=
        (get(lapse) *
             (get(sqrt_det_spatial_metric) * get(pressure_star) -
              get<2>(tilde_b) * get<3>(comoving_magnetic_field_one_form) /
                  get(lorentz_factor)) +
         get<2>(tilde_s) * get<2>(transport_velocity)) /
        get<0>(inertial_coords);
    get<2>(*source_tilde_s) +=
        (get(lapse) * get<2>(tilde_b) *
             get<1>(comoving_magnetic_field_one_form) / get(lorentz_factor) -
         get<0>(tilde_s) * get<2>(transport_velocity)) /
        get<0>(inertial_coords);

    get<0>(*source_tilde_b) +=
        (get(lapse) * (get<2, 2>(inv_spatial_metric) * get(tilde_phi) -
                       get<2>(spatial_velocity) * get<2>(tilde_b)) +
         get<2>(tilde_b) * get<2>(transport_velocity)) /
        get<0>(inertial_coords);
    get<2>(*source_tilde_b) +=
        (get(lapse) * (-get<2, 0>(inv_spatial_metric) * get(tilde_phi) +
                       get<0>(spatial_velocity) * get<2>(tilde_b)) -
         get<0>(tilde_b) * get<2>(transport_velocity)) /
        get<0>(inertial_coords);
  }
}
}  // namespace detail

template <typename DataType>
KOKKOS_FUNCTION void ComputeSources::apply(
    const gsl::not_null<Scalar<DataType>*> source_tilde_tau,
    const gsl::not_null<tnsr::i<DataType, 3, Frame::Inertial>*> source_tilde_s,
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*> source_tilde_b,
    const gsl::not_null<Scalar<DataType>*> source_tilde_phi,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& tilde_ye,
    const Scalar<DataType>& tilde_tau,
    const tnsr::i<DataType, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataType, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataType>& tilde_phi,
    const tnsr::I<DataType, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataType, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& electron_fraction,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& lorentz_factor, const Scalar<DataType>& pressure,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::i<DataType, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataType, 3, Frame::Inertial>& d_shift,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::ijj<DataType, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, 3, Frame::Inertial>& extrinsic_curvature,
    const double constraint_damping_parameter,
    const tnsr::I<DataType, 3, Frame::Inertial>& inertial_coords,
    const Mesh<3>& dg_mesh) {
  TempBuffer<
      tmpl::list<TildeSUp<DataType>, DensitizedStress<DataType>,
                 MagneticFieldOneForm<DataType>,
                 hydro::Tags::MagneticFieldDotSpatialVelocity<DataType>,
                 hydro::Tags::MagneticFieldSquared<DataType>,
                 OneOverLorentzFactorSquared<DataType>, PressureStar<DataType>,
                 EnthalpyTimesDensityWSquaredPlusBSquared<DataType>,
                 gr::Tags::TraceSpatialChristoffelFirstKind<DataType, 3>,
                 gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3>>>
      temp_tensors{get_size(get(tilde_d))};

  auto& magnetic_field_oneform =
      get<MagneticFieldOneForm<DataType>>(temp_tensors);
  raise_or_lower_index(make_not_null(&magnetic_field_oneform), magnetic_field,
                       spatial_metric);

  auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataType>>(temp_tensors);
  dot_product(make_not_null(&magnetic_field_squared), magnetic_field,
              magnetic_field_oneform);

  auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataType>>(temp_tensors);
  dot_product(make_not_null(&magnetic_field_dot_spatial_velocity),
              magnetic_field_oneform, spatial_velocity);

  auto& one_over_w_squared =
      get<OneOverLorentzFactorSquared<DataType>>(temp_tensors);
  get(one_over_w_squared) = 1.0 / square(get(lorentz_factor));

  auto& pressure_star = get<PressureStar<DataType>>(temp_tensors);
  get(pressure_star) =
      get(pressure) + 0.5 * square(get(magnetic_field_dot_spatial_velocity)) +
      0.5 * get(magnetic_field_squared) * get(one_over_w_squared);

  auto& trace_spatial_christoffel_first =
      get<gr::Tags::TraceSpatialChristoffelFirstKind<DataType, 3>>(
          temp_tensors);
  gr::trace_spatial_christoffel_first_kind(
      make_not_null(&trace_spatial_christoffel_first), d_spatial_metric,
      inv_spatial_metric);
  auto& trace_spatial_christoffel_second =
      get<gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3>>(
          temp_tensors);
  raise_or_lower_index(make_not_null(&trace_spatial_christoffel_second),
                       trace_spatial_christoffel_first, inv_spatial_metric);

  detail::sources_impl(
      source_tilde_tau, source_tilde_s, source_tilde_b, source_tilde_phi,

      make_not_null(&get<TildeSUp<DataType>>(temp_tensors)),
      make_not_null(&get<DensitizedStress<DataType>>(temp_tensors)),
      make_not_null(&get<EnthalpyTimesDensityWSquaredPlusBSquared<DataType>>(
          temp_tensors)),

      magnetic_field_dot_spatial_velocity, magnetic_field_squared,
      one_over_w_squared, pressure_star, trace_spatial_christoffel_second,

      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
      sqrt_det_spatial_metric, inv_spatial_metric, d_lapse, d_shift,
      d_spatial_metric, spatial_velocity, lorentz_factor, magnetic_field,

      rest_mass_density, electron_fraction, pressure, specific_internal_energy,
      extrinsic_curvature, constraint_damping_parameter);

  if (dg_mesh.basis(2) == Spectral::Basis::Cartoon) {
    detail::cartoon_sources_impl(
        source_tilde_s, source_tilde_b, pressure_star, magnetic_field_oneform,
        magnetic_field_dot_spatial_velocity, tilde_s, tilde_b, tilde_phi,
        spatial_velocity, lorentz_factor, lapse, shift, spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric, inertial_coords,
        dg_mesh.quadrature(2));
  }
}
}  // namespace grmhd::ValenciaDivClean

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void grmhd::ValenciaDivClean::detail::sources_impl(                 \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_tau,                    \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*> source_tilde_s, \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> source_tilde_b, \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_phi,                    \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> tilde_s_up,     \
      gsl::not_null<tnsr::II<DTYPE(data), 3, Frame::Inertial>*>                \
          densitized_stress,                                                   \
      gsl::not_null<Scalar<DTYPE(data)>*> h_rho_w_squared_plus_b_squared,      \
      const Scalar<DTYPE(data)>& magnetic_field_dot_spatial_velocity,          \
      const Scalar<DTYPE(data)>& magnetic_field_squared,                       \
      const Scalar<DTYPE(data)>& one_over_w_squared,                           \
      const Scalar<DTYPE(data)>& pressure_star,                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&                          \
          trace_spatial_christoffel_second,                                    \
      const Scalar<DTYPE(data)>& tilde_d, const Scalar<DTYPE(data)>& tilde_ye, \
      const Scalar<DTYPE(data)>& tilde_tau,                                    \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi, const Scalar<DTYPE(data)>& lapse,  \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                      \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& d_lapse,                 \
      const tnsr::iJ<DTYPE(data), 3, Frame::Inertial>& d_shift,                \
      const tnsr::ijj<DTYPE(data), 3, Frame::Inertial>& d_spatial_metric,      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,        \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& magnetic_field,          \
      const Scalar<DTYPE(data)>& rest_mass_density,                            \
      const Scalar<DTYPE(data)>& electron_fraction,                            \
      const Scalar<DTYPE(data)>& pressure,                                     \
      const Scalar<DTYPE(data)>& specific_internal_energy,                     \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& extrinsic_curvature,    \
      double constraint_damping_parameter);                                    \
  template void grmhd::ValenciaDivClean::ComputeSources::apply(                \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_tau,                    \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*> source_tilde_s, \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> source_tilde_b, \
      gsl::not_null<Scalar<DTYPE(data)>*> source_tilde_phi,                    \
      const Scalar<DTYPE(data)>& tilde_d, const Scalar<DTYPE(data)>& tilde_ye, \
      const Scalar<DTYPE(data)>& tilde_tau,                                    \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi,                                    \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& magnetic_field,          \
      const Scalar<DTYPE(data)>& rest_mass_density,                            \
      const Scalar<DTYPE(data)>& electron_fraction,                            \
      const Scalar<DTYPE(data)>& specific_internal_energy,                     \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const Scalar<DTYPE(data)>& pressure, const Scalar<DTYPE(data)>& lapse,   \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& d_lapse,                 \
      const tnsr::iJ<DTYPE(data), 3, Frame::Inertial>& d_shift,                \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::ijj<DTYPE(data), 3, Frame::Inertial>& d_spatial_metric,      \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                      \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& extrinsic_curvature,    \
      double constraint_damping_parameter,                                     \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& inertial_coords,         \
      const Mesh<3>& dg_mesh);                                                 \
  template void grmhd::ValenciaDivClean::detail::cartoon_sources_impl(         \
      gsl::not_null<tnsr::i<DTYPE(data), 3, Frame::Inertial>*> source_tilde_s, \
      gsl::not_null<tnsr::I<DTYPE(data), 3, Frame::Inertial>*> source_tilde_b, \
      const Scalar<DTYPE(data)>& pressure_star,                                \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& magnetic_field_one_form, \
      const Scalar<DTYPE(data)>& magnetic_field_dot_spatial_velocity,          \
      const tnsr::i<DTYPE(data), 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DTYPE(data)>& tilde_phi,                                    \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& spatial_velocity,        \
      const Scalar<DTYPE(data)>& lorentz_factor,                               \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& shift,                   \
      const tnsr::ii<DTYPE(data), 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::II<DTYPE(data), 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>& inertial_coords,         \
      Spectral::Quadrature cartoon_quadrature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double))
// The `DataVector` versions are never called on the device
#ifndef SPECTRE_KOKKOS_DEVICE_PASS
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))
#endif  // SPECTRE_KOKKOS_DEVICE_PASS

#undef INSTANTIATE
#undef DTYPE
