// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.tpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

// Explicit Instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                     \
  gh::three_index_constraint(                                                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          d_spacetime_metric,                                                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);             \
  template void gh::three_index_constraint(                                   \
      const gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*>    \
          constraint,                                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          d_spacetime_metric,                                                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);             \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)> gh::gauge_constraint( \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,     \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);             \
  template void gh::gauge_constraint(                                         \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          constraint,                                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,     \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);             \
  template tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>                      \
  gh::two_index_constraint(                                                   \
      const tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_d_gauge_function,                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const Scalar<DTYPE(data)>& gamma2,                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint);                                            \
  template void gh::two_index_constraint(                                     \
      const gsl::not_null<tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>*>     \
          constraint,                                                         \
      const tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_d_gauge_function,                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const Scalar<DTYPE(data)>& gamma2,                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint);                                            \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)> gh::f_constraint(     \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,     \
      const tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_d_gauge_function,                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const Scalar<DTYPE(data)>& gamma2,                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint);                                            \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)> gh::f_constraint(     \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,     \
      const tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_d_gauge_function,                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const Scalar<DTYPE(data)>& gamma2,                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint,                                             \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>&                    \
          trace_reversed_stress_energy);                                      \
  template void gh::f_constraint(                                             \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          constraint,                                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,     \
      const tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_d_gauge_function,                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const Scalar<DTYPE(data)>& gamma2,                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint);                                            \
  template void gh::f_constraint(                                             \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          constraint,                                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,     \
      const tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_d_gauge_function,                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_one_form,                                          \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const Scalar<DTYPE(data)>& gamma2,                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint,                                             \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>&                    \
          trace_reversed_stress_energy);                                      \
  template Scalar<DTYPE(data)> gh::constraint_energy(                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_constraint,   \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& f_constraint,       \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>&                    \
          two_index_constraint,                                               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint,                                             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          four_index_constraint,                                              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& spatial_metric_determinant,                  \
      double gauge_constraint_multiplier,                                     \
      double two_index_constraint_multiplier,                                 \
      double three_index_constraint_multiplier,                               \
      double four_index_constraint_multiplier);                               \
  template void gh::constraint_energy(                                        \
      const gsl::not_null<Scalar<DTYPE(data)>*> energy,                       \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_constraint,   \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& f_constraint,       \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>&                    \
          two_index_constraint,                                               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          three_index_constraint,                                             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          four_index_constraint,                                              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& spatial_metric_determinant,                  \
      double gauge_constraint_multiplier,                                     \
      double two_index_constraint_multiplier,                                 \
      double three_index_constraint_multiplier,                               \
      double four_index_constraint_multiplier);                               \
  template Scalar<DTYPE(data)> gh::constraint_energy_normalization(           \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          d_spacetime_metric,                                                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& sqrt_spatial_metric_determinant,             \
      double dimensional_constant);                                           \
  template void gh::constraint_energy_normalization(                          \
      const gsl::not_null<Scalar<DTYPE(data)>*> energy_norm,                  \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          d_spacetime_metric,                                                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,             \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& sqrt_spatial_metric_determinant,             \
      double dimensional_constant);

template Scalar<double> gh::normalized_constraint_energy(
    const Scalar<double>& constraint_energy,
    const Scalar<double>& constraint_energy_normalization);
template Scalar<DataVector> gh::normalized_constraint_energy(
    const Scalar<DataVector>& constraint_energy,
    const Scalar<DataVector>& constraint_energy_normalization);
template void gh::normalized_constraint_energy(
    gsl::not_null<Scalar<double>*> normalized_energy,
    const Scalar<double>& constraint_energy,
    const Scalar<double>& constraint_energy_normalization);
template void gh::normalized_constraint_energy(
    gsl::not_null<Scalar<DataVector>*> normalized_energy,
    const Scalar<DataVector>& constraint_energy,
    const Scalar<DataVector>& constraint_energy_normalization);

#define INSTANTIATE_ONLY_3D(_, data)                                       \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                  \
  gh::four_index_constraint(                                               \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi);       \
  template void gh::four_index_constraint(                                 \
      const gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*> \
          constraint,                                                      \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Inertial))
GENERATE_INSTANTIATIONS(INSTANTIATE_ONLY_3D, (3),
                        SPECTRE_DEVICE_AND_HOST_TYPES((double),
                                                      (double, DataVector)),
                        (Frame::Grid, Frame::Inertial))
#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
