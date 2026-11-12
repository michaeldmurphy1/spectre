// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ConservativesOnDeviceKokkos.hpp"

#ifdef SPECTRE_KOKKOS
#include <boost/preprocessor/tuple/elem.hpp>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FaceVariablesAtGridPointKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/VisitDeviceView.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
namespace {
// With a `PointwiseStatusSummary` result if `deferred_status` is `nullptr`,
// and else recording the status in `*deferred_status`
template <typename FaceTags, typename StatusPointer>
PointwiseStatusSummary compute_conservatives_impl(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<FaceTags>*> faces,
    const StatusPointer deferred_status,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* const
        fix_to_atmosphere) {
  PointwiseStatusSummary status{};
  EquationsOfState::visit_device_view(eos, [&](const auto& device_view) {
    using DeviceView = std::decay_t<decltype(device_view)>;
    const auto apply = [&]<bool FixAtmosphere>(
                           std::bool_constant<FixAtmosphere> /*meta*/,
                           const auto&... fix_view) {
      using Function = FaceVariablesAtGridPoint<DeviceView, FixAtmosphere>;
      // The variables are both arguments and results. This is safe because
      // the arguments at a grid point are read before the results there are
      // written.
      status = ApplyPointwiseOnDevice_detail::apply_with_status<Function>(
          exec, faces, deferred_status, typename Function::return_tags{},
          typename Function::argument_tags{},
          [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
            return make_device_arguments<ArgumentTags...>(
                exec,
                get<::Tags::BatchView<
                    hydro::Tags::RestMassDensity<DataVector>>>(*faces),
                get<::Tags::BatchView<
                    hydro::Tags::SpecificInternalEnergy<DataVector>>>(*faces),
                get<::Tags::BatchView<hydro::Tags::Pressure<DataVector>>>(
                    *faces),
                get<::Tags::BatchView<hydro::Tags::Temperature<DataVector>>>(
                    *faces),
                get<::Tags::BatchView<
                    hydro::Tags::ElectronFraction<DataVector>>>(*faces),
                get<::Tags::BatchView<
                    hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector,
                                                                   3>>>(*faces),
                get<::Tags::BatchView<
                    hydro::Tags::MagneticField<DataVector, 3>>>(*faces),
                get<::Tags::BatchView<
                    hydro::Tags::DivergenceCleaningField<DataVector>>>(*faces),
                get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(
                    *faces),
                get<::Tags::BatchView<
                    gr::Tags::SqrtDetSpatialMetric<DataVector>>>(*faces),
                device_view, fix_view...);
          }(typename Function::argument_tags{}));
    };
    if (fix_to_atmosphere != nullptr) {
      apply(std::true_type{}, *fix_to_atmosphere);
    } else {
      apply(std::false_type{});
    }
  });
  return status;
}
}  // namespace

template <typename FaceTags>
PointwiseStatusSummary compute_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<FaceTags>*> faces,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* const
        fix_to_atmosphere) {
  return compute_conservatives_impl(exec, faces, nullptr, eos,
                                    fix_to_atmosphere);
}

template <typename FaceTags>
void compute_conservatives_on_device(
    const Kokkos::DefaultExecutionSpace& exec,
    const gsl::not_null<BatchedVariables<FaceTags>*> faces,
    const gsl::not_null<DeferredPointwiseStatus<face_atmosphere_snapshot_tags>*>
        status,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const VariableFixing::FixToAtmosphere<3>::DeviceView* const
        fix_to_atmosphere) {
  compute_conservatives_impl(exec, faces, status.get(), eos, fix_to_atmosphere);
}

#define FACE_TAGS(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                               \
  template PointwiseStatusSummary compute_conservatives_on_device(           \
      const Kokkos::DefaultExecutionSpace& exec,                             \
      gsl::not_null<BatchedVariables<FACE_TAGS(data)>*> faces,               \
      const EquationsOfState::EquationOfState<true, 3>& eos,                 \
      const VariableFixing::FixToAtmosphere<3>::DeviceView*                  \
          fix_to_atmosphere);                                                \
  template void compute_conservatives_on_device(                             \
      const Kokkos::DefaultExecutionSpace& exec,                             \
      gsl::not_null<BatchedVariables<FACE_TAGS(data)>*> faces,               \
      gsl::not_null<DeferredPointwiseStatus<face_atmosphere_snapshot_tags>*> \
          status,                                                            \
      const EquationsOfState::EquationOfState<true, 3>& eos,                 \
      const VariableFixing::FixToAtmosphere<3>::DeviceView*                  \
          fix_to_atmosphere);
GENERATE_INSTANTIATIONS(INSTANTIATION, (face_tags, device_face_tags))
#undef INSTANTIATION
#undef FACE_TAGS
}  // namespace grmhd::ValenciaDivClean::fd
#endif  // SPECTRE_KOKKOS
