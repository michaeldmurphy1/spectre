// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FdPatch/FixingAndRecoveryKokkos.hpp"

#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/ApplyPointwiseOnDeviceKokkos.hpp"
#include "DataStructures/BatchedVariablesKokkos.hpp"
#include "DataStructures/DeviceArgumentsKokkos.hpp"
#include "DataStructures/PointwiseStatus.hpp"
#include "Evolution/DgSubcell/FdPatch/FdPatchElementsKokkos.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservativesOnDeviceKokkos.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TMPL.hpp"

#ifdef SPECTRE_KOKKOS
namespace grmhd::ValenciaDivClean::fd_patch {
template <typename OrderedListOfPrimitiveRecoverySchemes>
FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::Statuses::Statuses(
    const Kokkos::DefaultExecutionSpace& exec)
    : face_status_(exec),
      fix_conservatives_status_(exec),
      recovery_status_(exec),
      atmosphere_status_(exec) {
  reset(exec);
  copy_to_host(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::Statuses::reset(
    const Kokkos::DefaultExecutionSpace& exec) const {
  face_status_.reset(exec);
  fix_conservatives_status_.reset(exec);
  recovery_status_.reset(exec);
  atmosphere_status_.reset(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::Statuses::
    copy_to_host(const Kokkos::DefaultExecutionSpace& exec) const {
  face_status_.copy_to_host(exec);
  fix_conservatives_status_.copy_to_host(exec);
  recovery_status_.copy_to_host(exec);
  atmosphere_status_.copy_to_host(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::FixingAndRecovery(
    const Kokkos::DefaultExecutionSpace& exec,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const ::VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere,
    const FixConservatives& fix_conservatives,
    const PrimitiveFromConservativeOptions& primitive_from_conservative_options)
    : equation_of_state_(equation_of_state.get_clone()),
      fix_to_atmosphere_(fix_to_atmosphere.device_view()),
      fix_conservatives_(fix_conservatives),
      primitive_from_conservative_options_(primitive_from_conservative_options),
      statuses_(exec) {}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::reset_statuses(
    const Kokkos::DefaultExecutionSpace& exec) const {
  statuses_.reset(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::
    copy_statuses_to_host(const Kokkos::DefaultExecutionSpace& exec) const {
  statuses_.copy_to_host(exec);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
template <typename EvolvedTags, typename MetricTags>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::
    fix_conservatives_and_recover_primitives(
        const Kokkos::DefaultExecutionSpace& exec,
        const gsl::not_null<BatchedVariables<EvolvedTags>*> evolved_variables,
        const gsl::not_null<Primitives*> primitives,
        const BatchedVariables<MetricTags>& metric) {
  fix_conservatives_and_recover_primitives(exec, evolved_variables, primitives,
                                           metric, make_not_null(&statuses_));
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
template <typename EvolvedTags, typename MetricTags>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::
    fix_conservatives_and_recover_primitives(
        const Kokkos::DefaultExecutionSpace& exec,
        const gsl::not_null<BatchedVariables<EvolvedTags>*> evolved_variables,
        const gsl::not_null<Primitives*> primitives,
        const BatchedVariables<MetricTags>& metric,
        const gsl::not_null<Statuses*> statuses) {
  const auto& spatial_metric =
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(metric);
  const auto& inv_spatial_metric =
      get<::Tags::BatchView<gr::Tags::InverseSpatialMetric<DataVector, 3>>>(
          metric);
  const auto& sqrt_det_spatial_metric =
      get<::Tags::BatchView<gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
          metric);
  std::optional<Kokkos::Profiling::ScopedRegion> region{};
  region.emplace("ValenciaDivClean::fd_patch::fix_conservatives");
  fix_conservatives_on_device(
      exec, evolved_variables,
      make_not_null(&statuses->fix_conservatives_status_), spatial_metric,
      inv_spatial_metric, sqrt_det_spatial_metric, fix_conservatives_,
      &fix_conservatives_scratch_);
  region.reset();
  region.emplace("ValenciaDivClean::fd_patch::primitive_recovery");
  primitive_from_conservative_on_device<OrderedListOfPrimitiveRecoverySchemes,
                                        true>(
      exec, primitives, make_not_null(&statuses->recovery_status_),
      get<::Tags::BatchView<Tags::TildeD>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeYe>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeTau>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeS<>>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildeB<>>>(*evolved_variables),
      get<::Tags::BatchView<Tags::TildePhi>>(*evolved_variables),
      spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric,
      *equation_of_state_, primitive_from_conservative_options_,
      &primitive_recovery_scratch_);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
template <typename MetricTags>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::
    fix_to_atmosphere(const Kokkos::DefaultExecutionSpace& exec,
                      const gsl::not_null<Primitives*> primitives,
                      const BatchedVariables<MetricTags>& metric) {
  const Kokkos::Profiling::ScopedRegion region{
      "ValenciaDivClean::fd_patch::fix_to_atmosphere"};
  ::VariableFixing::fix_to_atmosphere_on_device(
      exec, primitives, make_not_null(&statuses_.atmosphere_status_),
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(metric),
      *equation_of_state_, fix_to_atmosphere_);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
template <typename MetricTags>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::
    fix_to_atmosphere_now(
        const Kokkos::DefaultExecutionSpace& exec,
        const gsl::not_null<Primitives*> primitives,
        const BatchedVariables<MetricTags>& metric,
        const evolution::dg::subcell::fd_patch::FdPatchElements& patch_elements)
        const {
  const auto status = ::VariableFixing::fix_to_atmosphere_on_device(
      exec, primitives,
      get<::Tags::BatchView<gr::Tags::SpatialMetric<DataVector, 3>>>(metric),
      *equation_of_state_, fix_to_atmosphere_);
  if (status.number_of_failures > 0) {
    ::VariableFixing::error_fixing_to_atmosphere_on_device(
        exec, status, *primitives,
        patch_elements.element_description(status.first_failed_element));
  }
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
template <typename EvolvedTags, typename MetricTags>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::
    compute_conservatives(
        const Kokkos::DefaultExecutionSpace& exec,
        const gsl::not_null<BatchedVariables<EvolvedTags>*> evolved_variables,
        const Primitives& primitives,
        const BatchedVariables<MetricTags>& metric) const {
  const Kokkos::Profiling::ScopedRegion region{
      "ValenciaDivClean::fd_patch::compute_conservatives"};
  const auto argument = [&primitives,
                         &metric]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    if constexpr (tmpl::list_contains_v<hydro::grmhd_tags<DataVector>, Tag>) {
      return get<::Tags::BatchView<Tag>>(primitives);
    } else {
      return get<::Tags::BatchView<Tag>>(metric);
    }
  };
  apply_pointwise_on_device<ConservativeFromPrimitive>(
      exec, evolved_variables, ConservativeFromPrimitive::return_tags{},
      ConservativeFromPrimitive::argument_tags{},
      [&]<typename... ArgumentTags>(tmpl::list<ArgumentTags...> /*meta*/) {
        return make_device_arguments<ArgumentTags...>(
            exec, argument(tmpl::type_<ArgumentTags>{})...);
      }(ConservativeFromPrimitive::argument_tags{}));
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::check(
    const Kokkos::DefaultExecutionSpace& exec,
    const evolution::dg::subcell::fd_patch::FdPatchElements& patch_elements,
    const size_t number_of_face_points) const {
  statuses_.check(exec, patch_elements, number_of_face_points);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void FixingAndRecovery<OrderedListOfPrimitiveRecoverySchemes>::Statuses::check(
    const Kokkos::DefaultExecutionSpace& exec,
    const evolution::dg::subcell::fd_patch::FdPatchElements& patch_elements,
    const size_t number_of_face_points, const std::string& context) const {
  exec.fence("ValenciaDivClean::fd_patch::FixingAndRecovery::check");
  const auto description = [&patch_elements,
                            &context](const PointwiseStatusSummary& status) {
    return context +
           patch_elements.element_description(status.first_failed_element);
  };
  const size_t number_of_cells_of_all_elements =
      patch_elements.subcell_mesh().number_of_grid_points() *
      patch_elements.number_of_elements();
  // In the order of a step, so the first failure is reported
  if (face_status_.failed()) {
    const auto status = face_status_.summary();
    // The faces of all elements are in one batch, 3 face sets per element
    ::VariableFixing::error_fixing_to_atmosphere_on_device(
        face_status_, 2 * number_of_face_points,
        context + "a face of " +
            patch_elements.element_description(
                status.first_failed_element %
                patch_elements.number_of_elements()));
  }
  if (fix_conservatives_status_.failed()) {
    error_fixing_conservatives_on_device(
        fix_conservatives_status_, number_of_cells_of_all_elements,
        description(fix_conservatives_status_.summary()));
  }
  if (recovery_status_.failed()) {
    error_recovering_primitives_on_device(
        recovery_status_, description(recovery_status_.summary()));
  }
  if (atmosphere_status_.failed()) {
    ::VariableFixing::error_fixing_to_atmosphere_on_device(
        atmosphere_status_, number_of_cells_of_all_elements,
        description(atmosphere_status_.summary()));
  }
}
}  // namespace grmhd::ValenciaDivClean::fd_patch
#endif  // SPECTRE_KOKKOS
