// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "ControlSystem/Component.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::measurements {
namespace VolumeIntegrals_detail {
// The tags that the `IntegrandTags` (compute tags) are computed from
template <typename IntegrandTags>
struct argument_tags;

template <typename... IntegrandTags>
struct argument_tags<tmpl::list<IntegrandTags...>> {
  using type = tmpl::remove_duplicates<
      tmpl::flatten<tmpl::list<typename IntegrandTags::argument_tags...>>>;
};

// The sum of the integrals of two parts of the elements
struct AddIntegrals {
  template <size_t Size>
  std::array<double, Size> operator()(const std::array<double, Size>& lhs,
                                      const std::array<double, Size>& rhs) const {
    std::array<double, Size> result{};
    for (size_t i = 0; i < Size; ++i) {
      gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
    }
    return result;
  }
};
}  // namespace VolumeIntegrals_detail

/// \cond
template <
    typename Submeasurement, typename ControlSystems,
    typename IntegrandTags = typename Submeasurement::volume_integrand_tags,
    typename ArgumentTags =
        typename VolumeIntegrals_detail::argument_tags<IntegrandTags>::type>
class VolumeIntegralsEvent;
/// \endcond

/*!
 * \brief The `::Event` of a submeasurement that integrates tensors over the
 * volume of all elements.
 *
 * \details The `Submeasurement` defines the tensors by
 * `volume_integrand_tags`, a list of compute tags, and receives the integrals
 * of all their components, summed over the elements, in the reduction action
 * `post_reduction_action<ControlSystems>` (as a `std::array<double, N>` per
 * integrand, after the `LinkedMessageId` of the measurement). The integral of
 * a component \f$u\f$ over an element is \f$\sum_i w_i J_i u_i\f$ with the
 * quadrature weights \f$w_i\f$ and the determinant \f$J_i\f$ of the Jacobian
 * from the logical to the inertial frame, at the grid points of the observer
 * mesh (e.g. the FD grid of FD elements), as `Events::ObserveNorms` computes a
 * `VolumeIntegral`.
 *
 * The compute tags are evaluated from their argument tags, so they don't need
 * to be in the `ObservationBox` of the events. The integrals are a declarative
 * definition that can be evaluated elsewhere, e.g. by FD patches on a device
 * (whose functions of the compute tags must then be device-callable function
 * templates over the data type), which then `contribute` them.
 *
 * The result is reduced to the control component of the first control system
 * (any of them works, see `RunCallbacks`). If the `Metavariables` set
 * `combined_reduction` (e.g. to `evolution::patches::CombinedReduction`), the
 * data is reduced with it, so that elements in several parallel components
 * contribute.
 */
template <typename Submeasurement, typename ControlSystems,
          typename... IntegrandTags, typename... ArgumentTags>
class VolumeIntegralsEvent<Submeasurement, ControlSystems,
                           tmpl::list<IntegrandTags...>,
                           tmpl::list<ArgumentTags...>> : public ::Event {
 public:
  /// \cond
  // LCOV_EXCL_START
  explicit VolumeIntegralsEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(VolumeIntegralsEvent);  // NOLINT
  // LCOV_EXCL_STOP
  /// \endcond

  // This event is created during control system initialization, not
  // from the input file.
  static constexpr bool factory_creatable = false;
  VolumeIntegralsEvent() = default;

  /// The compute tags of the tensors that are integrated
  using volume_integrand_tags = tmpl::list<IntegrandTags...>;
  /// The integrals of the components of each tensor
  using integrals_type =
      std::tuple<std::array<double, IntegrandTags::type::size()>...>;
  using reduction_data = Parallel::ReductionData<
      Parallel::ReductionDatum<LinkedMessageId<double>, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<std::array<double, IntegrandTags::type::size()>,
                               VolumeIntegrals_detail::AddIntegrals>...>;

  using compute_tags_for_observation_box = tmpl::list<>;
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Time, ::Tags::PreviousTriggerTime,
                 ::Events::Tags::ObserverMesh<3>,
                 ::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                        Frame::Inertial>,
                 ArgumentTags...>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const double time,
                  const std::optional<double>& previous_trigger_time,
                  const Mesh<3>& mesh,
                  const Scalar<DataVector>& det_inv_jacobian,
                  const typename ArgumentTags::type&... arguments,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    contribute(
        cache,
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        time, previous_trigger_time,
        integrals(mesh, det_inv_jacobian, arguments...));
  }

  /*!
   * \brief The integrals over an element with the `mesh`, the determinant
   * `det_inv_jacobian` of the inverse Jacobian from the logical to the
   * inertial frame and the arguments of the compute tags.
   */
  static integrals_type integrals(
      const Mesh<3>& mesh, const Scalar<DataVector>& det_inv_jacobian,
      const typename ArgumentTags::type&... arguments) {
    const DataVector det_jacobian = 1.0 / get(det_inv_jacobian);
    const auto all_arguments = std::forward_as_tuple(arguments...);
    const auto integral = [&]<typename IntegrandTag>(
                              tmpl::type_<IntegrandTag> /*meta*/) {
      typename IntegrandTag::type integrand{mesh.number_of_grid_points()};
      [&]<typename... TagArguments>(tmpl::list<TagArguments...> /*meta*/) {
        IntegrandTag::function(
            make_not_null(&integrand),
            std::get<tmpl::index_of<tmpl::list<ArgumentTags...>,
                                    TagArguments>::value>(all_arguments)...);
      }(typename IntegrandTag::argument_tags{});
      std::array<double, IntegrandTag::type::size()> result{};
      for (size_t i = 0; i < integrand.size(); ++i) {
        gsl::at(result, i) =
            definite_integral(det_jacobian * integrand[i], mesh);
      }
      return result;
    };
    return {integral(tmpl::type_<IntegrandTags>{})...};
  }

  /// The integrals over an element from the `box` of an element (e.g. an
  /// `ObservationBox`) that holds the `argument_tags`
  template <typename Box>
  static integrals_type integrals(const Box& box) {
    return integrals(
        get<::Events::Tags::ObserverMesh<3>>(box),
        get<::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                   Frame::Inertial>>(box),
        get<ArgumentTags>(box)...);
  }

  /*!
   * \brief Contribute the `integrals` over the elements of the sender
   * `sender_proxy` (an element, or e.g. a patch of elements) to the
   * measurement at `time`.
   */
  template <typename Metavariables, typename SenderProxy>
  static void contribute(Parallel::GlobalCache<Metavariables>& cache,
                         const SenderProxy& sender_proxy, const double time,
                         const std::optional<double>& previous_trigger_time,
                         const integrals_type& integrals) {
    reduction_data data = std::apply(
        [&](const auto&... values) {
          return reduction_data{
              LinkedMessageId<double>{.id = time,
                                      .previous = previous_trigger_time},
              values...};
        },
        integrals);
    using Action =
        typename Submeasurement::template post_reduction_action<ControlSystems>;
    using Target = ControlComponent<Metavariables, tmpl::front<ControlSystems>>;
    if constexpr (requires { typename Metavariables::combined_reduction; }) {
      Metavariables::combined_reduction::template contribute<
          Action, tmpl::list<Target>>(cache, std::move(data), sender_proxy);
    } else {
      Parallel::contribute_to_reduction<Action>(
          std::move(data), sender_proxy,
          Parallel::get_parallel_component<Target>(cache));
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*component*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }
};

/// \cond
template <typename Submeasurement, typename ControlSystems,
          typename... IntegrandTags, typename... ArgumentTags>
PUP::able::PUP_ID VolumeIntegralsEvent<Submeasurement, ControlSystems,
                                       tmpl::list<IntegrandTags...>,
                                       tmpl::list<ArgumentTags...>>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace control_system::measurements
