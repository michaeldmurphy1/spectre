// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Measurements/VolumeIntegrals.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
template <size_t Dim>
class Mesh;
namespace control_system::Tags {
struct WriteDataToDisk;
}  // namespace control_system::Tags
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace domain::Tags {
template <size_t Dim>
struct Mesh;
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
namespace grmhd::ValenciaDivClean::Tags {
struct TildeD;
}  // namespace grmhd::ValenciaDivClean::Tags
namespace Tags {
struct PreviousTriggerTime;
}  // namespace Tags
/// \endcond

namespace control_system {
namespace measurements {
/// \cond
template <typename ControlSystems>
struct PostReductionSendBNSStarCentersToControlSystem;
/// \endcond

namespace Tags {
/*!
 * \brief The conserved rest mass density \f$\tilde{D}\f$ in the half of the
 * grid frame of the star `Label`: \f$x \geq 0\f$ for `domain::ObjectLabel::A`
 * and \f$x < 0\f$ for `domain::ObjectLabel::B`, zero elsewhere.
 */
template <::domain::ObjectLabel Label>
struct TildeDOfStar : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "TildeDOfStar" + ::domain::name(Label); }
};

/*!
 * \brief The conserved rest mass density \f$\tilde{D}\f$ times the grid
 * coordinates in the half of the grid frame of the star `Label` (see
 * `TildeDOfStar`).
 */
template <::domain::ObjectLabel Label>
struct TildeDTimesGridCoordsOfStar : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
  static std::string name() {
    return "TildeDTimesGridCoordsOfStar" + ::domain::name(Label);
  }
};
}  // namespace Tags

namespace BNSCenterOfMass_detail {
// 1 in the half of the grid frame of the star `Label` (x >= 0 for A, x < 0
// for B) and 0 elsewhere, as a function of the grid coordinate `x`
template <::domain::ObjectLabel Label, typename DataType>
KOKKOS_FUNCTION DataType in_half_of_star(const DataType& x) {
  static_assert(Label == ::domain::ObjectLabel::A or
                Label == ::domain::ObjectLabel::B);
  if constexpr (std::is_same_v<DataType, double>) {
    const double positive_x = x >= 0.0 ? 1.0 : 0.0;
    return Label == ::domain::ObjectLabel::A ? positive_x : 1.0 - positive_x;
  } else {
    if constexpr (Label == ::domain::ObjectLabel::A) {
      return step_function(x);
    } else {
      return 1.0 - step_function(x);
    }
  }
}
}  // namespace BNSCenterOfMass_detail

namespace Tags {
/// Compute tag for `TildeDOfStar`
template <::domain::ObjectLabel Label>
struct TildeDOfStarCompute : TildeDOfStar<Label>, db::ComputeTag {
  using base = TildeDOfStar<Label>;
  using return_type = Scalar<DataVector>;
  using argument_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                 ::Events::Tags::ObserverCoordinates<3, Frame::Grid>>;
  template <typename DataType>
  KOKKOS_FUNCTION static void function(
      const gsl::not_null<Scalar<DataType>*> result,
      const Scalar<DataType>& tilde_d,
      const tnsr::I<DataType, 3, Frame::Grid>& grid_coords) {
    get(*result) =
        BNSCenterOfMass_detail::in_half_of_star<Label>(get<0>(grid_coords)) *
        get(tilde_d);
  }
};

/// Compute tag for `TildeDTimesGridCoordsOfStar`
template <::domain::ObjectLabel Label>
struct TildeDTimesGridCoordsOfStarCompute : TildeDTimesGridCoordsOfStar<Label>,
                                            db::ComputeTag {
  using base = TildeDTimesGridCoordsOfStar<Label>;
  using return_type = tnsr::I<DataVector, 3, Frame::Grid>;
  using argument_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                 ::Events::Tags::ObserverCoordinates<3, Frame::Grid>>;
  template <typename DataType>
  KOKKOS_FUNCTION static void function(
      const gsl::not_null<tnsr::I<DataType, 3, Frame::Grid>*> result,
      const Scalar<DataType>& tilde_d,
      const tnsr::I<DataType, 3, Frame::Grid>& grid_coords) {
    for (size_t i = 0; i < 3; ++i) {
      result->get(i) =
          BNSCenterOfMass_detail::in_half_of_star<Label>(get<0>(grid_coords)) *
          get(tilde_d) * grid_coords.get(i);
    }
  }
};
}  // namespace Tags

/*!
 * \brief The inertial coordinates of the grid coordinates `grid_centers` at
 * `time`, with the map of the block each of them is in.
 */
std::array<std::array<double, 3>, 3> inertial_centers(
    const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, double time,
    const std::array<std::array<double, 3>, 3>& grid_centers);
}  // namespace measurements

namespace measurements {
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for location of neutron star center (or more accurately, center
/// of mass of the matter in the x>0 (label A) or x<0 (label B) region, in grid
/// coordinates).
template <::domain::ObjectLabel Center, typename Fr>
struct NeutronStarCenter : db::SimpleTag {
  using type = std::array<double, 3>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for location of center of mass of all of the matter.
template <typename Fr>
struct SystemCenterOfMass : db::SimpleTag {
  using type = std::array<double, 3>;
};
}  // namespace Tags

/*!
 * \brief Measurement providing the location of the center of mass of the matter
 * in the \f$x \geq 0\f$ and \f$x < 0\f$ regions of the grid frame (assumed to
 * correspond to the center of mass of the two neutron stars in a BNS merger).
 *
 * \details The masses and first moments of the two regions are the volume
 * integrals of `Tags::TildeDOfStar` and `Tags::TildeDTimesGridCoordsOfStar`
 * (see `VolumeIntegralsEvent`), at the grid points of the observer mesh (e.g.
 * the FD grid of FD elements).
 */
struct BothNSCenters : tt::ConformsTo<protocols::Measurement> {
  struct FindTwoCenters : tt::ConformsTo<protocols::Submeasurement> {
    static std::string name() { return "BothNSCenters::FindTwoCenters"; }
    /// Unused aliases needed to conform to the submeasurement protocol.
    template <typename ControlSystems>
    using interpolation_target_tag = void;
    template <typename ControlSystems>
    using horizon_metavars = void;

    /// The tensors whose volume integrals give the centers of mass
    using volume_integrand_tags = tmpl::list<
        Tags::TildeDOfStarCompute<::domain::ObjectLabel::A>,
        Tags::TildeDOfStarCompute<::domain::ObjectLabel::B>,
        Tags::TildeDTimesGridCoordsOfStarCompute<::domain::ObjectLabel::A>,
        Tags::TildeDTimesGridCoordsOfStarCompute<::domain::ObjectLabel::B>>;
    template <typename ControlSystems>
    using post_reduction_action =
        PostReductionSendBNSStarCentersToControlSystem<ControlSystems>;

    template <typename ControlSystems>
    using event = VolumeIntegralsEvent<FindTwoCenters, ControlSystems>;
  };
  /// List of submeasurements used by this measurement -- only FindTwoCenters
  /// here.
  using submeasurements = tmpl::list<FindTwoCenters>;
};

/*!
 * \brief Simple action called after reduction of the center of mass data.
 *
 * \details `mass_a`, `mass_b`, `first_moment_a`, and `first_moment_b` will
 * contain the reduced data for the integral of the density (and its first
 * moment) in the x>=0 (A label) and x<0 (B label) regions, the volume
 * integrals of the `BothNSCenters::FindTwoCenters::volume_integrand_tags`
 * (see `VolumeIntegralsEvent`). This action calculates the center of mass in
 * each region, and sends the result to the control system.
 *
 * If any control system writes data to disk, i.e. if the
 * `control_system::Tags::WriteDataToDisk` tag is not empty, then this will
 * also write the centers of mass to the `/ControlSystems/BnsCenters` subfile of
 * the reductions h5 file. The columns of the file are
 *
 * - %Time
 * - Center_A_x
 * - Center_A_y
 * - Center_A_z
 * - Center_B_x
 * - Center_B_y
 * - Center_B_z
 */
template <typename ControlSystems>
struct PostReductionSendBNSStarCentersToControlSystem {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const LinkedMessageId<double>& measurement_id,
                    const std::array<double, 1>& mass_a,
                    const std::array<double, 1>& mass_b,
                    const std::array<double, 3>& first_moment_a,
                    const std::array<double, 3>& first_moment_b) {
    // Function called after reduction of the CoM data.
    // Calculate CoM from integrals
    const std::array<double, 3> grid_center_a = first_moment_a / mass_a[0];
    const std::array<double, 3> grid_center_b = first_moment_b / mass_b[0];
    const std::array<double, 3> grid_center_total =
        (first_moment_a + first_moment_b) / (mass_a[0] + mass_b[0]);

    // To convert grid coords to inertial coords, we must find the block that
    // these coords are in and use that grid to inertial map
    const auto [inertial_center_a, inertial_center_b, inertial_center_total] =
        inertial_centers(Parallel::get<domain::Tags::Domain<3>>(cache),
                         Parallel::get<domain::Tags::FunctionsOfTime>(cache),
                         measurement_id.id,
                         {{grid_center_a, grid_center_b, grid_center_total}});

    const auto center_databox = db::create<db::AddSimpleTags<
        Tags::NeutronStarCenter<::domain::ObjectLabel::A, Frame::Grid>,
        Tags::NeutronStarCenter<::domain::ObjectLabel::B, Frame::Grid>,
        Tags::NeutronStarCenter<::domain::ObjectLabel::A, Frame::Inertial>,
        Tags::NeutronStarCenter<::domain::ObjectLabel::B, Frame::Inertial>,
        Tags::SystemCenterOfMass<Frame::Grid>,
        Tags::SystemCenterOfMass<Frame::Inertial>>>(
        grid_center_a, grid_center_b, inertial_center_a, inertial_center_b,
        grid_center_total, inertial_center_total);

    // Send results to the control system(s)
    RunCallbacks<BothNSCenters::FindTwoCenters, ControlSystems>::apply(
        center_databox, cache, measurement_id);

    if (not Parallel::get<control_system::Tags::WriteDataToDisk>(cache)
                .empty()) {
      std::vector<double> grid_data_to_write{
          measurement_id.id,   grid_center_a[0],     grid_center_a[1],
          grid_center_a[2],    grid_center_b[0],     grid_center_b[1],
          grid_center_b[2],    grid_center_total[0], grid_center_total[1],
          grid_center_total[2]};

      std::vector<double> inertial_data_to_write{measurement_id.id};
      inertial_data_to_write.insert(inertial_data_to_write.end(),
                                    inertial_center_a.begin(),
                                    inertial_center_a.end());
      inertial_data_to_write.insert(inertial_data_to_write.end(),
                                    inertial_center_b.begin(),
                                    inertial_center_b.end());
      inertial_data_to_write.insert(inertial_data_to_write.end(),
                                    inertial_center_total.begin(),
                                    inertial_center_total.end());

      auto& writer_proxy = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);

      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          // Node 0 is always the writer
          writer_proxy[0], grid_subfile_path_, legend_,
          std::make_tuple(std::move(grid_data_to_write)));
      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          // Node 0 is always the writer
          writer_proxy[0], inertial_subfile_path_, legend_,
          std::make_tuple(std::move(inertial_data_to_write)));
    }
  }

 private:
  const static inline std::vector<std::string> legend_{
      "Time",           "Center_A_x",      "Center_A_y",
      "Center_A_z",     "Center_B_x",      "Center_B_y",
      "Center_B_z",     "Center_System_x", "Center_System_y",
      "Center_System_z"};
  const static inline std::string grid_subfile_path_{
      "/ControlSystems/BnsGridCenters"};
  const static inline std::string inertial_subfile_path_{
      "/ControlSystems/BnsInertialCenters"};
};
}  // namespace measurements
}  // namespace control_system
