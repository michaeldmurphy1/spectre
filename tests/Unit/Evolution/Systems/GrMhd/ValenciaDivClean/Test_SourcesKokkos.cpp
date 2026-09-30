// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Helpers/DataStructures/CheckPointwiseOnDevice.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Sources.Kokkos",
                  "[Unit][GrMhd]") {
  // Check the device version with and without the Cartoon terms. The random
  // coordinates are positive, so x != 0.
  for (const auto& dg_mesh :
       {Mesh<3>{3_st, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        Mesh<3>{{3_st, 3_st, 1_st},
                {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                  Spectral::Basis::Cartoon}},
                {{Spectral::Quadrature::GaussLobatto,
                  Spectral::Quadrature::GaussLobatto,
                  Spectral::Quadrature::AxialSymmetry}}},
        Mesh<3>{{3_st, 1_st, 1_st},
                {{Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
                  Spectral::Basis::Cartoon}},
                {{Spectral::Quadrature::GaussLobatto,
                  Spectral::Quadrature::SphericalSymmetry,
                  Spectral::Quadrature::SphericalSymmetry}}}}) {
    CAPTURE(dg_mesh);
    TestHelpers::check_pointwise_on_device<
        grmhd::ValenciaDivClean::ComputeSources>(
        &grmhd::ValenciaDivClean::ComputeSources::apply<DataVector>,
        grmhd::ValenciaDivClean::ComputeSources::return_tags{},
        grmhd::ValenciaDivClean::ComputeSources::argument_tags{}, 5, 0.1, 1.0,
        tuples::TaggedTuple<domain::Tags::Mesh<3>>{dg_mesh});
  }
}
