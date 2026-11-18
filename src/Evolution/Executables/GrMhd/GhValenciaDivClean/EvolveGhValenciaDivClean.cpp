// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/GrMhd/GhValenciaDivClean/EvolveGhValenciaDivClean.hpp"

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Parallel/CharmMain.tpp"
#include "PointwiseFunctions/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

// Parameters chosen in CMakeLists.txt
#if defined(USE_FD_PATCHES) and USE_FD_PATCHES
using metavariables = EvolutionMetavarsWithFdPatches<
    USE_CONTROL_SYSTEMS, USE_PARAMETRIZED_DELEPTONIZATION, BondiSachs>;
#else
using metavariables =
    EvolutionMetavars<USE_CONTROL_SYSTEMS, USE_PARAMETRIZED_DELEPTONIZATION,
                      BondiSachs>;
#endif

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm,
       &domain::creators::time_dependence::register_derived_with_charm,
       &domain::FunctionsOfTime::register_derived_with_charm,
       &grmhd::GhValenciaDivClean::fd::register_derived_with_charm<
           typename metavariables::base::system>,
       &EquationsOfState::register_derived_with_charm,
       &ConstraintDamping::register_derived_with_charm,
       &register_factory_classes_with_charm<metavariables>},
      {});
}
