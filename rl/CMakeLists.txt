
set(RL_FLAGS -O3 -ffp-contract=off -Wall -Wextra)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-invalid-specialization HAS_WNO_INVALID_SPEC)
if(HAS_WNO_INVALID_SPEC)
    list(APPEND RL_FLAGS -Wno-invalid-specialization)
endif()
set(RL_FLAGS ${RL_FLAGS} PARENT_SCOPE)

add_library(rlcore STATIC
    src/config.cpp
    src/replay.cpp
    src/env/env.cpp
    src/env/registry.cpp
    src/env/standard_obs.cpp
    src/env/standard_actions.cpp
    src/env/rewards_builtin.cpp
    src/env/setters_builtin.cpp
    src/vec/worker_pool.cpp
    src/vec/vec_env.cpp
    src/model/policy.cpp
    src/model/distributions.cpp
    src/ppo/rollout.cpp
    src/ppo/learner.cpp
    src/selfplay/pool.cpp
    src/selfplay/elo.cpp
    src/selfplay/match.cpp
    src/runtime/checkpoint.cpp
    src/runtime/logger.cpp
)
target_include_directories(rlcore PUBLIC src)
target_compile_options(rlcore PRIVATE ${RL_FLAGS})
target_link_libraries(rlcore PUBLIC mcsim "${TORCH_LIBRARIES}")

