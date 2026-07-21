set(SIM_FLAGS -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra)

add_library(mcsim STATIC src/physics.cpp src/sim.cpp)
target_include_directories(mcsim PUBLIC src)
target_compile_options(mcsim PRIVATE ${SIM_FLAGS})
