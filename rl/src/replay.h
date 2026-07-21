#pragma once
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "env/interfaces.h"

namespace rl {

struct Replay {
    std::vector<SetupOp> setup;
    std::vector<std::array<mc::Input, 2>> ticks;
    std::vector<std::array<int32_t, 2>> pings;

    void applySetup(mc::Sim& sim) const {
        sim.reset();
        for (const SetupOp& op : setup) EnvHandle::apply(sim, op);
    }

    bool save(const std::string& path) const;
    static std::optional<Replay> load(const std::string& path);
};

}
