#pragma once
#include <array>
#include <memory>

#include "config.h"
#include "env/interfaces.h"

namespace rl {

struct ComponentSet {
    std::unique_ptr<ObsBuilder> obs[2];
    std::unique_ptr<CriticObsBuilder> criticObs;
    std::unique_ptr<ActionParser> parser[2];
    std::unique_ptr<RewardFunction> reward;
    std::vector<std::unique_ptr<TerminalCondition>> terminals;
    std::unique_ptr<StateSetter> setter;
};

class Env {
public:
    Env(int index, uint64_t seed, ComponentSet components);

    void reset();

    struct StepOut {
        float reward[2] = {0, 0};
        bool done = false;
        bool truncated = false;
        int winner = -1;
    };
    StepOut step(const mc::Input& in0, const mc::Input& in1);

    void buildObs(int side, float* out) {
        comp_.obs[side]->build(sim.client[side], lastInput[side], sim.tickCount,
                               sim.pingMs(side), out);
    }
    void buildCriticObs(int side, float* out) {
        comp_.criticObs->build(sim, side, lastInput[side], out);
    }
    mc::Input parseAction(int side, const int32_t* branches) {
        return comp_.parser[side]->parse(branches);
    }

    int obsSize() const { return comp_.obs[0]->size(); }
    int criticObsSize() const { return comp_.criticObs ? comp_.criticObs->size() : 0; }
    ActionSpec spec() const { return comp_.parser[0]->spec(); }
    RewardFunction& rewardFn() { return *comp_.reward; }

    mc::Sim sim;
    const int index;
    long episodes = 0;
    mc::Input lastInput[2];
    std::vector<SetupOp> setupOps;
    std::mt19937_64 rng;

    struct FightStats {
        int crits[2] = {0, 0};
        int jumps[2] = {0, 0};
        int maxCombo[2] = {0, 0};
    };
    FightStats fight;

private:
    ComponentSet comp_;
    int prevJumpTicks_[2] = {0, 0};
    int prevHitsStat_[2] = {0, 0};
    int comboStreak_[2] = {0, 0};
};

}
