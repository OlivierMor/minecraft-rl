#include "env/env.h"

#include <stdexcept>

namespace rl {

Env::Env(int index, uint64_t seed, ComponentSet components)
    : sim(seed), index(index), rng(seed), comp_(std::move(components)) {
    if (!comp_.obs[0] || !comp_.obs[1] || !comp_.parser[0] || !comp_.parser[1] ||
        !comp_.reward || !comp_.setter || comp_.terminals.empty())
        throw std::runtime_error("Env: incomplete component set");
    reset();
}

void Env::reset() {
    sim.reset();
    setupOps.clear();
    EnvHandle h(sim, setupOps, index, episodes);
    comp_.setter->apply(h, rng);
    fight = FightStats{};
    for (int s = 0; s < 2; ++s) {
        lastInput[s] = mc::Input{};
        prevJumpTicks_[s] = sim.client[s].self.jumpTicks;
        prevHitsStat_[s] = sim.sv[s].hits;
        comboStreak_[s] = 0;
        comp_.obs[s]->reset(sim.client[s]);
        comp_.parser[s]->reset();
    }
    if (comp_.criticObs)
        for (int s = 0; s < 2; ++s) comp_.criticObs->reset(sim, s);
    comp_.reward->reset(sim);
    for (auto& t : comp_.terminals) t->reset(sim);
}

Env::StepOut Env::step(const mc::Input& in0, const mc::Input& in1) {
    mc::StepResult r = sim.step(in0, in1);
    lastInput[0] = in0;
    lastInput[1] = in1;

    for (int i = 0; i < 2; ++i) {
        if (r.crit[i]) ++fight.crits[i];
        int jt = sim.client[i].self.jumpTicks;
        if (jt > prevJumpTicks_[i]) ++fight.jumps[i];
        prevJumpTicks_[i] = jt;
        int d = sim.sv[i].hits - prevHitsStat_[i];
        if (d > 0) {
            prevHitsStat_[i] = sim.sv[i].hits;
            comboStreak_[1 - i] = 0;
            if (r.damageDealt[i] >= 0.45f) {
                if (++comboStreak_[i] > fight.maxCombo[i]) fight.maxCombo[i] = comboStreak_[i];
            }
        }
    }

    StepOut out;
    out.reward[0] = comp_.reward->step(sim, r, 0);
    out.reward[1] = comp_.reward->step(sim, r, 1);

    for (auto& t : comp_.terminals) {
        int w = -1;
        Verdict v = t->check(sim, r, w);
        if (v == Verdict::Terminated) {
            out.done = true;
            out.truncated = false;
            out.winner = w;
            break;
        }
        if (v == Verdict::Truncated) {
            out.done = true;
            out.truncated = true;
        }
    }
    if (out.done) {
        out.reward[0] += comp_.reward->terminal(sim, 0, out.winner, out.truncated);
        out.reward[1] += comp_.reward->terminal(sim, 1, out.winner, out.truncated);
        ++episodes;
    }
    return out;
}

}
