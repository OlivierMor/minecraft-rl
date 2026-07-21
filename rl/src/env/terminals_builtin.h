#pragma once
#include "config.h"
#include "env/interfaces.h"

namespace rl {

class DeathMatch : public TerminalCondition {
public:
    explicit DeathMatch(const Config&) {}
    Verdict check(const mc::Sim& sim, const mc::StepResult&, int& winner) override {
        bool d0 = sim.sv[0].ent.dead, d1 = sim.sv[1].ent.dead;
        if (!d0 && !d1) return Verdict::Continue;
        if (!(d0 && d1)) winner = d0 ? 1 : 0;
        return Verdict::Terminated;
    }
};

class FirstToHits : public TerminalCondition {
public:
    explicit FirstToHits(const Config& cfg)
        : hits_((int)cfg.num("terminal.first_to_hits.hits", 50)) {}
    Verdict check(const mc::Sim& sim, const mc::StepResult&, int& winner) override {
        bool w0 = sim.sv[0].hits >= hits_, w1 = sim.sv[1].hits >= hits_;
        if (!w0 && !w1) return Verdict::Continue;
        if (w0 && w1 && sim.sv[0].hits == sim.sv[1].hits) return Verdict::Terminated;
        winner = (sim.sv[0].hits >= sim.sv[1].hits) ? 0 : 1;
        return Verdict::Terminated;
    }

private:
    int hits_;
};

class TickCap : public TerminalCondition {
public:
    explicit TickCap(const Config& cfg)
        : cap_((int)cfg.num("terminal.tick_cap.ticks", 6000)) {}
    Verdict check(const mc::Sim& sim, const mc::StepResult&, int&) override {
        return sim.tickCount >= cap_ ? Verdict::Truncated : Verdict::Continue;
    }

private:
    int cap_;
};

class NoHitTimeout : public TerminalCondition {
public:
    explicit NoHitTimeout(const Config& cfg)
        : window_((int)cfg.num("terminal.no_hit_timeout.ticks", 1200)) {}
    void reset(const mc::Sim& sim) override { lastHitTick_ = sim.tickCount; }
    Verdict check(const mc::Sim& sim, const mc::StepResult& r, int&) override {
        if (r.serverHit[0] || r.serverHit[1]) lastHitTick_ = sim.tickCount;
        return sim.tickCount - lastHitTick_ >= window_ ? Verdict::Truncated : Verdict::Continue;
    }

private:
    int window_;
    int lastHitTick_ = 0;
};

}
