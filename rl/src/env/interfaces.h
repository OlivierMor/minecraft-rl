#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "env/sim_api.h"

namespace rl {


struct ActionSpec {
    std::vector<int> branches;
    int numBranches() const { return (int)branches.size(); }
    int totalLogits() const {
        int s = 0;
        for (int b : branches) s += b;
        return s;
    }
};

class ActionParser {
public:
    virtual ~ActionParser() = default;
    virtual ActionSpec spec() const = 0;
    virtual void reset() {}
    virtual mc::Input parse(const int32_t* branches) = 0;
};


class ObsBuilder {
public:
    virtual ~ObsBuilder() = default;
    virtual int size() const = 0;
    virtual void reset(const mc::ClientSim& view) = 0;
    virtual void build(const mc::ClientSim& view, const mc::Input& lastAction,
                       int tick, int ownPingMs, float* out) = 0;
};

class CriticObsBuilder {
public:
    virtual ~CriticObsBuilder() = default;
    virtual int size() const = 0;
    virtual void reset(const mc::Sim& sim, int player) = 0;
    virtual void build(const mc::Sim& sim, int player, const mc::Input& lastAction,
                       float* out) = 0;
};


class RewardFunction {
public:
    virtual ~RewardFunction() = default;
    virtual void reset(const mc::Sim& sim) { (void)sim; }
    virtual float step(const mc::Sim& sim, const mc::StepResult& r, int player) = 0;
    virtual float terminal(const mc::Sim& sim, int player, int winner, bool truncated) {
        (void)sim; (void)player; (void)winner; (void)truncated;
        return 0.0f;
    }
    virtual void collectStats(std::vector<std::pair<std::string, double>>& out) { (void)out; }
    virtual void onUpdate(long update) { (void)update; }
};


enum class Verdict : uint8_t {
    Continue,
    Terminated,
    Truncated,
};

class TerminalCondition {
public:
    virtual ~TerminalCondition() = default;
    virtual void reset(const mc::Sim& sim) { (void)sim; }
    virtual Verdict check(const mc::Sim& sim, const mc::StepResult& r, int& winner) = 0;
};


struct SetupOp {
    enum Kind : uint8_t { Place, Motion, Ping, Hits, Seed } kind;
    int player = 0;
    double a = 0, b = 0, c = 0;
    float yaw = 0, pitch = 0;
    int ival = 0;
    uint64_t seed = 0;
};

class EnvHandle {
public:
    EnvHandle(mc::Sim& sim, std::vector<SetupOp>& rec, int envIndex, long episode)
        : envIndex(envIndex), episode(episode), sim_(sim), rec_(rec) {}

    const int envIndex;
    const long episode;

    void place(int i, double x, double y, double z, float yaw, float pitch) {
        sim_.place(i, x, y, z, yaw, pitch);
        rec_.push_back({SetupOp::Place, i, x, y, z, yaw, pitch, 0, 0});
    }
    void setMotion(int i, double mx, double my, double mz) {
        sim_.setMotion(i, mx, my, mz);
        rec_.push_back({SetupOp::Motion, i, mx, my, mz, 0, 0, 0, 0});
    }
    void setPingMs(int i, int ms) {
        sim_.setPingMs(i, ms);
        rec_.push_back({SetupOp::Ping, i, 0, 0, 0, 0, 0, ms, 0});
    }
    void setHits(int i, int n) {
        sim_.setHits(i, n);
        rec_.push_back({SetupOp::Hits, i, 0, 0, 0, 0, 0, n, 0});
    }
    void reseed(uint64_t seed) {
        sim_.reseed(seed);
        rec_.push_back({SetupOp::Seed, 0, 0, 0, 0, 0, 0, 0, seed});
    }
    const mc::Sim& sim() const { return sim_; }

    static void apply(mc::Sim& sim, const SetupOp& op) {
        switch (op.kind) {
            case SetupOp::Place:  sim.place(op.player, op.a, op.b, op.c, op.yaw, op.pitch); break;
            case SetupOp::Motion: sim.setMotion(op.player, op.a, op.b, op.c); break;
            case SetupOp::Ping:   sim.setPingMs(op.player, op.ival); break;
            case SetupOp::Hits:   sim.setHits(op.player, op.ival); break;
            case SetupOp::Seed:   sim.reseed(op.seed); break;
        }
    }

private:
    mc::Sim& sim_;
    std::vector<SetupOp>& rec_;
};

class StateSetter {
public:
    virtual ~StateSetter() = default;
    virtual void apply(EnvHandle& h, std::mt19937_64& rng) = 0;
};

}
