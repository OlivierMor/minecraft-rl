#pragma once
#include <cmath>

#include "config.h"
#include "env/interfaces.h"

namespace rl {

class StandardObs : public ObsBuilder {
public:
    StandardObs(const Config& cfg, int side);
    int size() const override;
    void reset(const mc::ClientSim& view) override;
    void build(const mc::ClientSim& view, const mc::Input& lastAction,
               int tick, int ownPingMs, float* out) override;

    static constexpr int SNAPSHOTS = 11;
    static constexpr int FRAME = 4;
    static constexpr int OFFSETS[SNAPSHOTS] =
        {1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40};
    static constexpr int RING = 41;
    static constexpr int SIZE = 10 + 22 + 9 + 10 + SNAPSHOTS * FRAME;

private:
    void advance(const mc::Player& me, const mc::RemoteView& op, int tick);

    float ring_[RING][FRAME] = {};
    int head_ = -1;
    bool histInit_ = false;
    int lastTick_ = -1000;

    int prevSelfHurt_ = 0, prevRemoteHurt_ = 0;
    int sinceTaken_ = 999, sinceGiven_ = 999;
    int myCombo_ = 0, oppCombo_ = 0, lastScorer_ = -1;

    float prevOppYaw_ = 0, prevOppPitch_ = 0;
    float oppYawRate_ = 0, oppPitchRate_ = 0;
    float prevHdist_ = 0, closing_ = 0;
};

class StandardCriticObs : public CriticObsBuilder {
public:
    explicit StandardCriticObs(const Config& cfg);
    int size() const override;
    void reset(const mc::Sim& sim, int player) override;
    void build(const mc::Sim& sim, int player, const mc::Input& lastAction,
               float* out) override;

    static constexpr int TRUTH_FLOATS = 17;

private:
    StandardObs fair_[2];
    float hitNorm_;
};

}
