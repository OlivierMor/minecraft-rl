#pragma once
#include "config.h"
#include "env/interfaces.h"

namespace rl {

class DefaultStateSetter : public StateSetter {
public:
    explicit DefaultStateSetter(const Config& cfg);
    void apply(EnvHandle& h, std::mt19937_64& rng) override;

private:
    int ping0_, ping1_;
};

class RandomStateSetter : public StateSetter {
public:
    explicit RandomStateSetter(const Config& cfg);
    void apply(EnvHandle& h, std::mt19937_64& rng) override;

private:
    double margin_, minDist_, maxDist_;
    int pingMin_, pingMax_;
    double pingPow_ = 1.0;
    std::vector<std::pair<double, double>> pingQ_;
    bool faceEachOther_;
    double yawJitter_ = 25.0;
};

}
