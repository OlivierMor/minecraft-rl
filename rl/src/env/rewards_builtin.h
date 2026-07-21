#pragma once
#include <memory>

#include "config.h"
#include "env/interfaces.h"

namespace rl {

class HitReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult& r, int p) override {
        return r.serverHit[p] ? 1.0f : 0.0f;
    }
};

class CritReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult& r, int p) override {
        return r.crit[p] ? 1.0f : 0.0f;
    }
};

class HitTakenReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult& r, int p) override {
        return r.serverHit[1 - p] ? 1.0f : 0.0f;
    }
};

class WinLossReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult&, int) override { return 0.0f; }
    float terminal(const mc::Sim&, int p, int winner, bool) override {
        if (winner < 0) return 0.0f;
        return winner == p ? 1.0f : -1.0f;
    }
};

class TimeReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult&, int) override { return 1.0f; }
};

class DrawReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult&, int) override { return 0.0f; }
    float terminal(const mc::Sim&, int, int winner, bool) override {
        return winner < 0 ? 1.0f : 0.0f;
    }
};

class JumpReward : public RewardFunction {
public:
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    int prev_[2] = {0, 0};
    float fired_[2] = {0, 0};
    int lastTick_ = -1;
};

class SneakReward : public RewardFunction {
public:
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override {
        return sim.client[p].self.sneaking ? 1.0f : 0.0f;
    }
};

class StrafeReward : public RewardFunction {
public:
    explicit StrafeReward(const Config& cfg)
        : maxDist_((float)cfg.num("reward.strafe.max_dist", 6.0)) {}
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    float maxDist_;
};

class FightClockReward : public RewardFunction {
public:
    explicit FightClockReward(const Config& cfg)
        : window_((int)cfg.num("reward.fight_clock.window_ticks", 40)) {}
    void reset(const mc::Sim& sim) override {
        prevHits_ = sim.sv[0].hits + sim.sv[1].hits;
        lastHitTick_ = 0;
        lastTick_ = -1;
        val_ = 0;
    }
    float step(const mc::Sim& sim, const mc::StepResult&, int) override {
        if (sim.tickCount != lastTick_) {
            int h = sim.sv[0].hits + sim.sv[1].hits;
            if (h != prevHits_) {
                prevHits_ = h;
                lastHitTick_ = sim.tickCount;
            }
            val_ = (sim.tickCount - lastHitTick_ > window_) ? 1.0f : 0.0f;
            lastTick_ = sim.tickCount;
        }
        return val_;
    }

private:
    int window_ = 40, prevHits_ = 0, lastHitTick_ = 0, lastTick_ = -1;
    float val_ = 0;
};

class FastWinReward : public RewardFunction {
public:
    explicit FastWinReward(const Config& cfg)
        : decay_((float)cfg.num("reward.fast_win.decay_ticks", 1200)) {}
    float step(const mc::Sim&, const mc::StepResult&, int) override { return 0.0f; }
    float terminal(const mc::Sim& sim, int p, int winner, bool) override {
        if (winner != p || decay_ <= 0) return 0.0f;
        float f = 1.0f - (float)sim.tickCount / decay_;
        return f > 0 ? f : 0.0f;
    }

private:
    float decay_;
};

class AimPressureReward : public RewardFunction {
public:
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    static float bearing(const mc::Sim& sim);
    float prev_ = 0, value_ = 0;
    int lastTick_ = -1;
};

class CursorPressureReward : public RewardFunction {
public:
    explicit CursorPressureReward(const Config& cfg);
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    float maxDist_, normDeg_;
};

class ApproachReward : public RewardFunction {
public:
    explicit ApproachReward(const Config& cfg);
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    float prev_[2] = {0, 0};
    int lastTick_ = -1;
    float delta_ = 0;
    float exponent_ = 1.0f, refSpeed_ = 0.3f;
    static float dist(const mc::Sim& sim);
};

class LookFarReward : public RewardFunction {
public:
    explicit LookFarReward(const Config& cfg);
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    float farMin_, farGate_, soft_;
};

class AimReward : public RewardFunction {
public:
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    static float phi(const mc::Sim& sim, int p);
    float prev_[2] = {0, 0}, delta_[2] = {0, 0};
    int lastTick_ = -1;
};

class ComboExpReward : public RewardFunction {
public:
    explicit ComboExpReward(const Config& cfg);
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult& r, int p) override;

private:
    float base_, cap_, minDamage_, mirror_;
    int prevHits_[2] = {0, 0}, streak_[2] = {0, 0};
    int lastTick_ = -1;
    float r_[2] = {0, 0};
};

class CritComboReward : public RewardFunction {
public:
    explicit CritComboReward(const Config& cfg);
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult& r, int p) override;

private:
    float base_, cap_, minDamage_, mirror_;
    int prevHits_[2] = {0, 0}, streak_[2] = {0, 0}, crits_[2] = {0, 0};
    int lastTick_ = -1;
    float r_[2] = {0, 0};
};

class AimJerkReward : public RewardFunction {
public:
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult&, int p) override;

private:
    float prevYaw_[2] = {0, 0}, prevPitch_[2] = {0, 0};
    float prevDYaw_[2] = {0, 0}, prevDPitch_[2] = {0, 0};
    int lastTick_ = -1;
    float r_[2] = {0, 0};
};

class MissedSwingReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult& r, int p) override {
        return r.swungAndMissed[p] ? 1.0f : 0.0f;
    }
};

class EarlySwingReward : public RewardFunction {
public:
    explicit EarlySwingReward(const Config& cfg)
        : minCharge_((float)cfg.num("reward.early_swing.min_charge", 0.9)) {}
    float step(const mc::Sim&, const mc::StepResult& r, int p) override {
        return (r.swingCharge[p] >= 0.0f && r.swingCharge[p] < minCharge_) ? 1.0f : 0.0f;
    }

private:
    float minCharge_;
};

class DamageDealtReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult& r, int p) override {
        return r.damageDealt[p];
    }
};

class DamageTakenReward : public RewardFunction {
public:
    float step(const mc::Sim&, const mc::StepResult& r, int p) override {
        return r.damageDealt[1 - p];
    }
};

class CombinedReward : public RewardFunction {
public:
    void add(std::string name, float weight, std::unique_ptr<RewardFunction> fn,
             bool shaped = false);
    void setShapingSchedule(std::vector<std::pair<double, double>> pts) {
        schedule_ = std::move(pts);
    }
    void onUpdate(long update) override;
    void reset(const mc::Sim& sim) override;
    float step(const mc::Sim& sim, const mc::StepResult& r, int p) override;
    float terminal(const mc::Sim& sim, int p, int winner, bool truncated) override;
    void collectStats(std::vector<std::pair<std::string, double>>& out) override;

private:
    struct Part {
        std::string name;
        float weight;
        bool shaped;
        std::unique_ptr<RewardFunction> fn;
        double sum = 0;
    };
    std::vector<Part> parts_;
    std::vector<std::pair<double, double>> schedule_;
    double scale_ = 1.0;
    long ticks_ = 0;
};

class ZeroSumReward : public RewardFunction {
public:
    explicit ZeroSumReward(std::unique_ptr<RewardFunction> inner) : inner_(std::move(inner)) {}
    void reset(const mc::Sim& sim) override { inner_->reset(sim); }
    float step(const mc::Sim& sim, const mc::StepResult& r, int p) override {
        return inner_->step(sim, r, p) - inner_->step(sim, r, 1 - p);
    }
    float terminal(const mc::Sim& sim, int p, int w, bool t) override {
        return inner_->terminal(sim, p, w, t) - inner_->terminal(sim, 1 - p, w, t);
    }
    void collectStats(std::vector<std::pair<std::string, double>>& out) override {
        inner_->collectStats(out);
    }
    void onUpdate(long update) override { inner_->onUpdate(update); }

private:
    std::unique_ptr<RewardFunction> inner_;
};

}
