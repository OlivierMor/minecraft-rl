#include "env/rewards_builtin.h"

#include <cmath>

namespace rl {


void JumpReward::reset(const mc::Sim& sim) {
    prev_[0] = sim.client[0].self.jumpTicks;
    prev_[1] = sim.client[1].self.jumpTicks;
    fired_[0] = fired_[1] = 0;
    lastTick_ = -1;
}

float JumpReward::step(const mc::Sim& sim, const mc::StepResult&, int p) {
    if (sim.tickCount != lastTick_) {
        for (int i = 0; i < 2; ++i) {
            int now = sim.client[i].self.jumpTicks;
            fired_[i] = now > prev_[i] ? 1.0f : 0.0f;
            prev_[i] = now;
        }
        lastTick_ = sim.tickCount;
    }
    return fired_[p];
}


float StrafeReward::step(const mc::Sim& sim, const mc::StepResult&, int p) {
    const mc::Player& me = sim.client[p].self;
    const mc::Player& op = sim.client[1 - p].self;
    double dx = op.posX - me.posX, dz = op.posZ - me.posZ;
    double d = std::sqrt(dx * dx + dz * dz);
    if (d < 1e-6 || d > (double)maxDist_) return 0.0f;
    double vx = me.posX - me.prevPosX, vz = me.posZ - me.prevPosZ;
    float lat = (float)(std::fabs(vx * dz - vz * dx) / d) / 0.25f;
    return lat > 1.0f ? 1.0f : lat;
}


float AimPressureReward::bearing(const mc::Sim& sim) {
    const mc::Player& a = sim.client[0].self;
    const mc::Player& b = sim.client[1].self;
    return (float)(std::atan2(b.posZ - a.posZ, b.posX - a.posX) * 180.0 / M_PI);
}

void AimPressureReward::reset(const mc::Sim& sim) {
    prev_ = bearing(sim);
    value_ = 0;
    lastTick_ = -1;
}

float AimPressureReward::step(const mc::Sim& sim, const mc::StepResult&, int) {
    if (sim.tickCount != lastTick_) {
        const mc::Player& a = sim.client[0].self;
        const mc::Player& b = sim.client[1].self;
        double dx = b.posX - a.posX, dz = b.posZ - a.posZ;
        float now = bearing(sim);
        float d = std::fabs(mc::MathHelper::wrapAngleTo180_float(now - prev_));
        prev_ = now;
        value_ = (dx * dx + dz * dz < 0.25) ? 0.0f : (d > 15.0f ? 1.0f : d / 15.0f);
        lastTick_ = sim.tickCount;
    }
    return value_;
}


CursorPressureReward::CursorPressureReward(const Config& cfg) {
    maxDist_ = (float)cfg.num("reward.cursor_pressure.max_dist", 6.0);
    normDeg_ = (float)cfg.num("reward.cursor_pressure.norm_deg", 15.0);
    if (normDeg_ < 1e-3f) normDeg_ = 1e-3f;
}

float CursorPressureReward::step(const mc::Sim& sim, const mc::StepResult&, int p) {
    const mc::Player& me = sim.client[p].self;
    const mc::Player& op = sim.client[1 - p].self;
    if (!me.onGround) return 0.0f;
    double dx = op.posX - me.posX, dz = op.posZ - me.posZ;
    double d2 = dx * dx + dz * dz;
    if (d2 < 0.25 || d2 > (double)maxDist_ * maxDist_) return 0.0f;
    double vx = me.posX - me.prevPosX, vz = me.posZ - me.prevPosZ;
    float deg = (float)(std::fabs(vx * dz - vz * dx) / d2 * 180.0 / M_PI);
    float v = deg / normDeg_;
    return v > 1.0f ? 1.0f : v;
}


float ApproachReward::dist(const mc::Sim& sim) {
    const mc::Player& a = sim.client[0].self;
    const mc::Player& b = sim.client[1].self;
    double dx = a.posX - b.posX, dz = a.posZ - b.posZ;
    return (float)std::sqrt(dx * dx + dz * dz);
}

ApproachReward::ApproachReward(const Config& cfg) {
    exponent_ = (float)cfg.num("reward.approach.exponent", 1.0);
    refSpeed_ = (float)cfg.num("reward.approach.ref_speed", 0.3);
    if (refSpeed_ < 1e-4f) refSpeed_ = 1e-4f;
}

void ApproachReward::reset(const mc::Sim& sim) {
    prev_[0] = prev_[1] = dist(sim);
    lastTick_ = -1;
    delta_ = 0;
}

float ApproachReward::step(const mc::Sim& sim, const mc::StepResult&, int) {
    if (sim.tickCount != lastTick_) {
        float d = dist(sim);
        float closed = prev_[0] - d;
        if (exponent_ == 1.0f) {
            delta_ = closed;
        } else {
            float norm = closed / refSpeed_;
            float mag = std::pow(std::fabs(norm), exponent_) * refSpeed_;
            delta_ = norm >= 0.0f ? mag : -mag;
        }
        prev_[0] = d;
        lastTick_ = sim.tickCount;
    }
    return delta_;
}


LookFarReward::LookFarReward(const Config& cfg) {
    farMin_ = (float)cfg.num("reward.look_far.far_min", 5.0);
    farGate_ = (float)cfg.num("reward.look_far.far_gate", 2.0);
    soft_ = (float)cfg.num("reward.look_far.soft_deg", 15.0);
    if (farGate_ < 1e-3f) farGate_ = 1e-3f;
    if (soft_ < 1e-3f) soft_ = 1e-3f;
}

float LookFarReward::step(const mc::Sim& sim, const mc::StepResult&, int p) {
    const mc::Player& me = sim.client[p].self;
    const mc::RemoteView& op = sim.client[p].remote;
    double dx = op.posX - me.posX, dz = op.posZ - me.posZ;
    float hdist = (float)std::sqrt(dx * dx + dz * dz);
    if (hdist <= farMin_) return 0.0f;
    float gate = (hdist - farMin_) / farGate_;
    if (gate > 1.0f) gate = 1.0f;

    const mc::AABB& box = op.bb;
    mc::Vec3 eye = me.eyePos();
    mc::Vec3 look = mc::getVectorForRotation(me.pitch, me.yaw);
    mc::Vec3 ctr = box.getCenter();
    double range = std::sqrt(mc::Mth::lengthSquared(ctr.x - eye.x, ctr.y - eye.y, ctr.z - eye.z));
    if (range < 1e-6) return 0.0f;

    mc::Vec3 end{eye.x + look.x * (range + 4.0), eye.y + look.y * (range + 4.0),
                 eye.z + look.z * (range + 4.0)};
    if (box.clip(eye, end)) return gate;

    mc::Vec3 at{eye.x + look.x * range, eye.y + look.y * range, eye.z + look.z * range};
    double miss = std::sqrt(box.distanceToSqr(at));
    float missDeg = (float)(std::atan2(miss, range) * 180.0 / M_PI);
    if (missDeg >= soft_) return 0.0f;
    return gate * (soft_ - missDeg) / soft_;
}


float AimReward::phi(const mc::Sim& sim, int p) {
    const mc::Player& me = sim.client[p].self;
    const mc::RemoteView& op = sim.client[p].remote;
    const mc::AABB& box = op.bb;
    mc::Vec3 ctr = box.getCenter();
    double dx = ctr.x - me.posX, dz = ctr.z - me.posZ;
    double hdist = std::sqrt(dx * dx + dz * dz);
    double eyeY = me.posY + (double)me.eyeHeight();
    float bearing = (float)(std::atan2(dz, dx) * 180.0 / M_PI) - 90.0f;
    float yawErr = std::fabs(mc::MathHelper::wrapAngleTo180_float(bearing - me.yaw));
    float targetPitch = (float)(-(std::atan2(ctr.y - eyeY, hdist) * 180.0 / M_PI));
    float pitchErr = std::fabs(targetPitch - me.pitch);
    double halfW = (box.maxX - box.minX) * 0.5;
    double halfH = (box.maxY - box.minY) * 0.5;
    float yawHalf = (float)(std::atan2(halfW, hdist) * 180.0 / M_PI);
    float pitchHalf = (float)(std::atan2(halfH, hdist) * 180.0 / M_PI);
    yawErr = std::max(0.0f, yawErr - yawHalf);
    pitchErr = std::max(0.0f, pitchErr - pitchHalf);
    return -(yawErr / 180.0f + pitchErr / 90.0f) * 0.5f;
}

void AimReward::reset(const mc::Sim& sim) {
    prev_[0] = phi(sim, 0);
    prev_[1] = phi(sim, 1);
    delta_[0] = delta_[1] = 0;
    lastTick_ = -1;
}

float AimReward::step(const mc::Sim& sim, const mc::StepResult&, int p) {
    if (sim.tickCount != lastTick_) {
        for (int i = 0; i < 2; ++i) {
            float f = phi(sim, i);
            delta_[i] = f - prev_[i];
            prev_[i] = f;
        }
        lastTick_ = sim.tickCount;
    }
    return delta_[p];
}


void AimJerkReward::reset(const mc::Sim& sim) {
    for (int i = 0; i < 2; ++i) {
        prevYaw_[i] = sim.client[i].self.yaw;
        prevPitch_[i] = sim.client[i].self.pitch;
        prevDYaw_[i] = prevDPitch_[i] = 0;
        r_[i] = 0;
    }
    lastTick_ = -1;
}

float AimJerkReward::step(const mc::Sim& sim, const mc::StepResult&, int p) {
    if (sim.tickCount != lastTick_) {
        for (int i = 0; i < 2; ++i) {
            const mc::Player& me = sim.client[i].self;
            float dy = mc::MathHelper::wrapAngleTo180_float(me.yaw - prevYaw_[i]);
            float dp = me.pitch - prevPitch_[i];
            prevYaw_[i] = me.yaw;
            prevPitch_[i] = me.pitch;
            float ay = std::fabs(dy - prevDYaw_[i]);
            float ap = std::fabs(dp - prevDPitch_[i]);
            prevDYaw_[i] = dy;
            prevDPitch_[i] = dp;
            r_[i] = std::min(ay / 45.0f, 1.0f) + std::min(ap / 45.0f, 1.0f);
        }
        lastTick_ = sim.tickCount;
    }
    return r_[p];
}


ComboExpReward::ComboExpReward(const Config& cfg) {
    base_ = (float)cfg.num("reward.combo_exp.base", 1.6);
    cap_ = (float)cfg.num("reward.combo_exp.cap", 8.0);
    minDamage_ = (float)cfg.num("reward.combo_exp.min_damage", 0.45);
    mirror_ = (float)cfg.num("reward.combo_exp.mirror", 1.0);
    if (base_ < 1.0f) base_ = 1.0f;
}

void ComboExpReward::reset(const mc::Sim& sim) {
    prevHits_[0] = sim.sv[0].hits;
    prevHits_[1] = sim.sv[1].hits;
    streak_[0] = streak_[1] = 0;
    lastTick_ = -1;
    r_[0] = r_[1] = 0;
}

float ComboExpReward::step(const mc::Sim& sim, const mc::StepResult& r, int p) {
    if (sim.tickCount != lastTick_) {
        r_[0] = r_[1] = 0;
        for (int i = 0; i < 2; ++i) {
            int d = sim.sv[i].hits - prevHits_[i];
            if (d <= 0) continue;
            prevHits_[i] = sim.sv[i].hits;
            streak_[1 - i] = 0;
            if (r.damageDealt[i] < minDamage_) continue;
            streak_[i] += 1;
            float bonus = std::min(std::pow(base_, (float)(streak_[i] - 1)) - 1.0f, cap_);
            r_[i] += bonus;
            r_[1 - i] -= mirror_ * bonus;
        }
        lastTick_ = sim.tickCount;
    }
    return r_[p];
}


CritComboReward::CritComboReward(const Config& cfg) {
    base_ = (float)cfg.num("reward.crit_combo.base", 2.0);
    cap_ = (float)cfg.num("reward.crit_combo.cap", 12.0);
    minDamage_ = (float)cfg.num("reward.combo_exp.min_damage", 0.45);
    mirror_ = (float)cfg.num("reward.combo_exp.mirror", 1.0);
    if (base_ < 1.0f) base_ = 1.0f;
}

void CritComboReward::reset(const mc::Sim& sim) {
    prevHits_[0] = sim.sv[0].hits;
    prevHits_[1] = sim.sv[1].hits;
    streak_[0] = streak_[1] = 0;
    crits_[0] = crits_[1] = 0;
    lastTick_ = -1;
    r_[0] = r_[1] = 0;
}

float CritComboReward::step(const mc::Sim& sim, const mc::StepResult& r, int p) {
    if (sim.tickCount != lastTick_) {
        r_[0] = r_[1] = 0;
        for (int i = 0; i < 2; ++i) {
            int d = sim.sv[i].hits - prevHits_[i];
            if (d <= 0) continue;
            prevHits_[i] = sim.sv[i].hits;
            streak_[1 - i] = 0;
            crits_[1 - i] = 0;
            if (r.damageDealt[i] < minDamage_) continue;
            streak_[i] += 1;
            if (r.crit[i]) crits_[i] += 1;
            if (streak_[i] >= 2 && crits_[i] > 0) {
                float bonus = std::min(std::pow(base_, (float)crits_[i]) - 1.0f, cap_);
                r_[i] += bonus;
                r_[1 - i] -= mirror_ * bonus;
            }
        }
        lastTick_ = sim.tickCount;
    }
    return r_[p];
}


void CombinedReward::add(std::string name, float weight, std::unique_ptr<RewardFunction> fn,
                         bool shaped) {
    parts_.push_back({std::move(name), weight, shaped, std::move(fn), 0});
}

void CombinedReward::onUpdate(long update) {
    if (!schedule_.empty()) {
        double x = (double)update, px = 0, py = 1.0;
        bool first = true;
        scale_ = schedule_.back().second;
        for (auto& [cx, cy] : schedule_) {
            if (x <= cx) {
                scale_ = first ? cy : py + (cy - py) * (x - px) / (cx - px);
                break;
            }
            px = cx; py = cy; first = false;
        }
    }
    for (auto& p : parts_) p.fn->onUpdate(update);
}

void CombinedReward::reset(const mc::Sim& sim) {
    for (auto& p : parts_) p.fn->reset(sim);
}

float CombinedReward::step(const mc::Sim& sim, const mc::StepResult& r, int player) {
    float total = 0;
    for (auto& p : parts_) {
        float w = p.shaped ? (float)(p.weight * scale_) : p.weight;
        float v = w * p.fn->step(sim, r, player);
        p.sum += v;
        total += v;
    }
    ++ticks_;
    return total;
}

float CombinedReward::terminal(const mc::Sim& sim, int player, int winner, bool truncated) {
    float total = 0;
    for (auto& p : parts_) {
        float w = p.shaped ? (float)(p.weight * scale_) : p.weight;
        float v = w * p.fn->terminal(sim, player, winner, truncated);
        p.sum += v;
        total += v;
    }
    return total;
}

void CombinedReward::collectStats(std::vector<std::pair<std::string, double>>& out) {
    for (auto& p : parts_) {
        out.emplace_back("rew_" + p.name, ticks_ > 0 ? p.sum / (double)ticks_ : 0.0);
        p.sum = 0;
    }
    ticks_ = 0;
}

}
