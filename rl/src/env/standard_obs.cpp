#include "env/standard_obs.h"

#include <cmath>

namespace rl {


namespace {

constexpr float DEG2RAD = 3.14159265358979323846f / 180.0f;
constexpr float RAD2DEG = 180.0f / 3.14159265358979323846f;

inline void egoFrame(float yawDeg, double dx, double dz, float& right, float& fwd) {
    float s = std::sin(yawDeg * DEG2RAD), c = std::cos(yawDeg * DEG2RAD);
    right = (float)(dx * c + dz * s);
    fwd = (float)(-dx * s + dz * c);
}

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

}

constexpr int StandardObs::OFFSETS[StandardObs::SNAPSHOTS];

StandardObs::StandardObs(const Config& cfg, int side) {
    (void)cfg;
    (void)side;
}

int StandardObs::size() const { return SIZE; }

void StandardObs::reset(const mc::ClientSim& view) {
    histInit_ = false;
    head_ = -1;
    lastTick_ = -1000;
    prevSelfHurt_ = view.self.hurtTime;
    prevRemoteHurt_ = view.remote.hurtTime;
    sinceTaken_ = sinceGiven_ = 999;
    myCombo_ = oppCombo_ = 0;
    lastScorer_ = -1;
    prevOppYaw_ = view.remote.yaw;
    prevOppPitch_ = view.remote.pitch;
    oppYawRate_ = oppPitchRate_ = 0;
    {
        double dx = view.remote.posX - view.self.posX;
        double dz = view.remote.posZ - view.self.posZ;
        prevHdist_ = (float)std::sqrt(dx * dx + dz * dz);
    }
    closing_ = 0;
}

void StandardObs::advance(const mc::Player& me, const mc::RemoteView& op, int tick) {
    if (tick == lastTick_) return;
    lastTick_ = tick;

    bool gaveHit = op.hurtTime > prevRemoteHurt_;
    bool tookHit = me.hurtTime > prevSelfHurt_;
    if (tookHit) sinceTaken_ = 0;
    else if (sinceTaken_ < 999) ++sinceTaken_;
    if (gaveHit) sinceGiven_ = 0;
    else if (sinceGiven_ < 999) ++sinceGiven_;
    prevSelfHurt_ = me.hurtTime;
    prevRemoteHurt_ = op.hurtTime;
    if (gaveHit) {
        myCombo_ = (lastScorer_ == 0) ? myCombo_ + 1 : 1;
        oppCombo_ = 0;
        lastScorer_ = 0;
    }
    if (tookHit) {
        oppCombo_ = (lastScorer_ == 1) ? oppCombo_ + 1 : 1;
        if (!gaveHit) myCombo_ = 0;
        lastScorer_ = 1;
    }

    oppYawRate_ = mc::MathHelper::wrapAngleTo180_float(op.yaw - prevOppYaw_);
    oppPitchRate_ = op.pitch - prevOppPitch_;
    prevOppYaw_ = op.yaw;
    prevOppPitch_ = op.pitch;
    double dx = op.posX - me.posX, dz = op.posZ - me.posZ;
    float hdist = (float)std::sqrt(dx * dx + dz * dz);
    closing_ = prevHdist_ - hdist;
    prevHdist_ = hdist;

    float rr, rf, ovx, ovz;
    egoFrame(me.yaw, dx, dz, rr, rf);
    ovx = (float)(op.posX - op.prevPosX);
    ovz = (float)(op.posZ - op.prevPosZ);
    float cur[FRAME] = {clampf(rr / 10.0f, -3, 3), clampf(rf / 10.0f, -3, 3),
                        clampf((float)(op.posY - me.posY) / 3.0f, -2, 2),
                        clampf(std::sqrt(ovx * ovx + ovz * ovz) / 0.3f, 0, 3)};
    if (!histInit_) {
        for (auto& fr : ring_)
            for (int c = 0; c < FRAME; ++c) fr[c] = cur[c];
        head_ = 0;
        histInit_ = true;
    } else {
        head_ = (head_ + 1) % RING;
        for (int c = 0; c < FRAME; ++c) ring_[head_][c] = cur[c];
    }
}

void StandardObs::build(const mc::ClientSim& view, const mc::Input& la,
                        int tick, int ownPingMs, float* o) {
    (void)ownPingMs;
    const mc::Player& me = view.self;
    const mc::RemoteView& op = view.remote;
    advance(me, op, tick);
    int k = 0;

    float vr, vf;
    egoFrame(me.yaw, me.posX - me.prevPosX, me.posZ - me.prevPosZ, vr, vf);
    o[k++] = clampf(vr / 0.3f, -3, 3);
    o[k++] = clampf(vf / 0.3f, -3, 3);
    o[k++] = clampf((float)(me.posY - me.prevPosY) / 0.5f, -3, 3);
    o[k++] = me.pitch / 90.0f;
    o[k++] = me.onGround ? 1.f : 0.f;
    o[k++] = me.sprinting ? 1.f : 0.f;
    o[k++] = me.collidedHorizontally ? 1.f : 0.f;
    o[k++] = clampf((float)me.fallDistance / 3.0f, 0, 2);
    o[k++] = me.hurtTime / 10.0f;
    o[k++] = me.jumpTicks / 10.0f;

    double dx = op.posX - me.posX, dy = op.posY - me.posY, dz = op.posZ - me.posZ;
    float rr, rf;
    egoFrame(me.yaw, dx, dz, rr, rf);
    float hdist = std::sqrt((float)(dx * dx + dz * dz));
    o[k++] = clampf(rr / 10.0f, -3, 3);
    o[k++] = clampf(rf / 10.0f, -3, 3);
    o[k++] = clampf((float)dy / 3.0f, -2, 2);
    o[k++] = clampf(hdist / 10.0f, 0, 4);
    float ovr, ovf;
    egoFrame(me.yaw, op.posX - op.prevPosX, op.posZ - op.prevPosZ, ovr, ovf);
    o[k++] = clampf(ovr / 0.3f, -3, 3);
    o[k++] = clampf(ovf / 0.3f, -3, 3);
    o[k++] = clampf((float)(op.posY - op.prevPosY) / 0.5f, -3, 3);
    o[k++] = clampf(closing_ / 0.3f, -3, 3);
    float relYaw = mc::MathHelper::wrapAngleTo180_float(op.yaw - me.yaw);
    o[k++] = std::sin(relYaw * DEG2RAD);
    o[k++] = std::cos(relYaw * DEG2RAD);
    o[k++] = op.pitch / 90.0f;
    o[k++] = clampf(oppYawRate_ / 45.0f, -2, 2);
    o[k++] = clampf(oppPitchRate_ / 45.0f, -2, 2);
    o[k++] = op.onGround ? 1.f : 0.f;
    o[k++] = op.sprinting ? 1.f : 0.f;
    o[k++] = op.sneaking ? 1.f : 0.f;
    o[k++] = op.hurtTime / 10.0f;
    o[k++] = op.swinging ? 1.0f : 0.0f;
    double eyeY = me.posY + (double)me.eyeHeight();
    float bearing = (float)(std::atan2(dz, dx) * RAD2DEG) - 90.0f;
    float dyawAim = mc::MathHelper::wrapAngleTo180_float(bearing - me.yaw);
    float targetPitch = (float)(-(std::atan2(op.posY + 0.9 - eyeY, (double)hdist) * RAD2DEG));
    o[k++] = dyawAim / 180.0f;
    o[k++] = clampf((targetPitch - me.pitch) / 90.0f, -2, 2);
    o[k++] = (hdist < 3.0f) ? 1.f : 0.f;
    o[k++] = clampf((3.0f - hdist) / 3.0f, -2, 1);

    o[k++] = me.health / 20.0f;
    o[k++] = op.health / 20.0f;
    o[k++] = (float)me.food.foodLevel / 20.0f;
    o[k++] = me.food.saturationLevel / 20.0f;
    o[k++] = me.attackStrengthScale(0.5f);
    o[k++] = clampf(sinceTaken_ / 60.0f, 0, 2);
    o[k++] = clampf(sinceGiven_ / 60.0f, 0, 2);
    o[k++] = clampf(myCombo_ / 5.0f, 0, 2);
    o[k++] = clampf(oppCombo_ / 5.0f, 0, 2);

    o[k++] = la.forward ? 1.f : 0.f;
    o[k++] = la.back ? 1.f : 0.f;
    o[k++] = la.left ? 1.f : 0.f;
    o[k++] = la.right ? 1.f : 0.f;
    o[k++] = la.jump ? 1.f : 0.f;
    o[k++] = la.sneak ? 1.f : 0.f;
    o[k++] = la.sprintKey ? 1.f : 0.f;
    o[k++] = la.attack ? 1.f : 0.f;
    o[k++] = clampf(la.yawDelta / 45.0f, -2, 2);
    o[k++] = clampf(la.pitchDelta / 45.0f, -2, 2);

    for (int s = 0; s < SNAPSHOTS; ++s) {
        int idx = ((head_ - OFFSETS[s]) % RING + RING) % RING;
        for (int c = 0; c < FRAME; ++c) o[k++] = ring_[idx][c];
    }
}


StandardCriticObs::StandardCriticObs(const Config& cfg)
    : fair_{StandardObs(cfg, 0), StandardObs(cfg, 1)} {
    hitNorm_ = (float)cfg.num("standard_obs.hit_norm", cfg.num("terminal.first_to_hits.hits", 50));
    if (hitNorm_ <= 0) hitNorm_ = 50;
}

int StandardCriticObs::size() const { return fair_[0].size() + TRUTH_FLOATS; }

void StandardCriticObs::reset(const mc::Sim& sim, int player) {
    fair_[player].reset(sim.client[player]);
}

void StandardCriticObs::build(const mc::Sim& sim, int player, const mc::Input& la,
                              float* out) {
    fair_[player].build(sim.client[player], la, sim.tickCount, sim.pingMs(player), out);
    float* o = out + fair_[player].size();
    const mc::Player& me = sim.client[player].self;
    const mc::Player& op = sim.client[1 - player].self;
    int k = 0;
    double dx = op.posX - me.posX, dy = op.posY - me.posY, dz = op.posZ - me.posZ;
    float rr, rf, ovr, ovf;
    egoFrame(me.yaw, dx, dz, rr, rf);
    egoFrame(me.yaw, op.posX - op.prevPosX, op.posZ - op.prevPosZ, ovr, ovf);
    o[k++] = clampf(rr / 10.0f, -3, 3);
    o[k++] = clampf(rf / 10.0f, -3, 3);
    o[k++] = clampf((float)dy / 3.0f, -2, 2);
    o[k++] = clampf(ovr / 0.3f, -3, 3);
    o[k++] = clampf(ovf / 0.3f, -3, 3);
    o[k++] = clampf((float)(op.posY - op.prevPosY) / 0.5f, -3, 3);
    const mc::RemoteView& theirViewOfMe = sim.client[1 - player].remote;
    o[k++] = clampf((float)(theirViewOfMe.posX - me.posX) / 1.0f, -3, 3);
    o[k++] = clampf((float)(theirViewOfMe.posY - me.posY) / 1.0f, -3, 3);
    o[k++] = clampf((float)(theirViewOfMe.posZ - me.posZ) / 1.0f, -3, 3);
    o[k++] = sim.sv[1 - player].ent.hurtResistantTime / 20.0f;
    o[k++] = sim.sv[player].ent.hurtResistantTime / 20.0f;
    o[k++] = sim.sv[player].ent.sprinting ? 1.f : 0.f;
    o[k++] = sim.sv[1 - player].ent.sprinting ? 1.f : 0.f;
    o[k++] = clampf(sim.sv[player].hits / hitNorm_, 0, 1.5f);
    o[k++] = clampf(sim.sv[1 - player].hits / hitNorm_, 0, 1.5f);
    o[k++] = sim.sv[player].ent.health / 20.0f;
    o[k++] = sim.sv[1 - player].ent.health / 20.0f;
}


}
