#pragma once
#include <random>
#include "net.h"
#include "physics.h"

namespace mc1218 {

struct RemoteView {
    double posX = 0, posY = 0, posZ = 0;
    double prevPosX = 0, prevPosY = 0, prevPosZ = 0;
    float yaw = 0, pitch = 0;
    AABB bb;
    Pose pose = Pose::Standing;
    bool onGround = false;
    int hurtTime = 0;
    bool sprinting = false, sneaking = false;
    float health = Loadout::MAX_HEALTH;

    int lerpSteps = 0;
    double lerpX = 0, lerpY = 0, lerpZ = 0;
    float lerpYaw = 0, lerpPitch = 0;

    double baseX = 0, baseY = 0, baseZ = 0;

    double motionX = 0, motionY = 0, motionZ = 0;
    int lerpMotionSteps = 0;
    double lerpMotionX = 0, lerpMotionY = 0, lerpMotionZ = 0;

    int16_t gearDamage[5] = {0, 0, 0, 0, 0};
    bool gearBroken[5] = {false, false, false, false, false};

    bool swinging = false;
    int swingTime = 0;
};

struct ClientSim {
    Player self;
    RemoteView remote;

    double xLast = 0, yLast = 0, zLast = 0;
    float yRotLast = 0, xRotLast = 0;
    bool lastOnGround = false, lastHorizontalCollision = false;
    int positionReminder = 0;
    bool wasSprinting = false;
    Input lastSentInput;
    bool sentAnyInput = false;

    int missTime = 0;
    bool flashOnSetHealth = false;

    int knownHits[2] = {0, 0};
};

struct ServerPlayer {
    Player ent;

    double firstGoodX = 0, firstGoodY = 0, firstGoodZ = 0;
    double lastGoodX = 0, lastGoodY = 0, lastGoodZ = 0;
    Vec3 knownMovement;
    bool receivedMovementThisTick = false;

    float lastSentHealth = -1.0E8f;
    int lastSentFood = -99999999;
    bool lastFoodSaturationZero = false;
    int16_t lastEquipDamage[5] = {0, 0, 0, 0, 0};
    bool lastEquipBroken[5] = {false, false, false, false, false};
    int16_t lastSlotDamage[5] = {0, 0, 0, 0, 0};
    bool lastSlotBroken[5] = {false, false, false, false, false};

    bool metaDirty = false;
    bool attrDirty = false;
    bool hurtMarked = false;

    double codecBaseX = 0, codecBaseY = 0, codecBaseZ = 0;
    int8_t lastSentYRot = 0, lastSentXRot = 0;
    Vec3 lastSentMovement;
    int trackerTickCount = 0;
    int teleportDelay = 0;
    bool wasOnGround = false;

    int hits = 0;
};

struct StepResult {
    bool clientAttack[2] = {false, false};
    bool swungAndMissed[2] = {false, false};
    float swingCharge[2] = {-1.0f, -1.0f};
    bool kbReceived[2] = {false, false};
    Vec3 kb[2];
    bool hurtFlash[2] = {false, false};

    bool serverHit[2] = {false, false};
    bool serverFreshHit[2] = {false, false};
    float damageDealt[2] = {0, 0};
    bool crit[2] = {false, false};
    bool sprintHit[2] = {false, false};
    bool sweep[2] = {false, false};
    bool death[2] = {false, false};
    bool swordBroke[2] = {false, false};
    bool armorBroke[2] = {false, false};
};

class Sim {
public:
    static constexpr double SPAWN_DIST = 15.0;

    Arena arena;
    ClientSim client[2];
    ServerPlayer sv[2];
    int tickCount = 0;

    explicit Sim(uint64_t seed = 1) : rng_(seed) { reset(); }
    void reset();

    void setPingMs(int i, int ms) { pingMs_[i] = ms < 0 ? 0 : ms; }
    int pingMs(int i) const { return pingMs_[i]; }

    StepResult step(const Input& in0, const Input& in1);

    void place(int i, double x, double y, double z, float yaw, float pitch);
    void setMotion(int i, double mx, double my, double mz);
    void setHits(int i, int n);
    void setVitals(int i, float health, int food, float saturation);
    void reseed(uint64_t seed) { rng_.seed(seed); }

private:
    Channel c2s_[2], s2c_[2];
    int pingMs_[2] = {0, 0};
    bool scoreDirty_ = false;
    std::mt19937_64 rng_;

    int upDelay(int i) const   { int L = pingMs_[i] / 2; return (L + 49) / 50; }
    int downDelay(int i) const { int L = pingMs_[i] / 2; return L / 50 + 1; }
    void sendToServer(int i, const Pkt& p) { c2s_[i].send(tickCount, upDelay(i), p); }
    void sendToClient(int i, const Pkt& p) { s2c_[i].send(tickCount, downDelay(i), p); }

    void clientClick(int i, StepResult& r);
    void clientSendMovePackets(int i);
    void remoteTick(RemoteView& v);
    void handleClientPacket(int i, const Pkt& p, StepResult& r);

    void handleServerPacket(int i, const Pkt& p, StepResult& r);
    void serverHandleMove(int i, const Pkt& p);
    void serverAttack(int attacker, StepResult& r);
    bool serverHurt(int victim, int attacker, float amount, bool bypassArmor,
                    bool bypassEffects, bool noKnockback, float exhaustion,
                    StepResult& r, float* dealtOut,
                    bool* freshOut = nullptr);
    void trackerTick(int i);
    void serverConnectionTick(int i, StepResult& r);
    void sendMeta(int i);
    void sendAttributes(int i);
};

}
