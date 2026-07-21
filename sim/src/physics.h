#pragma once
#include <cstdint>
#include <random>
#include "geom.h"
#include "items.h"
#include "world.h"

namespace mc1218 {

struct MathHelper {
    static float wrapAngleTo180_float(float f) { return Mth::wrapDegrees(f); }
    static float clamp_float(float f, float lo, float hi) { return Mth::clamp(f, lo, hi); }
    static double clamp_double(double d, double lo, double hi) { return Mth::clamp(d, lo, hi); }
};

struct Input {
    bool forward = false, back = false, left = false, right = false;
    bool jump = false, sneak = false;
    bool sprintKey = false;
    bool attack = false;
    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;
};

enum class Pose : uint8_t { Standing, Crouching };

struct FoodData {
    int foodLevel = 20;
    float saturationLevel = 5.0f;
    float exhaustionLevel = 0.0f;
    int tickTimer = 0;

    void addExhaustion(float f) { exhaustionLevel = std::min(exhaustionLevel + f, 40.0f); }
};

struct Player {
    static constexpr float WIDTH = 0.6f;
    static constexpr float HEIGHT_STANDING = 1.8f;
    static constexpr float EYE_STANDING = 1.62f;
    static constexpr float HEIGHT_CROUCHING = 1.5f;
    static constexpr float EYE_CROUCHING = 1.27f;

    double posX = 0, posY = 0, posZ = 0;
    double prevPosX = 0, prevPosY = 0, prevPosZ = 0;
    double motionX = 0, motionY = 0, motionZ = 0;
    float yaw = 0, pitch = 0;
    float prevYaw = 0, prevPitch = 0;
    double fallDistance = 0;
    bool onGround = false;
    bool collidedHorizontally = false;
    bool collidedVertically = false;
    bool verticalCollisionBelow = false;
    bool minorHorizontalCollision = false;
    bool hasImpulse = false;
    AABB bb;
    Pose pose = Pose::Standing;

    float xxa = 0, zza = 0;
    bool jumping = false;
    int jumpTicks = 0;
    int sprintTriggerTime = 0;
    bool sprinting = false;
    bool sneaking = false;
    bool shiftKeyDown = false;
    bool speedModSprint = false;
    float speed = 0.1f;
    bool prevShift = false;
    bool prevForwardImpulse = false;

    float health = Loadout::MAX_HEALTH;
    int hurtResistantTime = 0;
    int hurtTime = 0;
    float lastHurt = 0.0f;
    int attackStrengthTicker = 0;
    bool dead = false;
    FoodData food;
    Equipment gear;

    bool swinging = false;
    int swingTime = 0;

    float height() const { return pose == Pose::Crouching ? HEIGHT_CROUCHING : HEIGHT_STANDING; }
    float eyeHeight() const { return pose == Pose::Crouching ? EYE_CROUCHING : EYE_STANDING; }
    Vec3 eyePos() const { return {posX, posY + (double)eyeHeight(), posZ}; }
    Vec3 position() const { return {posX, posY, posZ}; }
    Vec3 deltaMovement() const { return {motionX, motionY, motionZ}; }
    void setDeltaMovement(double x, double y, double z) { motionX = x; motionY = y; motionZ = z; }

    float attackStrengthScale(float partial) const {
        return Mth::clamp(((float)attackStrengthTicker + partial) / Loadout::attackStrengthDelay(),
                          0.0f, 1.0f);
    }

    double movementSpeedAttribute() const {
        double d = Loadout::MOVEMENT_SPEED_BASE;
        if (speedModSprint) d *= 1.0 + Loadout::SPRINT_SPEED_MODIFIER;
        return d;
    }
    void setSprinting(bool b) {
        sprinting = b;
        speedModSprint = b;
    }
};

AABB makePlayerBB(double x, double y, double z, Pose pose);
void setPosition(Player& p, double x, double y, double z);

void applyLook(Player& p, float yawDelta, float pitchDelta);

void jumpFromGround(Player& p);

void clientPlayerTick(const Arena& arena, Player& p, const Input& in);

bool serverPlayerAiStep(const Arena& arena, Player& e, Player& other);

void moveEntity(const Arena& arena, Player& p, Vec3 vec, bool localAuthority);

void knockback(Player& p, double strength, double dx, double dz, std::mt19937_64& rng);

void updatePlayerPose(const Arena& arena, Player& p);

void foodTick(Player& p);

bool pointedAtEntity(const Arena& arena, const Vec3& eye, float pitch, float yaw,
                     const AABB& targetBB);

Vec3 getViewVector(float pitch, float yaw);
inline Vec3 getVectorForRotation(float pitch, float yaw) { return getViewVector(pitch, yaw); }

float getDamageAfterAbsorb(float damage, float armor, float toughness);
float getDamageAfterMagicAbsorb(float damage, float epf);

bool hurtAndBreak(ItemDur& item, int points, float ignoreChance, std::mt19937_64& rng);

void updateSwingTime(Player& p);
void startSwing(Player& p);

}
