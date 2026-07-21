#include "physics.h"

#include <algorithm>

namespace mc1218 {


AABB makePlayerBB(double x, double y, double z, Pose pose) {
    float w = Player::WIDTH;
    float h = pose == Pose::Crouching ? Player::HEIGHT_CROUCHING : Player::HEIGHT_STANDING;
    double f = (double)w / 2.0;
    return AABB::raw(x - f, y, z - f, x + f, y + (double)h, z + f);
}

void setPosition(Player& p, double x, double y, double z) {
    p.posX = x; p.posY = y; p.posZ = z;
    p.bb = makePlayerBB(x, y, z, p.pose);
}

void applyLook(Player& p, float yawDelta, float pitchDelta) {
    p.pitch = p.pitch + pitchDelta;
    p.yaw = p.yaw + yawDelta;
    p.pitch = Mth::clamp(p.pitch, -90.0f, 90.0f);
    p.prevPitch += pitchDelta;
    p.prevYaw += yawDelta;
    p.prevPitch = Mth::clamp(p.prevPitch, -90.0f, 90.0f);
}

Vec3 getViewVector(float pitch, float yaw) {
    float f = pitch * DEG_TO_RAD_F;
    float f2 = -yaw * DEG_TO_RAD_F;
    float f3 = Mth::cos(f2);
    float f4 = Mth::sin(f2);
    float f5 = Mth::cos(f);
    float f6 = Mth::sin(f);
    return {(double)(f4 * f5), (double)(-f6), (double)(f3 * f5)};
}

void jumpFromGround(Player& p) {
    float f = Loadout::JUMP_STRENGTH;
    if (f <= 1.0E-5f) return;
    p.motionY = std::max((double)f, p.motionY);
    if (p.sprinting) {
        float f2 = p.yaw * DEG_TO_RAD_F;
        p.motionX += (double)(-Mth::sin(f2)) * 0.2;
        p.motionZ += (double)Mth::cos(f2) * 0.2;
    }
    p.hasImpulse = true;
}

void knockback(Player& p, double strength, double dx, double dz, std::mt19937_64& rng) {
    strength *= 1.0 - Loadout::KNOCKBACK_RESISTANCE;
    if (strength <= 0.0) return;
    p.hasImpulse = true;
    Vec3 vec3 = p.deltaMovement();
    while (dx * dx + dz * dz < (double)1.0E-5f) {
        auto rand01 = [&] { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };
        dx = (rand01() - rand01()) * 0.01;
        dz = (rand01() - rand01()) * 0.01;
    }
    Vec3 vec32 = Vec3(dx, 0.0, dz).normalize().scale(strength);
    p.setDeltaMovement(vec3.x / 2.0 - vec32.x,
                       p.onGround ? std::min(0.4, vec3.y / 2.0 + strength) : vec3.y,
                       vec3.z / 2.0 - vec32.z);
}


static Vec3 collideWithShapes(const Vec3& vec, AABB box, const BoxList& shapes) {
    if (shapes.empty()) return vec;
    double dy = vec.y;
    if (dy != 0.0) {
        if (std::abs(dy) < 1.0E-7) dy = 0.0;
        else
            for (const auto& b : shapes) {
                dy = cubeCollideY(b, box, dy);
                if (std::abs(dy) < 1.0E-7) { dy = 0.0; break; }
            }
        if (dy != 0.0) box = box.move(0.0, dy, 0.0);
    }
    bool zFirst = std::abs(vec.x) < std::abs(vec.z);
    double dx = vec.x, dz = vec.z;
    if (zFirst && dz != 0.0) {
        if (std::abs(dz) < 1.0E-7) dz = 0.0;
        else
            for (const auto& b : shapes) {
                dz = cubeCollideZ(b, box, dz);
                if (std::abs(dz) < 1.0E-7) { dz = 0.0; break; }
            }
        if (dz != 0.0) box = box.move(0.0, 0.0, dz);
    }
    if (dx != 0.0) {
        if (std::abs(dx) < 1.0E-7) dx = 0.0;
        else
            for (const auto& b : shapes) {
                dx = cubeCollideX(b, box, dx);
                if (std::abs(dx) < 1.0E-7) { dx = 0.0; break; }
            }
        if (!zFirst && dx != 0.0) box = box.move(dx, 0.0, 0.0);
    }
    if (!zFirst && dz != 0.0) {
        if (std::abs(dz) < 1.0E-7) dz = 0.0;
        else
            for (const auto& b : shapes) {
                dz = cubeCollideZ(b, box, dz);
                if (std::abs(dz) < 1.0E-7) { dz = 0.0; break; }
            }
    }
    return {dx, dy, dz};
}

static int collectCandidateStepUpHeights(const AABB& box, const BoxList& shapes,
                                         float maxUpStep, float collidedY, float* out) {
    int n = 0;
    for (const auto& b : shapes) {
        double coords[2] = {b.minY, b.maxY};
        for (double d : coords) {
            float f = (float)(d - box.minY);
            if (f < 0.0f || f == collidedY) continue;
            if (f > maxUpStep) continue;
            bool dup = false;
            for (int i = 0; i < n; ++i)
                if (out[i] == f) { dup = true; break; }
            if (!dup && n < 2 * BoxList::CAP) out[n++] = f;
        }
    }
    std::sort(out, out + n);
    return n;
}

static Vec3 collide(const Arena& arena, Player& p, const Vec3& vec) {
    AABB box = p.bb;
    static thread_local BoxList list;
    arena.getCollidingBoundingBoxes(box.expandTowards(vec.x, vec.y, vec.z), list);
    Vec3 vec32 = vec.lengthSqr() == 0.0 ? vec : collideWithShapes(vec, box, list);
    bool collidedX = vec.x != vec32.x;
    bool collidedY = vec.y != vec32.y;
    bool collidedZ = vec.z != vec32.z;
    bool bl = collidedY && vec.y < 0.0;
    if (Loadout::STEP_HEIGHT > 0.0f && (bl || p.onGround) && (collidedX || collidedZ)) {
        AABB box2 = bl ? box.move(0.0, vec32.y, 0.0) : box;
        AABB probe = box2.expandTowards(vec.x, (double)Loadout::STEP_HEIGHT, vec.z);
        if (!bl) probe = probe.expandTowards(0.0, (double)-1.0E-5f, 0.0);
        static thread_local BoxList stepList;
        arena.getCollidingBoundingBoxes(probe, stepList);
        float candidates[2 * BoxList::CAP];
        int n = collectCandidateStepUpHeights(box2, stepList, Loadout::STEP_HEIGHT,
                                              (float)vec32.y, candidates);
        for (int i = 0; i < n; ++i) {
            Vec3 vec33 = collideWithShapes({vec.x, (double)candidates[i], vec.z}, box2, stepList);
            if (vec33.horizontalDistanceSqr() > vec32.horizontalDistanceSqr()) {
                double d = box.minY - box2.minY;
                return vec33.add(0.0, -d, 0.0);
            }
        }
    }
    return vec32;
}

static bool canFallAtLeast(const Arena& arena, const Player& p, double dx, double dz, double dy) {
    const AABB& b = p.bb;
    return arena.noCollision(AABB::raw(b.minX + 1.0E-7 + dx, b.minY - dy - 1.0E-7,
                                       b.minZ + 1.0E-7 + dz, b.maxX - 1.0E-7 + dx,
                                       b.minY, b.maxZ - 1.0E-7 + dz));
}
static bool isAboveGround(const Arena& arena, const Player& p, float f) {
    return p.onGround ||
           (p.fallDistance < (double)f && !canFallAtLeast(arena, p, 0.0, 0.0, (double)f - p.fallDistance));
}
static Vec3 maybeBackOffFromEdge(const Arena& arena, const Player& p, Vec3 vec) {
    float f = Loadout::STEP_HEIGHT;
    if (vec.y > 0.0 || !p.shiftKeyDown || !isAboveGround(arena, p, f)) return vec;
    double d = vec.x;
    double d2 = vec.z;
    double d4 = Mth::sign(d) * 0.05;
    double d5 = Mth::sign(d2) * 0.05;
    while (d != 0.0 && canFallAtLeast(arena, p, d, 0.0, (double)f)) {
        if (std::abs(d) <= 0.05) { d = 0.0; break; }
        d -= d4;
    }
    while (d2 != 0.0 && canFallAtLeast(arena, p, 0.0, d2, (double)f)) {
        if (std::abs(d2) <= 0.05) { d2 = 0.0; break; }
        d2 -= d5;
    }
    while (d != 0.0 && d2 != 0.0 && canFallAtLeast(arena, p, d, d2, (double)f)) {
        d = std::abs(d) <= 0.05 ? 0.0 : d - d4;
        if (std::abs(d2) <= 0.05) d2 = 0.0;
        else d2 -= d5;
    }
    return {d, vec.y, d2};
}

static bool isHorizontalCollisionMinor(const Player& p, const Vec3& moved) {
    float f = p.yaw * DEG_TO_RAD_F;
    double d = (double)Mth::sin(f);
    double d2 = (double)Mth::cos(f);
    double d3 = (double)p.xxa * d2 - (double)p.zza * d;
    double d4 = (double)p.zza * d2 + (double)p.xxa * d;
    double d5 = Mth::square(d3) + Mth::square(d4);
    double d6 = Mth::square(moved.x) + Mth::square(moved.z);
    if (d5 < (double)1.0E-5f || d6 < (double)1.0E-5f) return false;
    double d7 = d3 * moved.x + d4 * moved.z;
    double d8 = std::acos(d7 / std::sqrt(d5 * d6));
    return d8 < 0.13962633907794952;
}

void moveEntity(const Arena& arena, Player& p, Vec3 vec, bool localAuthority) {
    vec = maybeBackOffFromEdge(arena, p, vec);
    Vec3 vec32 = collide(arena, p, vec);
    double d = vec32.lengthSqr();
    if (d > 1.0E-7 || vec.lengthSqr() - d < 1.0E-7) {
        setPosition(p, p.posX + vec32.x, p.posY + vec32.y, p.posZ + vec32.z);
    }
    bool bl = !Mth::equal(vec.x, vec32.x);
    bool bl2 = !Mth::equal(vec.z, vec32.z);
    p.collidedHorizontally = bl || bl2;
    if (std::abs(vec.y) > 0.0 || localAuthority) {
        p.collidedVertically = vec.y != vec32.y;
        p.verticalCollisionBelow = p.collidedVertically && vec.y < 0.0;
        p.onGround = p.verticalCollisionBelow;
    }
    p.minorHorizontalCollision =
        p.collidedHorizontally && localAuthority ? isHorizontalCollisionMinor(p, vec32) : false;
    if (localAuthority) {
        if (vec32.y < 0.0) p.fallDistance -= (double)(float)vec32.y;
        if (p.onGround) p.fallDistance = 0.0;
    }
    if (p.collidedHorizontally) {
        if (bl) p.motionX = 0.0;
        if (bl2) p.motionZ = 0.0;
    }
    if (vec.y != vec32.y) p.motionY = 0.0;
}


static Vec3 getInputVector(const Vec3& rel, float speed, float yaw) {
    double d = rel.lengthSqr();
    if (d < 1.0E-7) return {0, 0, 0};
    Vec3 vec3 = (d > 1.0 ? rel.normalize() : rel).scale((double)speed);
    float f = Mth::sin(yaw * DEG_TO_RAD_F);
    float f2 = Mth::cos(yaw * DEG_TO_RAD_F);
    return {vec3.x * (double)f2 - vec3.z * (double)f, vec3.y,
            vec3.z * (double)f2 + vec3.x * (double)f};
}
static void moveRelative(Player& p, float speed, const Vec3& rel) {
    Vec3 v = getInputVector(rel, speed, p.yaw);
    p.motionX += v.x; p.motionY += v.y; p.motionZ += v.z;
}

static float frictionInfluencedSpeed(const Player& p, float friction) {
    if (p.onGround) return p.speed * (0.21600002f / (friction * friction * friction));
    return p.sprinting ? 0.025999999f : 0.02f;
}

static void travelInAir(const Arena& arena, Player& p, const Vec3& input, bool localAuthority) {
    float f = p.onGround ? 0.6f : 1.0f;
    float f2 = f * 0.91f;
    moveRelative(p, frictionInfluencedSpeed(p, f), input);
    moveEntity(arena, p, p.deltaMovement(), localAuthority);
    Vec3 vec32 = p.deltaMovement();
    double d = vec32.y;
    d -= Loadout::GRAVITY;
    p.setDeltaMovement(vec32.x * (double)f2, d * (double)0.98f, vec32.z * (double)f2);
}


static void zeroSmallMotion(Player& p) {
    double mx = p.motionX, my = p.motionY, mz = p.motionZ;
    if (p.motionX * p.motionX + p.motionZ * p.motionZ < 9.0E-6) { mx = 0.0; mz = 0.0; }
    if (std::abs(p.motionY) < 0.003) my = 0.0;
    p.setDeltaMovement(mx, my, mz);
}

static void aiStepJump(Player& p) {
    if (p.jumping) {
        if (p.onGround && p.jumpTicks == 0) {
            jumpFromGround(p);
            p.jumpTicks = 10;
        }
    } else {
        p.jumpTicks = 0;
    }
}

void updateSwingTime(Player& p) {
    int duration = 6;
    if (p.swinging) {
        ++p.swingTime;
        if (p.swingTime >= duration) {
            p.swingTime = 0;
            p.swinging = false;
        }
    } else {
        p.swingTime = 0;
    }
}

void startSwing(Player& p) {
    if (!p.swinging || p.swingTime >= 6 / 2 || p.swingTime < 0) {
        p.swingTime = -1;
        p.swinging = true;
    }
}

static bool canFit(const Arena& arena, const Player& p, Pose pose) {
    return arena.noCollision(makePlayerBB(p.posX, p.posY, p.posZ, pose).deflate(1.0E-7));
}
void updatePlayerPose(const Arena& arena, Player& p) {
    Pose desired = p.shiftKeyDown ? Pose::Crouching : Pose::Standing;
    Pose next = canFit(arena, p, desired) ? desired : Pose::Crouching;
    if (next != p.pose) {
        p.pose = next;
        p.bb = makePlayerBB(p.posX, p.posY, p.posZ, p.pose);
    }
}


void clientPlayerTick(const Arena& arena, Player& p, const Input& in) {
    if (p.hurtTime > 0) --p.hurtTime;
    if (p.hurtResistantTime > 0) --p.hurtResistantTime;

    if (p.sprintTriggerTime > 0) --p.sprintTriggerTime;

    bool bl2 = p.prevShift;
    bool bl3 = p.prevForwardImpulse;

    p.sneaking = canFit(arena, p, Pose::Crouching) &&
                 (bl2 || !canFit(arena, p, Pose::Standing));

    float impF = (in.forward ? 1.0f : 0.0f) - (in.back ? 1.0f : 0.0f);
    float impS = (in.left ? 1.0f : 0.0f) - (in.right ? 1.0f : 0.0f);
    Vec2 moveVector = Vec2(impS, impF).normalized();
    bool newForwardImpulse = moveVector.y > 1.0E-5f;
    p.shiftKeyDown = in.sneak;

    if (bl2 || in.back) p.sprintTriggerTime = 0;

    bool hasEnoughFood = (float)p.food.foodLevel > 6.0f;
    bool canStart = !(p.sprinting || !newForwardImpulse || !hasEnoughFood || p.sneaking);
    if (canStart) {
        if (!bl3) {
            if (p.sprintTriggerTime > 0) p.setSprinting(true);
            else p.sprintTriggerTime = 7;
        }
        if (in.sprintKey) p.setSprinting(true);
    }
    if (p.sprinting &&
        (!newForwardImpulse || !hasEnoughFood ||
         (p.collidedHorizontally && !p.minorHorizontalCollision)))
        p.setSprinting(false);

    if (p.jumpTicks > 0) --p.jumpTicks;
    zeroSmallMotion(p);

    Vec2 v = moveVector;
    if (v.lengthSquared() != 0.0f) {
        v = v.scale(0.98f);
        if (p.sneaking) v = v.scale((float)Loadout::SNEAKING_SPEED);
        float len = v.length();
        if (len > 0.0f) {
            Vec2 unit = v.scale(1.0f / len);
            float ax = std::abs(unit.x), ay = std::abs(unit.y);
            float ratio = ay > ax ? ax / ay : ay / ax;
            float f2 = Mth::sqrt(1.0f + Mth::square(ratio));
            float f3 = std::min(len * f2, 1.0f);
            v = unit.scale(f3);
        }
    }
    p.xxa = v.x;
    p.zza = v.y;
    p.jumping = in.jump;

    aiStepJump(p);
    travelInAir(arena, p, {(double)p.xxa, 0.0, (double)p.zza}, true);

    updateSwingTime(p);
    p.speed = (float)p.movementSpeedAttribute();

    ++p.attackStrengthTicker;
    updatePlayerPose(arena, p);

    p.prevShift = in.sneak;
    p.prevForwardImpulse = newForwardImpulse;
}


static bool pushPair(Player& e, Player& other) {
    double d2 = other.posX - e.posX;
    double d = other.posZ - e.posZ;
    double d3 = Mth::absMax(d2, d);
    if (d3 >= (double)0.01f) {
        d3 = std::sqrt(d3);
        d2 /= d3;
        d /= d3;
        double d4 = 1.0 / d3;
        if (d4 > 1.0) d4 = 1.0;
        d2 *= d4;
        d *= d4;
        d2 *= (double)0.05f;
        d *= (double)0.05f;
        e.motionX += -d2; e.motionZ += -d;
        e.hasImpulse = true;
        other.motionX += d2; other.motionZ += d;
        other.hasImpulse = true;
        return true;
    }
    return false;
}

bool serverPlayerAiStep(const Arena& arena, Player& e, Player& other) {
    if (e.hurtTime > 0) --e.hurtTime;

    if (e.jumpTicks > 0) --e.jumpTicks;
    zeroSmallMotion(e);
    e.xxa = 0.0f;
    e.zza = 0.0f;
    e.jumping = false;
    e.jumpTicks = 0;
    travelInAir(arena, e, {0.0, 0.0, 0.0}, false);

    bool pushed = false;
    if (other.bb.intersects(e.bb)) pushed = pushPair(e, other);

    updateSwingTime(e);
    e.speed = (float)e.movementSpeedAttribute();
    return pushed;
}


void foodTick(Player& p) {
    FoodData& fd = p.food;
    if (fd.exhaustionLevel > 4.0f) {
        fd.exhaustionLevel -= 4.0f;
        if (fd.saturationLevel > 0.0f) {
            fd.saturationLevel = std::max(fd.saturationLevel - 1.0f, 0.0f);
        } else {
            fd.foodLevel = std::max(fd.foodLevel - 1, 0);
        }
    }
    bool isHurt = p.health > 0.0f && p.health < Loadout::MAX_HEALTH;
    auto heal = [&](float f) {
        if (p.health > 0.0f) p.health = std::min(p.health + f, Loadout::MAX_HEALTH);
    };
    if (fd.saturationLevel > 0.0f && isHurt && fd.foodLevel >= 20) {
        ++fd.tickTimer;
        if (fd.tickTimer >= 10) {
            float f = std::min(fd.saturationLevel, 6.0f);
            heal(f / 6.0f);
            fd.addExhaustion(f);
            fd.tickTimer = 0;
        }
    } else if (fd.foodLevel >= 18 && isHurt) {
        ++fd.tickTimer;
        if (fd.tickTimer >= 80) {
            heal(1.0f);
            fd.addExhaustion(6.0f);
            fd.tickTimer = 0;
        }
    } else if (fd.foodLevel <= 0) {
        ++fd.tickTimer;
        if (fd.tickTimer >= 80) {
            if (p.health > 10.0f || p.health > 1.0f) {
                p.health = std::max(p.health - 1.0f, 0.0f);
                if (p.health <= 0.0f) p.dead = true;
            }
            fd.tickTimer = 0;
        }
    } else {
        fd.tickTimer = 0;
    }
}


float getDamageAfterAbsorb(float damage, float armor, float toughness) {
    float f5 = 2.0f + toughness / 4.0f;
    float f6 = Mth::clamp(armor - damage / f5, armor * 0.2f, 20.0f);
    float f7 = f6 / 25.0f;
    return damage * (1.0f - f7);
}

float getDamageAfterMagicAbsorb(float damage, float epf) {
    float f3 = Mth::clamp(epf, 0.0f, 20.0f);
    return damage * (1.0f - f3 / 25.0f);
}

bool hurtAndBreak(ItemDur& item, int points, float ignoreChance, std::mt19937_64& rng) {
    if (item.broken || points <= 0) return false;
    int kept = 0;
    for (int i = 0; i < points; ++i) {
        float roll = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
        if (!(roll < ignoreChance)) ++kept;
    }
    if (kept == 0) return false;
    item.damage += kept;
    if (item.damage >= item.maxDamage) {
        item.broken = true;
        return true;
    }
    return false;
}


bool pointedAtEntity(const Arena& arena, const Vec3& eye, float pitch, float yaw,
                     const AABB& targetBB) {
    double blockRange = Loadout::BLOCK_INTERACTION_RANGE;
    double entityRange = Loadout::ENTITY_INTERACTION_RANGE;
    double d3 = std::max(blockRange, entityRange);
    double d4 = Mth::square(d3);

    Vec3 view = getViewVector(pitch, yaw);
    Vec3 end = eye.add(view.x * d3, view.y * d3, view.z * d3);
    double blockDistSq = d4;
    if (auto hit = arena.rayTraceBlocks(eye, end)) {
        blockDistSq = hit->hitVec.distanceToSqr(eye);
        d3 = std::sqrt(blockDistSq);
        end = eye.add(view.x * d3, view.y * d3, view.z * d3);
    }

    std::optional<Vec3> hitVec;
    double bestSq = d4;
    if (targetBB.contains(eye)) {
        auto clip = targetBB.clip(eye, end);
        hitVec = clip ? *clip : eye;
        bestSq = 0.0;
    } else if (auto clip = targetBB.clip(eye, end)) {
        double distSq = eye.distanceToSqr(*clip);
        if (distSq < bestSq) {
            hitVec = *clip;
            bestSq = distSq;
        }
    }
    if (!hitVec) return false;
    if (!(bestSq < blockDistSq)) return false;
    if (!hitVec->closerThan(eye, entityRange)) return false;
    return true;
}

}
