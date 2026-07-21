#include "sim.h"

namespace mc1218 {

static int javaRoundF(float f) { return Mth::floor(f + 0.5f); }

static bool sameInput(const Input& a, const Input& b) {
    return a.forward == b.forward && a.back == b.back && a.left == b.left &&
           a.right == b.right && a.jump == b.jump && a.sneak == b.sneak &&
           a.sprintKey == b.sprintKey;
}

static void setRemotePos(RemoteView& v, double x, double y, double z) {
    v.posX = x; v.posY = y; v.posZ = z;
    v.bb = makePlayerBB(x, y, z, v.pose);
}


void Sim::reset() {
    for (int i = 0; i < 2; ++i) {
        client[i] = ClientSim{};
        sv[i] = ServerPlayer{};
        c2s_[i].clear();
        s2c_[i].clear();
    }
    tickCount = 0;
    scoreDirty_ = false;
    double cx = Arena::CENTER_X, cz = Arena::CENTER_Z;
    place(0, cx - SPAWN_DIST / 2.0, 0.0, cz, -90.0f, 0.0f);
    place(1, cx + SPAWN_DIST / 2.0, 0.0, cz, 90.0f, 0.0f);
}

void Sim::place(int i, double x, double y, double z, float yaw, float pitch) {
    Player& pl = client[i].self;
    setPosition(pl, x, y, z);
    pl.prevPosX = x; pl.prevPosY = y; pl.prevPosZ = z;
    pl.motionX = pl.motionY = pl.motionZ = 0.0;
    pl.yaw = pl.prevYaw = yaw;
    pl.pitch = pl.prevPitch = pitch;
    pl.fallDistance = 0.0;
    pl.onGround = true;
    pl.collidedHorizontally = pl.collidedVertically = false;
    pl.verticalCollisionBelow = pl.minorHorizontalCollision = false;
    pl.hasImpulse = false;

    ServerPlayer& s = sv[i];
    Player& e = s.ent;
    e.pose = pl.pose;
    setPosition(e, x, y, z);
    e.prevPosX = x; e.prevPosY = y; e.prevPosZ = z;
    e.motionX = e.motionY = e.motionZ = 0.0;
    e.yaw = Mth::wrapDegrees(yaw);
    e.pitch = Mth::wrapDegrees(pitch);
    e.prevYaw = e.yaw;
    e.prevPitch = e.pitch;
    e.fallDistance = 0.0;
    e.onGround = true;
    e.collidedHorizontally = e.collidedVertically = false;
    e.verticalCollisionBelow = e.minorHorizontalCollision = false;
    e.hasImpulse = false;
    e.xxa = e.zza = 0.0f;
    e.jumping = false;
    s.firstGoodX = s.lastGoodX = x;
    s.firstGoodY = s.lastGoodY = y;
    s.firstGoodZ = s.lastGoodZ = z;
    s.knownMovement = Vec3{};
    s.receivedMovementThisTick = false;
    s.metaDirty = false;
    s.attrDirty = false;
    s.hurtMarked = false;
    s.codecBaseX = x; s.codecBaseY = y; s.codecBaseZ = z;
    s.lastSentYRot = Mth::packDegrees(s.ent.yaw);
    s.lastSentXRot = Mth::packDegrees(s.ent.pitch);
    s.lastSentMovement = Vec3{};
    s.trackerTickCount = 1;
    s.teleportDelay = 0;
    s.wasOnGround = true;
    for (int g = 0; g < 5; ++g) {
        const ItemDur& it = g < 4 ? s.ent.gear.armor[g] : s.ent.gear.sword;
        s.lastEquipDamage[g] = (int16_t)it.damage;
        s.lastEquipBroken[g] = it.broken;
        s.lastSlotDamage[g] = (int16_t)it.damage;
        s.lastSlotBroken[g] = it.broken;
    }

    RemoteView& v = client[1 - i].remote;
    v.pose = s.ent.pose;
    setRemotePos(v, x, y, z);
    v.prevPosX = x; v.prevPosY = y; v.prevPosZ = z;
    v.baseX = x; v.baseY = y; v.baseZ = z;
    v.yaw = Mth::unpackDegrees(Mth::packDegrees(s.ent.yaw));
    v.pitch = Mth::unpackDegrees(Mth::packDegrees(s.ent.pitch));
    v.lerpSteps = 0;
    v.onGround = true;
    v.hurtTime = 0;
    v.sprinting = s.ent.sprinting;
    v.sneaking = s.ent.pose == Pose::Crouching;
    v.health = s.ent.health;
    v.motionX = v.motionY = v.motionZ = 0.0;
    v.lerpMotionSteps = 0;
    v.swinging = false;
    v.swingTime = 0;
    for (int g = 0; g < 5; ++g) {
        const ItemDur& it = g < 4 ? s.ent.gear.armor[g] : s.ent.gear.sword;
        v.gearDamage[g] = (int16_t)it.damage;
        v.gearBroken[g] = it.broken;
    }

    ClientSim& c = client[i];
    c.xLast = x; c.yLast = y; c.zLast = z;
    c.yRotLast = yaw;
    c.xRotLast = pitch;
    c.positionReminder = 0;
    c.lastOnGround = true;
    c.lastHorizontalCollision = false;
    c.wasSprinting = pl.sprinting;
    c.lastSentInput = Input{};
    c.missTime = 0;

    for (int k = 0; k < 2; ++k) { c2s_[k].clear(); s2c_[k].clear(); }
}

void Sim::setMotion(int i, double mx, double my, double mz) {
    client[i].self.setDeltaMovement(mx, my, mz);
    sv[i].ent.setDeltaMovement(mx, my, mz);
}

void Sim::setHits(int i, int n) {
    sv[i].hits = n < 0 ? 0 : n;
    scoreDirty_ = true;
}

void Sim::setVitals(int i, float health, int food, float saturation) {
    health = Mth::clamp(health, 0.0f, Loadout::MAX_HEALTH);
    food = Mth::clamp(food, 0, 20);
    saturation = Mth::clamp(saturation, 0.0f, (float)food);
    sv[i].ent.health = health;
    sv[i].ent.dead = health <= 0.0f;
    sv[i].ent.food.foodLevel = food;
    sv[i].ent.food.saturationLevel = saturation;
    sv[i].ent.food.exhaustionLevel = 0.0f;
    sv[i].ent.food.tickTimer = 0;
    client[i].self.health = health;
    client[i].self.food.foodLevel = food;
    client[i].self.food.saturationLevel = saturation;
    client[i].flashOnSetHealth = true;
    client[1 - i].remote.health = health;
    sv[i].lastSentHealth = health;
    sv[i].lastSentFood = food;
    sv[i].lastFoodSaturationZero = saturation == 0.0f;
}


StepResult Sim::step(const Input& in0, const Input& in1) {
    const Input* ins[2] = {&in0, &in1};
    StepResult r;

    for (int i = 0; i < 2; ++i) {
        s2c_[i].drain(tickCount, [&](const Pkt& p) { handleClientPacket(i, p, r); });

        ClientSim& c = client[i];
        Player& pl = c.self;

        if (c.missTime > 0) --c.missTime;

        pl.prevPosX = pl.posX; pl.prevPosY = pl.posY; pl.prevPosZ = pl.posZ;
        pl.prevYaw = pl.yaw; pl.prevPitch = pl.pitch;
        applyLook(pl, ins[i]->yawDelta, ins[i]->pitchDelta);

        if (ins[i]->attack) clientClick(i, r);
        c.missTime = 0;

        clientPlayerTick(arena, pl, *ins[i]);
        if (!sameInput(*ins[i], c.lastSentInput)) {
            Pkt p{};
            p.type = Pkt::C_Input;
            p.kForward = ins[i]->forward; p.kBack = ins[i]->back;
            p.kLeft = ins[i]->left; p.kRight = ins[i]->right;
            p.kJump = ins[i]->jump; p.kShift = ins[i]->sneak;
            p.kSprint = ins[i]->sprintKey;
            sendToServer(i, p);
            c.lastSentInput = *ins[i];
        }
        clientSendMovePackets(i);

        remoteTick(c.remote);

        Pkt te{};
        te.type = Pkt::C_TickEnd;
        sendToServer(i, te);
    }

    int first = tickCount & 1;
    for (int k = 0; k < 2; ++k) {
        int i = (first + k) % 2;
        c2s_[i].drain(tickCount, [&](const Pkt& p) { handleServerPacket(i, p, r); });
    }

    for (int i = 0; i < 2; ++i) trackerTick(i);

    for (int i = 0; i < 2; ++i) {
        Player& e = sv[i].ent;
        if (e.hurtResistantTime > 0) --e.hurtResistantTime;
        bool changed = false;
        for (int g = 0; g < 5; ++g) {
            const ItemDur& it = g < 4 ? e.gear.armor[g] : e.gear.sword;
            if (sv[i].lastSlotDamage[g] != (int16_t)it.damage ||
                sv[i].lastSlotBroken[g] != it.broken) {
                changed = true;
                sv[i].lastSlotDamage[g] = (int16_t)it.damage;
                sv[i].lastSlotBroken[g] = it.broken;
            }
        }
        if (changed) {
            Pkt p{};
            p.type = Pkt::S_OwnSlots;
            for (int g = 0; g < 5; ++g) {
                p.gearDamage[g] = sv[i].lastSlotDamage[g];
                p.gearBroken[g] = sv[i].lastSlotBroken[g];
            }
            sendToClient(i, p);
        }
    }

    for (int i = 0; i < 2; ++i) serverConnectionTick(i, r);

    if (scoreDirty_) {
        Pkt p{};
        p.type = Pkt::S_Score;
        p.hits0 = sv[0].hits;
        p.hits1 = sv[1].hits;
        sendToClient(0, p);
        sendToClient(1, p);
        scoreDirty_ = false;
    }

    ++tickCount;
    return r;
}


void Sim::clientClick(int i, StepResult& r) {
    ClientSim& c = client[i];
    Player& a = c.self;

    if (c.missTime > 0) return;
    if (a.health <= 0.0f) return;

    r.swingCharge[i] = a.attackStrengthScale(0.5f);

    if (pointedAtEntity(arena, a.eyePos(), a.pitch, a.yaw, c.remote.bb)) {
        r.clientAttack[i] = true;
        Pkt p{};
        p.type = Pkt::C_Attack;
        p.secondaryAction = a.shiftKeyDown;
        sendToServer(i, p);

        float f3 = a.attackStrengthScale(0.5f);
        bool bl3 = f3 > 0.9f;
        if (a.sprinting && bl3) {
            a.motionX *= 0.6;
            a.motionZ *= 0.6;
            a.setSprinting(false);
        }
        a.attackStrengthTicker = 0;
    } else {
        r.swungAndMissed[i] = true;
        c.missTime = 10;
        a.attackStrengthTicker = 0;
    }
    startSwing(a);
    Pkt sw{};
    sw.type = Pkt::C_Swing;
    sendToServer(i, sw);
}

void Sim::clientSendMovePackets(int i) {
    ClientSim& c = client[i];
    Player& pl = c.self;

    if (pl.sprinting != c.wasSprinting) {
        Pkt p{};
        p.type = Pkt::C_Command;
        p.action = pl.sprinting ? Pkt::StartSprint : Pkt::StopSprint;
        sendToServer(i, p);
        c.wasSprinting = pl.sprinting;
    }

    double d = pl.posX - c.xLast;
    double d2 = pl.posY - c.yLast;
    double d3 = pl.posZ - c.zLast;
    double d4 = (double)(pl.yaw - c.yRotLast);
    double d5 = (double)(pl.pitch - c.xRotLast);
    ++c.positionReminder;
    bool bl2 = Mth::lengthSquared(d, d2, d3) > Mth::square(2.0E-4) || c.positionReminder >= 20;
    bool bl = d4 != 0.0 || d5 != 0.0;

    Pkt p{};
    p.type = Pkt::C_Move;
    p.onGround = pl.onGround;
    p.horizontalCollision = pl.collidedHorizontally;
    bool send = false;
    if (bl2) { p.hasPos = true; p.x = pl.posX; p.y = pl.posY; p.z = pl.posZ; send = true; }
    if (bl) { p.hasLook = true; p.yaw = pl.yaw; p.pitch = pl.pitch; send = true; }
    if (!send && (c.lastOnGround != pl.onGround ||
                  c.lastHorizontalCollision != pl.collidedHorizontally))
        send = true;
    if (send) sendToServer(i, p);

    if (bl2) {
        c.xLast = pl.posX; c.yLast = pl.posY; c.zLast = pl.posZ;
        c.positionReminder = 0;
    }
    if (bl) {
        c.yRotLast = pl.yaw;
        c.xRotLast = pl.pitch;
    }
    c.lastOnGround = pl.onGround;
    c.lastHorizontalCollision = pl.collidedHorizontally;
}

void Sim::remoteTick(RemoteView& v) {
    v.prevPosX = v.posX; v.prevPosY = v.posY; v.prevPosZ = v.posZ;
    if (v.hurtTime > 0) --v.hurtTime;
    if (v.lerpSteps > 0) {
        double d = 1.0 / (double)v.lerpSteps;
        double nx = Mth::lerp(d, v.posX, v.lerpX);
        double ny = Mth::lerp(d, v.posY, v.lerpY);
        double nz = Mth::lerp(d, v.posZ, v.lerpZ);
        v.yaw = (float)Mth::rotLerp(d, (double)v.yaw, (double)v.lerpYaw);
        v.pitch = (float)Mth::lerp(d, (double)v.pitch, (double)v.lerpPitch);
        --v.lerpSteps;
        setRemotePos(v, nx, ny, nz);
    }
    if (v.lerpMotionSteps > 0) {
        v.motionX += (v.lerpMotionX - v.motionX) / (double)v.lerpMotionSteps;
        v.motionY += (v.lerpMotionY - v.motionY) / (double)v.lerpMotionSteps;
        v.motionZ += (v.lerpMotionZ - v.motionZ) / (double)v.lerpMotionSteps;
        --v.lerpMotionSteps;
    }
    if (v.swinging) {
        ++v.swingTime;
        if (v.swingTime >= 6) {
            v.swingTime = 0;
            v.swinging = false;
        }
    } else {
        v.swingTime = 0;
    }
}

void Sim::handleClientPacket(int i, const Pkt& p, StepResult& r) {
    ClientSim& c = client[i];
    switch (p.type) {
        case Pkt::S_Motion: {
            double mx = (double)p.mx / 8000.0;
            double my = (double)p.my / 8000.0;
            double mz = (double)p.mz / 8000.0;
            if (p.aboutSelf) {
                c.self.setDeltaMovement(mx, my, mz);
                r.kbReceived[i] = true;
                r.kb[i] = {mx, my, mz};
            } else {
                c.remote.lerpMotionX = mx;
                c.remote.lerpMotionY = my;
                c.remote.lerpMotionZ = mz;
                c.remote.lerpMotionSteps = 3;
            }
            break;
        }
        case Pkt::S_MoveRel: {
            RemoteView& v = c.remote;
            double nx = p.dx == 0 ? v.baseX
                                  : (double)(javaRound(v.baseX * 4096.0) + (int64_t)p.dx) / 4096.0;
            double ny = p.dy == 0 ? v.baseY
                                  : (double)(javaRound(v.baseY * 4096.0) + (int64_t)p.dy) / 4096.0;
            double nz = p.dz == 0 ? v.baseZ
                                  : (double)(javaRound(v.baseZ * 4096.0) + (int64_t)p.dz) / 4096.0;
            if (p.hasPos) {
                v.baseX = nx; v.baseY = ny; v.baseZ = nz;
                v.lerpX = nx; v.lerpY = ny; v.lerpZ = nz;
                v.lerpYaw = p.hasLook ? Mth::unpackDegrees(p.byaw) : v.yaw;
                v.lerpPitch = p.hasLook ? Mth::unpackDegrees(p.bpitch) : v.pitch;
                v.lerpSteps = 3;
            } else if (p.hasLook) {
                v.lerpX = v.posX; v.lerpY = v.posY; v.lerpZ = v.posZ;
                v.lerpYaw = Mth::unpackDegrees(p.byaw);
                v.lerpPitch = Mth::unpackDegrees(p.bpitch);
                v.lerpSteps = 3;
            }
            v.onGround = p.onGround;
            break;
        }
        case Pkt::S_PosSync: {
            RemoteView& v = c.remote;
            v.baseX = p.x; v.baseY = p.y; v.baseZ = p.z;
            v.lerpX = p.x; v.lerpY = p.y; v.lerpZ = p.z;
            v.lerpYaw = p.yaw;
            v.lerpPitch = p.pitch;
            v.lerpSteps = 3;
            v.onGround = p.onGround;
            break;
        }
        case Pkt::S_Meta: {
            RemoteView& v = c.remote;
            v.sprinting = p.sprinting;
            v.sneaking = p.crouching;
            Pose np = p.crouching ? Pose::Crouching : Pose::Standing;
            if (np != v.pose) {
                v.pose = np;
                setRemotePos(v, v.posX, v.posY, v.posZ);
            }
            v.health = p.health;
            break;
        }
        case Pkt::S_Attributes:
            if (p.aboutSelf) c.self.speedModSprint = p.sprintMod;
            break;
        case Pkt::S_Equipment:
            for (int g = 0; g < 5; ++g) {
                c.remote.gearDamage[g] = p.gearDamage[g];
                c.remote.gearBroken[g] = p.gearBroken[g];
            }
            break;
        case Pkt::S_OwnSlots:
            for (int g = 0; g < 5; ++g) {
                ItemDur& it = g < 4 ? c.self.gear.armor[g] : c.self.gear.sword;
                it.damage = p.gearDamage[g];
                it.broken = p.gearBroken[g];
            }
            break;
        case Pkt::S_Health: {
            Player& pl = c.self;
            if (!c.flashOnSetHealth) {
                pl.health = p.health;
                c.flashOnSetHealth = true;
            } else {
                float f2 = pl.health - p.health;
                if (f2 <= 0.0f) {
                    pl.health = p.health;
                    if (f2 < 0.0f) pl.hurtResistantTime = 10;
                } else {
                    pl.lastHurt = f2;
                    pl.hurtResistantTime = 20;
                    pl.health = p.health;
                    pl.hurtTime = 10;
                }
            }
            pl.food.foodLevel = p.food;
            pl.food.saturationLevel = p.saturation;
            break;
        }
        case Pkt::S_DamageEvent:
            if (p.aboutSelf) {
                c.self.hurtResistantTime = 20;
                c.self.hurtTime = 10;
                r.hurtFlash[i] = true;
            } else {
                c.remote.hurtTime = 10;
            }
            break;
        case Pkt::S_SwingAnim:
            if (!c.remote.swinging || c.remote.swingTime >= 3 || c.remote.swingTime < 0) {
                c.remote.swingTime = -1;
                c.remote.swinging = true;
            }
            break;
        case Pkt::S_Score:
            c.knownHits[0] = p.hits0;
            c.knownHits[1] = p.hits1;
            break;
        default:
            break;
    }
}


void Sim::handleServerPacket(int i, const Pkt& p, StepResult& r) {
    switch (p.type) {
        case Pkt::C_Move:
            serverHandleMove(i, p);
            break;
        case Pkt::C_Input:
            sv[i].ent.shiftKeyDown = p.kShift;
            break;
        case Pkt::C_Command:
            sv[i].ent.setSprinting(p.action == Pkt::StartSprint);
            sv[i].metaDirty = true;
            sv[i].attrDirty = true;
            break;
        case Pkt::C_Attack:
            serverAttack(i, r);
            break;
        case Pkt::C_Swing:
            startSwing(sv[i].ent);
            {
                Pkt sw{};
                sw.type = Pkt::S_SwingAnim;
                sendToClient(1 - i, sw);
            }
            break;
        case Pkt::C_TickEnd:
            if (!sv[i].receivedMovementThisTick) sv[i].knownMovement = Vec3{};
            sv[i].receivedMovementThisTick = false;
            break;
        default:
            break;
    }
}

void Sim::serverHandleMove(int i, const Pkt& p) {
    ServerPlayer& s = sv[i];
    Player& e = s.ent;

    float yaw = p.hasLook ? Mth::wrapDegrees(p.yaw) : e.yaw;
    float pitch = p.hasLook ? Mth::wrapDegrees(p.pitch) : e.pitch;
    double px = p.hasPos ? p.x : e.posX;
    double py = p.hasPos ? p.y : e.posY;
    double pz = p.hasPos ? p.z : e.posZ;

    double d4 = e.posX, d5 = e.posY, d6 = e.posZ;
    double d8 = py - s.lastGoodY;
    bool bl = d8 > 0.0;

    if (e.onGround && !p.onGround && bl) {
        jumpFromGround(e);
        e.food.addExhaustion(e.sprinting ? 0.2f : 0.05f);
    }

    moveEntity(arena, e, {px - s.lastGoodX, py - s.lastGoodY, pz - s.lastGoodZ},
               false);

    setPosition(e, px, py, pz);
    e.yaw = yaw;
    e.pitch = pitch;

    Vec3 actual{e.posX - d4, e.posY - d5, e.posZ - d6};

    e.onGround = p.onGround;
    e.collidedHorizontally = p.horizontalCollision;

    if (actual.y < 0.0) e.fallDistance -= (double)(float)actual.y;
    if (p.onGround) {
        if (e.fallDistance > 0.0) {
            int dmg = Mth::floor((e.fallDistance + 1.0E-6 - Loadout::SAFE_FALL_DISTANCE) *
                                 Loadout::FALL_DAMAGE_MULTIPLIER);
            if (dmg > 0) {
                StepResult dummy;
                serverHurt(i, -1, (float)dmg, true,
                           false, true, 0.0f,
                           dummy, nullptr);
            }
        }
        e.fallDistance = 0.0;
    }

    s.knownMovement = actual;
    s.receivedMovementThisTick = true;
    if (bl) e.fallDistance = 0.0;

    if (!(actual.x == 0.0 && actual.y == 0.0 && actual.z == 0.0) && e.onGround) {
        int n = javaRoundF(Mth::sqrt((float)(actual.x * actual.x + actual.z * actual.z)) * 100.0f);
        if (n > 0 && e.sprinting) e.food.addExhaustion(0.1f * (float)n * 0.01f);
    }

    s.lastGoodX = e.posX;
    s.lastGoodY = e.posY;
    s.lastGoodZ = e.posZ;
}

bool Sim::serverHurt(int victim, int attacker, float amount, bool bypassArmor,
                     bool bypassEffects, bool noKnockback, float exhaustion,
                     StepResult& r, float* dealtOut, bool* freshOut) {
    ServerPlayer& V = sv[victim];
    Player& v = V.ent;
    if (v.dead) return false;
    float f = amount < 0.0f ? 0.0f : amount;

    auto actuallyHurt = [&](float dmg) -> float {
        float f2 = dmg;
        if (!bypassArmor) {
            int n = (int)std::max(1.0f, dmg / 4.0f);
            for (int slot = 0; slot < 4; ++slot) {
                if (v.gear.armor[slot].broken) continue;
                if (hurtAndBreak(v.gear.armor[slot], n, Loadout::UNBREAKING_IGNORE_ARMOR, rng_))
                    r.armorBroke[victim] = true;
            }
            f2 = getDamageAfterAbsorb(f2, v.gear.armorValue(), v.gear.toughness());
        }
        if (!bypassEffects)
            f2 = getDamageAfterMagicAbsorb(f2, v.gear.protectionEpf());
        if (f2 == 0.0f) return 0.0f;
        v.food.addExhaustion(exhaustion);
        v.health = Mth::clamp(v.health - f2, 0.0f, Loadout::MAX_HEALTH);
        V.metaDirty = true;
        return f2;
    };

    float dealt;
    bool bl4;
    if ((float)v.hurtResistantTime > 10.0f) {
        if (f <= v.lastHurt) return false;
        dealt = actuallyHurt(f - v.lastHurt);
        v.lastHurt = f;
        bl4 = false;
    } else {
        v.lastHurt = f;
        v.hurtResistantTime = 20;
        dealt = actuallyHurt(f);
        v.hurtTime = 10;
        bl4 = true;
    }

    if (bl4) {
        Pkt de{};
        de.type = Pkt::S_DamageEvent;
        de.aboutSelf = true;
        sendToClient(victim, de);
        Pkt der{};
        der.type = Pkt::S_DamageEvent;
        der.aboutSelf = false;
        sendToClient(1 - victim, der);

        V.hurtMarked = true;

        if (!noKnockback && attacker >= 0) {
            const Player& a = sv[attacker].ent;
            knockback(v, 0.4, a.posX - v.posX, a.posZ - v.posZ, rng_);
        }
    }

    if (v.health <= 0.0f && !v.dead) {
        v.dead = true;
        r.death[victim] = true;
        v.setDeltaMovement((double)(-Mth::cos(v.yaw * DEG_TO_RAD_F) * 0.1f), 0.1,
                           (double)(-Mth::sin(v.yaw * DEG_TO_RAD_F) * 0.1f));
        v.hasImpulse = true;
    }
    if (dealtOut) *dealtOut = dealt;
    if (freshOut) *freshOut = bl4;
    return true;
}

void Sim::serverAttack(int attacker, StepResult& r) {
    ServerPlayer& A = sv[attacker];
    ServerPlayer& V = sv[1 - attacker];
    Player& a = A.ent;
    Player& v = V.ent;

    if (a.dead) return;

    double range = Loadout::ENTITY_INTERACTION_RANGE + 3.0;
    if (!(v.bb.distanceToSqr(a.eyePos()) < range * range)) return;

    float f = Loadout::SWORD_ATTACK_DAMAGE;
    float f2 = 0.0f;
    float f3 = a.attackStrengthScale(0.5f);
    f *= 0.2f + f3 * f3 * 0.8f;
    a.attackStrengthTicker = 0;

    if (!(f > 0.0f || f2 > 0.0f)) return;

    bool bl3 = f3 > 0.9f;
    bool bl2 = a.sprinting && bl3;
    bool bl = bl3 && a.fallDistance > 0.0 && !a.onGround && !a.sprinting;
    if (bl) f *= 1.5f;
    float f4 = f + f2;
    bool bl6 = bl3 && !bl && !bl2 && a.onGround &&
               A.knownMovement.horizontalDistanceSqr() <
                   Mth::square((double)a.speed * 2.5);

    Vec3 preMotion = v.deltaMovement();
    float dealt = 0.0f;
    bool fresh = false;
    bool bl7 = serverHurt(1 - attacker, attacker, f4, false,
                          false, false,
                          0.1f , r, &dealt, &fresh);
    if (!bl7) return;

    float f6 = 0.0f + (bl2 ? 1.0f : 0.0f);
    if (f6 > 0.0f) {
        knockback(v, (double)(f6 * 0.5f), (double)Mth::sin(a.yaw * DEG_TO_RAD_F),
                  (double)(-Mth::cos(a.yaw * DEG_TO_RAD_F)), rng_);
        a.motionX *= 0.6;
        a.motionZ *= 0.6;
        a.setSprinting(false);
        A.metaDirty = true;
        A.attrDirty = true;
    }

    if (bl6) r.sweep[attacker] = true;

    if (V.hurtMarked) {
        Pkt s12{};
        s12.type = Pkt::S_Motion;
        s12.aboutSelf = true;
        s12.mx = (int16_t)javaIntCast(Mth::clamp(v.motionX, -3.9, 3.9) * 8000.0);
        s12.my = (int16_t)javaIntCast(Mth::clamp(v.motionY, -3.9, 3.9) * 8000.0);
        s12.mz = (int16_t)javaIntCast(Mth::clamp(v.motionZ, -3.9, 3.9) * 8000.0);
        sendToClient(1 - attacker, s12);
        V.hurtMarked = false;
        v.setDeltaMovement(preMotion.x, preMotion.y, preMotion.z);
    }

    if (hurtAndBreak(a.gear.sword, Loadout::SWORD_DAMAGE_PER_ATTACK,
                     Loadout::UNBREAKING_IGNORE_TOOL, rng_))
        r.swordBroke[attacker] = true;

    a.food.addExhaustion(0.1f);

    A.hits += 1;
    scoreDirty_ = true;
    r.serverHit[attacker] = true;
    r.serverFreshHit[attacker] = fresh;
    r.crit[attacker] = bl;
    r.sprintHit[attacker] = bl2;
    r.damageDealt[attacker] = dealt;
}

void Sim::sendMeta(int i) {
    const Player& e = sv[i].ent;
    Pkt p{};
    p.type = Pkt::S_Meta;
    p.sprinting = e.sprinting;
    p.crouching = e.pose == Pose::Crouching;
    p.health = e.health;
    sendToClient(1 - i, p);
}

void Sim::sendAttributes(int i) {
    const Player& e = sv[i].ent;
    Pkt self{};
    self.type = Pkt::S_Attributes;
    self.aboutSelf = true;
    self.sprintMod = e.speedModSprint;
    sendToClient(i, self);
    Pkt watcher{};
    watcher.type = Pkt::S_Attributes;
    watcher.aboutSelf = false;
    watcher.sprintMod = e.speedModSprint;
    sendToClient(1 - i, watcher);
}

void Sim::trackerTick(int i) {
    ServerPlayer& s = sv[i];
    Player& e = s.ent;

    if (s.trackerTickCount % 2 == 0 || e.hasImpulse || s.metaDirty) {
        int8_t by = Mth::packDegrees(e.yaw);
        int8_t by2 = Mth::packDegrees(e.pitch);
        bool rotated = std::abs((int)by - (int)s.lastSentYRot) >= 1 ||
                       std::abs((int)by2 - (int)s.lastSentXRot) >= 1;

        ++s.teleportDelay;
        double ddx = e.posX - s.codecBaseX;
        double ddy = e.posY - s.codecBaseY;
        double ddz = e.posZ - s.codecBaseZ;
        bool moved = ddx * ddx + ddy * ddy + ddz * ddz >= 7.62939453125E-6;
        bool bl5 = moved || s.trackerTickCount % 60 == 0;
        int64_t l = javaRound(e.posX * 4096.0) - javaRound(s.codecBaseX * 4096.0);
        int64_t l2 = javaRound(e.posY * 4096.0) - javaRound(s.codecBaseY * 4096.0);
        int64_t l3 = javaRound(e.posZ * 4096.0) - javaRound(s.codecBaseZ * 4096.0);
        bool overflow = l < -32768 || l > 32767 || l2 < -32768 || l2 > 32767 ||
                        l3 < -32768 || l3 > 32767;

        bool sent = false, posUpdated = false, rotUpdated = false;
        Pkt p{};
        if (overflow || s.teleportDelay > 400 || s.wasOnGround != e.onGround) {
            s.wasOnGround = e.onGround;
            s.teleportDelay = 0;
            p.type = Pkt::S_PosSync;
            p.x = e.posX; p.y = e.posY; p.z = e.posZ;
            p.yaw = e.yaw; p.pitch = e.pitch;
            p.onGround = e.onGround;
            sent = posUpdated = rotUpdated = true;
        } else if (bl5 && rotated) {
            p.type = Pkt::S_MoveRel;
            p.hasPos = p.hasLook = true;
            p.dx = (int16_t)l; p.dy = (int16_t)l2; p.dz = (int16_t)l3;
            p.byaw = by; p.bpitch = by2;
            p.onGround = e.onGround;
            sent = posUpdated = rotUpdated = true;
        } else if (bl5) {
            p.type = Pkt::S_MoveRel;
            p.hasPos = true;
            p.dx = (int16_t)l; p.dy = (int16_t)l2; p.dz = (int16_t)l3;
            p.onGround = e.onGround;
            sent = posUpdated = true;
        } else if (rotated) {
            p.type = Pkt::S_MoveRel;
            p.hasLook = true;
            p.byaw = by; p.bpitch = by2;
            p.onGround = e.onGround;
            sent = rotUpdated = true;
        }

        if (e.hasImpulse) {
            Vec3 dm = e.deltaMovement();
            double d = dm.distanceToSqr(s.lastSentMovement);
            if (d > 1.0E-7 || (d > 0.0 && dm.lengthSqr() == 0.0)) {
                s.lastSentMovement = dm;
                Pkt vp{};
                vp.type = Pkt::S_Motion;
                vp.aboutSelf = false;
                vp.mx = (int16_t)javaIntCast(Mth::clamp(dm.x, -3.9, 3.9) * 8000.0);
                vp.my = (int16_t)javaIntCast(Mth::clamp(dm.y, -3.9, 3.9) * 8000.0);
                vp.mz = (int16_t)javaIntCast(Mth::clamp(dm.z, -3.9, 3.9) * 8000.0);
                sendToClient(1 - i, vp);
            }
        }

        if (sent) sendToClient(1 - i, p);
        if (posUpdated) {
            s.codecBaseX = e.posX;
            s.codecBaseY = e.posY;
            s.codecBaseZ = e.posZ;
        }
        if (rotUpdated) {
            s.lastSentYRot = by;
            s.lastSentXRot = by2;
        }
        if (s.metaDirty) {
            sendMeta(i);
            s.metaDirty = false;
        }
        if (s.attrDirty) {
            sendAttributes(i);
            s.attrDirty = false;
        }
        e.hasImpulse = false;
    }
    ++s.trackerTickCount;

    if (s.hurtMarked) {
        s.hurtMarked = false;
        Pkt toSelf{};
        toSelf.type = Pkt::S_Motion;
        toSelf.aboutSelf = true;
        toSelf.mx = (int16_t)javaIntCast(Mth::clamp(e.motionX, -3.9, 3.9) * 8000.0);
        toSelf.my = (int16_t)javaIntCast(Mth::clamp(e.motionY, -3.9, 3.9) * 8000.0);
        toSelf.mz = (int16_t)javaIntCast(Mth::clamp(e.motionZ, -3.9, 3.9) * 8000.0);
        sendToClient(i, toSelf);
        Pkt toWatcher = toSelf;
        toWatcher.aboutSelf = false;
        sendToClient(1 - i, toWatcher);
    }
}

void Sim::serverConnectionTick(int i, StepResult& r) {
    ServerPlayer& s = sv[i];
    Player& e = s.ent;

    s.firstGoodX = e.posX;
    s.firstGoodY = e.posY;
    s.firstGoodZ = e.posZ;
    double sx = e.posX, sy = e.posY, sz = e.posZ;
    e.prevPosX = e.posX; e.prevPosY = e.posY; e.prevPosZ = e.posZ;

    bool equipChanged = false;
    for (int g = 0; g < 5; ++g) {
        const ItemDur& it = g < 4 ? e.gear.armor[g] : e.gear.sword;
        if (s.lastEquipDamage[g] != (int16_t)it.damage || s.lastEquipBroken[g] != it.broken) {
            equipChanged = true;
            s.lastEquipDamage[g] = (int16_t)it.damage;
            s.lastEquipBroken[g] = it.broken;
        }
    }
    if (equipChanged) {
        Pkt p{};
        p.type = Pkt::S_Equipment;
        for (int g = 0; g < 5; ++g) {
            p.gearDamage[g] = s.lastEquipDamage[g];
            p.gearBroken[g] = s.lastEquipBroken[g];
        }
        sendToClient(1 - i, p);
    }

    Pose prePose = e.pose;
    bool preSprintMod = e.speedModSprint;
    serverPlayerAiStep(arena, e, sv[1 - i].ent);

    float preHealth = e.health;
    foodTick(e);
    if (e.health != preHealth) s.metaDirty = true;
    if (e.dead && preHealth > 0.0f) r.death[i] = true;

    ++e.attackStrengthTicker;
    updatePlayerPose(arena, e);
    if (e.pose != prePose) s.metaDirty = true;
    if (e.speedModSprint != preSprintMod) s.attrDirty = true;

    if (e.health != s.lastSentHealth || s.lastSentFood != e.food.foodLevel ||
        (e.food.saturationLevel == 0.0f) != s.lastFoodSaturationZero) {
        Pkt p{};
        p.type = Pkt::S_Health;
        p.health = e.health;
        p.food = e.food.foodLevel;
        p.saturation = e.food.saturationLevel;
        sendToClient(i, p);
        s.lastSentHealth = e.health;
        s.lastSentFood = e.food.foodLevel;
        s.lastFoodSaturationZero = e.food.saturationLevel == 0.0f;
    }

    setPosition(e, sx, sy, sz);
}

}
