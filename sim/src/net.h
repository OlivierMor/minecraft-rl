#pragma once
#include <cstdint>
#include <deque>

namespace mc1218 {

struct Pkt {
    enum Type : uint8_t {
        C_Input,
        C_Move,
        C_Command,
        C_Attack,
        C_Swing,
        C_TickEnd,
        S_MoveRel,
        S_PosSync,
        S_Motion,
        S_Meta,
        S_Attributes,
        S_Equipment,
        S_OwnSlots,
        S_Health,
        S_DamageEvent,
        S_SwingAnim,
        S_Score,
    } type;

    bool kForward = false, kBack = false, kLeft = false, kRight = false;
    bool kJump = false, kShift = false, kSprint = false;

    bool hasPos = false, hasLook = false;
    bool onGround = false, horizontalCollision = false;
    double x = 0, y = 0, z = 0;
    float yaw = 0, pitch = 0;

    enum Action : uint8_t { StartSprint, StopSprint };
    uint8_t action = 0;

    bool secondaryAction = false;

    int16_t dx = 0, dy = 0, dz = 0;
    int8_t byaw = 0, bpitch = 0;


    bool aboutSelf = false;
    int16_t mx = 0, my = 0, mz = 0;

    bool sprinting = false, crouching = false;
    float health = 0.0f;

    bool sprintMod = false;

    int16_t gearDamage[5] = {0, 0, 0, 0, 0};
    bool gearBroken[5] = {false, false, false, false, false};

    int food = 20;
    float saturation = 5.0f;

    int hits0 = 0, hits1 = 0;
};

class Channel {
public:
    void send(int now, int delayTicks, const Pkt& p) {
        int at = now + delayTicks;
        if (at < lastAt_) at = lastAt_;
        lastAt_ = at;
        q_.push_back({at, p});
    }

    template <class F>
    void drain(int now, F&& handle) {
        while (!q_.empty() && q_.front().at <= now) {
            Pkt p = q_.front().pkt;
            q_.pop_front();
            handle(p);
        }
    }

    void clear() {
        q_.clear();
        lastAt_ = 0;
    }

    size_t pending() const { return q_.size(); }

private:
    struct Item { int at; Pkt pkt; };
    std::deque<Item> q_;
    int lastAt_ = 0;
};

}
