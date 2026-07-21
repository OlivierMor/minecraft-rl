#include "env/standard_actions.h"

namespace rl {

namespace {
inline float ramp(float desired, float prev, float accel) {
    if (accel <= 0) return desired;
    float lo = prev - accel, hi = prev + accel;
    return desired < lo ? lo : (desired > hi ? hi : desired);
}
}

StandardActions::StandardActions(const Config& cfg) {
    auto y = cfg.numArr("actions.yaw_buckets");
    auto p = cfg.numArr("actions.pitch_buckets");
    yaw_.assign(y.begin(), y.end());
    pitch_.assign(p.begin(), p.end());
    if (yaw_.empty()) yaw_ = {-45, -20, -10, -4, -1, 0, 1, 4, 10, 20, 45};
    if (pitch_.empty()) pitch_ = {-20, -8, -3, -1, 0, 1, 3, 8, 20};
    allowSneak_ = cfg.boolean("actions.allow_sneak", false);
    yawAccel_ = (float)cfg.num("actions.yaw_accel", 0);
    pitchAccel_ = (float)cfg.num("actions.pitch_accel", 0);
}

ActionSpec StandardActions::spec() const {
    ActionSpec s;
    s.branches = {9, 2, 2, 2, (int)yaw_.size(), (int)pitch_.size()};
    if (allowSneak_) s.branches.push_back(2);
    return s;
}

void StandardActions::reset() { prevYaw_ = prevPitch_ = 0; }

mc::Input StandardActions::parse(const int32_t* b) {
    mc::Input in;
    switch (b[0]) {
        case 1: in.forward = true; break;
        case 2: in.forward = in.left = true; break;
        case 3: in.forward = in.right = true; break;
        case 4: in.left = true; break;
        case 5: in.right = true; break;
        case 6: in.back = true; break;
        case 7: in.back = in.left = true; break;
        case 8: in.back = in.right = true; break;
        default: break;
    }
    in.jump = b[1] == 1;
    in.sprintKey = b[2] == 1;
    in.attack = b[3] == 1;
    if (allowSneak_) in.sneak = b[6] == 1;

    prevYaw_ = ramp(yaw_[(size_t)b[4]], prevYaw_, yawAccel_);
    prevPitch_ = ramp(pitch_[(size_t)b[5]], prevPitch_, pitchAccel_);
    in.yawDelta = prevYaw_;
    in.pitchDelta = prevPitch_;
    return in;
}

}
