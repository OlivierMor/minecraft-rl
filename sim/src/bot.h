#pragma once
#include <cmath>
#include "sim.h"

namespace mc1218 {

class SimpleBot {
public:
    float maxTurnPerTick = 40.0f;
    float minCharge = 0.92f;
    int clickInterval = 3;

    Input act(const ClientSim& me, int tickCount) {
        const Player& self = me.self;
        const RemoteView& tgt = me.remote;
        Input in;

        double dx = tgt.posX - self.posX;
        double dz = tgt.posZ - self.posZ;
        double horiz = std::sqrt(dx * dx + dz * dz);

        float targetYaw = (float)(std::atan2(dz, dx) * 180.0 / 3.141592653589793) - 90.0f;
        float dyaw = MathHelper::wrapAngleTo180_float(targetYaw - self.yaw);
        in.yawDelta = MathHelper::clamp_float(dyaw, -maxTurnPerTick, maxTurnPerTick);

        double dy = (tgt.posY + 1.2) - (self.posY + (double)self.eyeHeight());
        float targetPitch = (float)(-(std::atan2(dy, horiz) * 180.0 / 3.141592653589793));
        in.pitchDelta =
            MathHelper::clamp_float(targetPitch - self.pitch, -maxTurnPerTick, maxTurnPerTick);

        in.forward = true;
        in.sprintKey = true;

        bool facing = std::fabs(dyaw) < 25.0f;
        bool charged = self.attackStrengthScale(0.5f) > minCharge;
        bool paced = clickInterval <= 1 || (tickCount % clickInterval) == 0;
        if (horiz < 3.1 && facing && charged && paced) in.attack = true;

        return in;
    }
};

}
