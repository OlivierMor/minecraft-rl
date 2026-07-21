#pragma once
#include "raylib.h"

#include "sim.h"

namespace mcview {

void init();
void unload();

Color skyColor();
void drawWorld(const Camera3D& cam);


struct PlayerAnim {
    float limbSwing = 0.0f;
    float limbAmount = 0.0f;
    void update(float velX, float velZ, float dt);
};

struct AimSmoother {
    float yaw = 0.0f, pitch = 0.0f;
    bool primed = false;
    void reset() { primed = false; }
    void update(float targetYaw, float targetPitch, float dt, float tau);
};

struct PlayerPose {
    Vector3 pos{};
    float yaw = 0.0f;
    float pitch = 0.0f;
    bool crouching = false;
    bool hurt = false;
    float swing = -1.0f;
    PlayerAnim anim;
    Color accent{255, 255, 255, 255};
};
void drawPlayer(const PlayerPose& p);

void drawNameTag(const Camera3D& cam, Vector3 feetPos, bool crouching, const char* name,
                 float health);

void drawFirstPerson(const Camera3D& cam, float swing, float bobPhase, float bobAmount);

void drawBoxWires(const mc1218::AABB& bb, Color c);

int hudScale(int screenH);

void drawCrosshair(int w, int h, bool hitFlash);
void drawAttackIndicator(int w, int h, float charge);
int drawHotbar(int w, int h, float swordDurFrac);
void drawHearts(int x, int y, float health, int scale);
void drawHunger(int xRight, int y, int food, int scale, unsigned char alpha = 255);
void drawArmorRow(int x, int y, float armorPoints, int scale);
void drawBar(int x, int y, int w, int h, float frac, Color fill);
void drawDamageOverlay(int w, int h, float a);
void drawEndOverlay(int w, int h, bool won, const char* subtitle);

}
