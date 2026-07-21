#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>

#include "env/sim_api.h"
#include "config.h"
#include "env/env.h"
#include "env/registry.h"
#include "model/distributions.h"
#include "model/policy.h"
#include "replay.h"
#include "runtime/checkpoint.h"

#include "mc_render.h"

namespace fs = std::filesystem;
using namespace rl;

namespace {

struct Side {
    bool scripted = true;
    PolicyNet net{nullptr};
    std::unique_ptr<ObsBuilder> obs;
    std::unique_ptr<ActionParser> parser;
    mc::SimpleBot bot;
    mc::Input last;
    torch::Tensor buf;

    mc::Input act(const mc::Sim& sim, int side) {
        if (scripted) return bot.act(sim.client[side], sim.tickCount);
        torch::NoGradGuard ng;
        obs->build(sim.client[side], last, sim.tickCount, sim.pingMs(side),
                   buf.data_ptr<float>());
        torch::Tensor a = sampleActions(net->logitsOnly(buf), parser->spec(), true)
                              .to(torch::kInt32);
        mc::Input in = parser->parse(a.data_ptr<int32_t>());
        last = in;
        return in;
    }
    void reset(const mc::Sim& sim, int side) {
        last = mc::Input{};
        if (obs) obs->reset(sim.client[side]);
        if (parser) parser->reset();
    }
};

}

int main(int argc, char** argv) try {
    std::string refA, refB, cfgPath, replayPath;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--config") cfgPath = next();
        else if (a == "--replay") replayPath = next();
        else pos.push_back(a);
    }
    bool replayMode = !replayPath.empty();
    refA = pos.size() > 0 ? pos[0] : "bot";
    refB = pos.size() > 1 ? pos[1] : "bot";
    auto tryRunCfg = [&](const std::string& r) {
        if (cfgPath.empty() && fs::is_directory(r) && fs::exists(r + "/config.toml"))
            cfgPath = r + "/config.toml";
    };
    tryRunCfg(refA);
    tryRunCfg(refB);
    if (cfgPath.empty()) cfgPath = MC_ENV_DEFAULT_CONFIG;
    Config cfg = Config::fromFile(cfgPath);

    torch::set_num_threads(1);

    std::unique_ptr<Env> env;
    mc::Sim rawSim;
    Replay rep;
    Side sides[2];
    if (replayMode) {
        auto loaded = Replay::load(replayPath);
        if (!loaded) throw std::runtime_error("cannot load replay: " + replayPath);
        rep = *loaded;
        rep.applySetup(rawSim);
    } else {
        env = std::make_unique<Env>(0, (uint64_t)time(nullptr), registry::makeComponents(cfg));
        for (int s = 0; s < 2; ++s) {
            const std::string& ref = s == 0 ? refA : refB;
            if (ref == "bot") continue;
            sides[s].scripted = false;
            sides[s].obs = registry::makeObs(cfg.str("components.obs", "standard"), cfg, s);
            sides[s].parser =
                registry::makeParser(cfg.str("components.actions", "standard"), cfg);
            sides[s].net = buildPolicy(cfg, sides[s].obs->size(), sides[s].parser->spec());
            if (!loadPolicyWeights(ref, sides[s].net))
                throw std::runtime_error("cannot load policy from: " + ref);
            sides[s].net->eval();
            sides[s].buf = torch::zeros({1, sides[s].obs->size()}, torch::kFloat32);
            sides[s].reset(env->sim, s);
        }
    }
    mc::Sim& sim = replayMode ? rawSim : env->sim;

    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1440, 900, replayMode ? "sword duel - replay" : "sword duel - agent vs agent");
    SetTargetFPS(120);
    mcview::init();

    float focusX = (float)mc::Arena::CENTER_X, focusZ = (float)mc::Arena::CENTER_Z;
    float camYaw = 45, camPitch = 35, camDist = 26;
    int fpView = -1;
    float fpFov = 70.0f;
    if (const char* v = std::getenv("MCRL_VIEW")) fpView = std::atoi(v);
    float aimSmoothTau = (float)cfg.num("viewer.aim_smooth", 0.06);
    if (const char* v = std::getenv("MCRL_AIM_SMOOTH")) aimSmoothTau = (float)std::atof(v);
    double speed = 1.0, acc = 0;
    bool paused = false, showHitboxes = false;
    size_t repTick = 0;
    int lastWinner = -1;
    double winnerFlash = 0;
    mcview::PlayerAnim anims[2];
    mcview::AimSmoother selfAim[2], foeAim[2];
    const Color accents[2] = {Color{120, 150, 255, 255}, Color{255, 160, 90, 255}};
    const char* names[2] = {"BLUE", "ORANGE"};

    const char* shotPath = std::getenv("MCRL_SHOT");
    int shotFrame = std::getenv("MCRL_SHOT_FRAME") ? std::atoi(std::getenv("MCRL_SHOT_FRAME")) : 120;
    int frames = 0;

    while (!WindowShouldClose()) {
        double dt = GetFrameTime();
        if (dt > 0.25) dt = 0.25;
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_H)) showHitboxes = !showHitboxes;
        if (IsKeyPressed(KEY_MINUS)) speed = std::max(0.25, speed * 0.5);
        if (IsKeyPressed(KEY_EQUAL)) speed = std::min(16.0, speed * 2.0);
        if (IsKeyPressed(KEY_R) && !replayMode) {
            env->reset();
            for (int s = 0; s < 2; ++s) sides[s].reset(sim, s);
            for (int s = 0; s < 2; ++s) { selfAim[s].reset(); foeAim[s].reset(); }
        }
        if (IsKeyPressed(KEY_V)) fpView = fpView >= 1 ? -1 : fpView + 1;
        if (fpView < 0 && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 md = GetMouseDelta();
            camYaw += md.x * 0.4f;
            camPitch = std::clamp(camPitch + md.y * 0.3f, 5.0f, 85.0f);
        }
        camDist = std::clamp(camDist - GetMouseWheelMove() * 2.0f, 6.0f, 60.0f);

        {
            float mx = (float)(sim.client[0].self.posX + sim.client[1].self.posX) / 2.0f;
            float mz = (float)(sim.client[0].self.posZ + sim.client[1].self.posZ) / 2.0f;
            float k = 1.0f - std::exp((float)(-dt * 3.0));
            focusX += (mx - focusX) * k;
            focusZ += (mz - focusZ) * k;
        }

        auto doTick = [&]() {
            if (replayMode) {
                if (repTick < rep.ticks.size()) {
                    if (repTick < rep.pings.size()) {
                        sim.setPingMs(0, rep.pings[repTick][0]);
                        sim.setPingMs(1, rep.pings[repTick][1]);
                    }
                    sim.step(rep.ticks[repTick][0], rep.ticks[repTick][1]);
                    ++repTick;
                }
                return;
            }
            mc::Input a = sides[0].act(sim, 0);
            mc::Input b = sides[1].act(sim, 1);
            Env::StepOut out = env->step(a, b);
            if (out.done) {
                lastWinner = out.winner;
                winnerFlash = 2.5;
                env->reset();
                for (int s = 0; s < 2; ++s) sides[s].reset(sim, s);
                for (int s = 0; s < 2; ++s) { selfAim[s].reset(); foeAim[s].reset(); }
            }
        };
        if (!paused) {
            acc += dt * speed;
            int maxTicks = 40;
            while (acc >= 0.05 && maxTicks-- > 0) {
                acc -= 0.05;
                doTick();
            }
        } else if (IsKeyPressed(KEY_N)) {
            doTick();
        }
        winnerFlash = std::max(0.0, winnerFlash - dt);

        double alpha = std::clamp(acc / 0.05, 0.0, 1.0);
        auto lerp = [&](double p, double c) { return (float)(p + (c - p) * alpha); };
        for (int s = 0; s < 2; ++s) {
            const mc::Player& p = sim.client[s].self;
            anims[s].update((float)(p.posX - p.prevPosX), (float)(p.posZ - p.prevPosZ),
                            (float)dt);
            selfAim[s].update(p.yaw, p.pitch, (float)dt, aimSmoothTau);
            const mc::RemoteView& r = sim.client[s].remote;
            foeAim[s].update(r.yaw, r.pitch, (float)dt, aimSmoothTau);
        }

        Camera3D cam{};
        cam.up = Vector3{0, 1, 0};
        cam.projection = CAMERA_PERSPECTIVE;
        if (fpView < 0) {
            float cy = camYaw * DEG2RAD, cp = camPitch * DEG2RAD;
            cam.position = Vector3{focusX + camDist * std::cos(cp) * std::cos(cy),
                                   1.0f + camDist * std::sin(cp),
                                   focusZ + camDist * std::cos(cp) * std::sin(cy)};
            cam.target = Vector3{focusX, 1, focusZ};
            cam.fovy = 55;
        } else {
            const mc::Player& me = sim.client[fpView].self;
            float fovTarget = me.sprinting ? 77.0f : 70.0f;
            fpFov += (fovTarget - fpFov) * (float)std::min(1.0, dt * 12.0);
            Vector3 eye{lerp(me.prevPosX, me.posX),
                        lerp(me.prevPosY, me.posY) + me.eyeHeight(),
                        lerp(me.prevPosZ, me.posZ)};
            mc::Vec3 lookv =
                mc::getVectorForRotation(selfAim[fpView].pitch, selfAim[fpView].yaw);
            cam.position = eye;
            cam.target = Vector3{eye.x + (float)lookv.x, eye.y + (float)lookv.y,
                                 eye.z + (float)lookv.z};
            cam.fovy = fpFov;
        }

        BeginDrawing();
        ClearBackground(mcview::skyColor());
        BeginMode3D(cam);
        mcview::drawWorld(cam);
        if (fpView < 0) {
            for (int s = 0; s < 2; ++s) {
                const mc::Player& p = sim.client[s].self;
                mcview::PlayerPose pose;
                pose.pos = Vector3{lerp(p.prevPosX, p.posX), lerp(p.prevPosY, p.posY),
                                   lerp(p.prevPosZ, p.posZ)};
                pose.yaw = selfAim[s].yaw;
                pose.pitch = selfAim[s].pitch;
                pose.crouching = p.pose == mc::Pose::Crouching;
                pose.hurt = p.hurtTime > 0;
                pose.swing = p.swinging
                                 ? std::min(1.0f, ((float)p.swingTime + (float)alpha) / 6.0f)
                                 : -1.0f;
                pose.anim = anims[s];
                pose.accent = accents[s];
                mcview::drawPlayer(pose);
                if (showHitboxes) {
                    mcview::drawBoxWires(p.bb, s == 0 ? BLUE : ORANGE);
                    mcview::drawBoxWires(sim.client[1 - s].remote.bb, WHITE);
                }
            }
        } else {
            const mc::Player& me = sim.client[fpView].self;
            const mc::RemoteView& foe = sim.client[fpView].remote;
            mcview::PlayerPose pose;
            pose.pos = Vector3{lerp(foe.prevPosX, foe.posX), lerp(foe.prevPosY, foe.posY),
                               lerp(foe.prevPosZ, foe.posZ)};
            pose.yaw = foeAim[fpView].yaw;
            pose.pitch = foeAim[fpView].pitch;
            pose.crouching = foe.pose == mc::Pose::Crouching;
            pose.hurt = foe.hurtTime > 0;
            pose.swing = foe.swinging
                             ? std::min(1.0f, ((float)foe.swingTime + (float)alpha) / 6.0f)
                             : -1.0f;
            pose.anim = anims[1 - fpView];
            pose.accent = accents[1 - fpView];
            mcview::drawPlayer(pose);
            if (showHitboxes) {
                mcview::drawBoxWires(foe.bb, WHITE);
                mcview::drawBoxWires(sim.client[1 - fpView].self.bb, YELLOW);
            }
            float mySwing = me.swinging
                                ? std::min(1.0f, ((float)me.swingTime + (float)alpha) / 6.0f)
                                : -1.0f;
            mcview::drawFirstPerson(cam, mySwing, anims[fpView].limbSwing,
                                    anims[fpView].limbAmount);
        }
        EndMode3D();

        int w = GetScreenWidth(), h = GetScreenHeight();
        int hs = mcview::hudScale(h);
        char buf[200];
        auto frac = [](const mc::ItemDur& it) {
            return it.broken ? 0.0f : 1.0f - (float)it.damage / (float)it.maxDamage;
        };
        for (int s = 0; s < 2; ++s) {
            const mc::Player& p = sim.sv[s].ent;
            int bw = 80 * hs;
            int bx = s == 0 ? 8 * hs : w - 8 * hs - bw;
            DrawRectangle(bx - 3 * hs, 5 * hs, bw + 6 * hs, 56 * hs, Color{0, 0, 0, 96});
            DrawText(names[s], bx, 8 * hs, 8 * hs, accents[s]);
            std::snprintf(buf, sizeof buf, "%.1f hp", p.health);
            DrawText(buf, bx + bw - MeasureText(buf, 6 * hs), 9 * hs, 6 * hs, WHITE);
            mcview::drawArmorRow(bx, 18 * hs, p.gear.armorValue(), hs);
            mcview::drawHearts(bx, 28 * hs, p.health, hs);
            mcview::drawHunger(bx + 80 * hs, 38 * hs, p.food.foodLevel, hs);
            mcview::drawBar(bx, 50 * hs, bw, 2 * hs, p.attackStrengthScale(0.5f),
                            Color{240, 240, 240, 255});
            mcview::drawBar(bx, 53 * hs, bw, 2 * hs, frac(p.gear.sword),
                            Color{110, 200, 255, 255});
        }
        if (fpView >= 0) {
            const mc::Player& me = sim.client[fpView].self;
            mcview::drawDamageOverlay(w, h, (float)me.hurtTime / 10.0f);
            mcview::drawCrosshair(w, h, false);
            mcview::drawAttackIndicator(w, h, me.attackStrengthScale(0.5f));
            std::snprintf(buf, sizeof buf, "first person: %s", names[fpView]);
            DrawText(buf, (w - MeasureText(buf, 18)) / 2, h - 48, 18, accents[fpView]);
        }
        std::snprintf(buf, sizeof buf, "to the death   (hits %d - %d)", sim.sv[0].hits,
                      sim.sv[1].hits);
        DrawText(buf, (w - MeasureText(buf, 20)) / 2, 12, 20, WHITE);
        std::snprintf(buf, sizeof buf, "tick %d   speed %.2gx%s   ping %d/%d ms", sim.tickCount,
                      speed, paused ? "   PAUSED" : "", sim.pingMs(0), sim.pingMs(1));
        DrawText(buf, (w - MeasureText(buf, 18)) / 2, 40, 18, Color{235, 235, 235, 255});
        if (replayMode) {
            std::snprintf(buf, sizeof buf, "replay %zu / %zu ticks", repTick, rep.ticks.size());
            DrawText(buf, (w - MeasureText(buf, 18)) / 2, 62, 18, Color{180, 220, 180, 255});
        }
        DrawText("drag orbit | scroll zoom | V first person | SPACE pause | N step | -/= speed | "
                 "H hitboxes | R new",
                 16, h - 24, 14, Color{220, 220, 220, 190});
        if (winnerFlash > 0) {
            const char* msg = lastWinner == 0 ? "BLUE WINS" : lastWinner == 1 ? "ORANGE WINS"
                                                                              : "DRAW";
            Color mc_ = lastWinner == 0 ? accents[0] : lastWinner == 1 ? accents[1] : WHITE;
            int fs = 12 * hs;
            int mw = MeasureText(msg, fs);
            DrawText(msg, (w - mw) / 2 + hs, h / 2 - fs / 2 + hs, fs, Color{20, 20, 20, 200});
            DrawText(msg, (w - mw) / 2, h / 2 - fs / 2, fs, mc_);
        }
        EndDrawing();

        if (shotPath && ++frames == shotFrame) {
            TakeScreenshot(shotPath);
            break;
        }
    }
    mcview::unload();
    CloseWindow();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
