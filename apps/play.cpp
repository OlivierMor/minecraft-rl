#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <random>
#include <string>

#include "env/sim_api.h"
#include "config.h"
#include "env/registry.h"
#include "model/distributions.h"
#include "model/policy.h"
#include "replay.h"
#include "runtime/checkpoint.h"

#include "mc_render.h"

namespace fs = std::filesystem;
using namespace rl;

int main(int argc, char** argv) try {
    std::string ref = argc > 1 ? argv[1] : "bot";
    std::string cfgPath;
    int ping0 = 0, ping1 = 0;
    bool stochastic = false;
    std::string recordDir = "replays";
    bool recordEnabled = true;
    std::string nameTag;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--config") cfgPath = next();
        else if (a == "--ping0") ping0 = std::stoi(next());
        else if (a == "--ping1") ping1 = std::stoi(next());
        else if (a == "--stochastic") stochastic = true;
        else if (a == "--record") recordDir = next();
        else if (a == "--no-record") recordEnabled = false;
        else if (a == "--name") nameTag = next();
    }
    if (cfgPath.empty() && fs::is_directory(ref) && fs::exists(ref + "/config.toml"))
        cfgPath = ref + "/config.toml";
    if (cfgPath.empty()) cfgPath = MC_ENV_DEFAULT_CONFIG;
    Config cfg = Config::fromFile(cfgPath);

    torch::set_num_threads(1);

    bool scripted = ref == "bot";
    auto obs = registry::makeObs(cfg.str("components.obs", "standard"), cfg, 1);
    auto parser = registry::makeParser(cfg.str("components.actions", "standard"), cfg);
    ActionSpec spec = parser->spec();
    PolicyNet net{nullptr};
    if (!scripted) {
        net = buildPolicy(cfg, obs->size(), spec);
        if (!loadPolicyWeights(ref, net))
            throw std::runtime_error("cannot load policy from: " + ref);
        net->eval();
    }
    mc::SimpleBot bot;
    torch::Tensor obsBuf = torch::zeros({1, obs->size()}, torch::kFloat32);

    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1440, 900, scripted ? "sword duel - you vs SimpleBot"
                                   : "sword duel - you vs trained agent");
    SetExitKey(KEY_NULL);
    DisableCursor();
    mcview::init();

    mc::Sim sim;
    mc::Input lastAgentIn;
    obs->reset(sim.client[1]);
    parser->reset();

    auto reward = registry::makeReward(cfg.str("components.reward", "combined"), cfg);
    reward->reset(sim);
    std::vector<std::pair<std::string, double>> rewNow;
    std::vector<double> rewSum;
    bool showRewards = false;

    const double TICK = 0.05;
    double acc = 0.0;
    bool mouseCaptured = true, showHitboxes = false;
    float liveYaw = sim.client[0].self.yaw, livePitch = sim.client[0].self.pitch;
    float pendingYaw = 0, pendingPitch = 0;
    bool pendingAttack = false;
    double hitFlash = 0, hurtFlash = 0;
    float fov = 70.0f;
    mcview::PlayerAnim selfAnim, foeAnim;
    mcview::AimSmoother foeAim;
    float aimSmoothTau = (float)cfg.num("viewer.aim_smooth", 0.06);
    if (const char* v = std::getenv("MCRL_AIM_SMOOTH")) aimSmoothTau = (float)std::atof(v);

    Replay rec;
    bool recActive = false, recSaved = false;
    int matchIdx = 0;
    std::mt19937_64 seedGen{std::random_device{}()};
    char stamp[32];
    std::time_t t0 = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&t0));
    std::string runStamp = stamp;

    auto startMatch = [&]() {
        if (!recordEnabled) return;
        uint64_t seed = seedGen();
        sim.reseed(seed);
        rec = Replay{};
        rec.setup.push_back({SetupOp::Seed, 0, 0, 0, 0, 0.f, 0.f, 0, seed});
        rec.setup.push_back({SetupOp::Ping, 0, 0, 0, 0, 0.f, 0.f, ping0, 0});
        rec.setup.push_back({SetupOp::Ping, 1, 0, 0, 0, 0.f, 0.f, ping1, 0});
        recActive = true;
        recSaved = false;
    };
    auto finalizeMatch = [&]() {
        if (!recordEnabled || !recActive || recSaved || rec.ticks.empty()) return;
        std::error_code ec;
        fs::create_directories(recordDir, ec);
        char path[512];
        std::snprintf(path, sizeof path, "%s/play-%s-%03d.mcrp", recordDir.c_str(),
                      runStamp.c_str(), matchIdx++);
        if (rec.save(path)) {
            std::printf("saved replay: %s  (rewatch: watch --replay %s, press V for "
                        "agent POV)\n",
                        path, path);
            std::fflush(stdout);
        }
        recSaved = true;
        recActive = false;
    };
    if (recordEnabled)
        std::printf("recording matches to %s/  (rewatch with: watch --replay <file>, "
                    "press V for agent POV)\n",
                    recordDir.c_str());
    startMatch();

    const char* shotPath = std::getenv("MCRL_SHOT");
    int shotFrame = std::getenv("MCRL_SHOT_FRAME") ? std::atoi(std::getenv("MCRL_SHOT_FRAME")) : 120;
    int frames = 0;

    while (!WindowShouldClose()) {
        double dt = GetFrameTime();
        if (dt > 0.25) dt = 0.25;
        if (IsKeyPressed(KEY_TAB) || IsKeyPressed(KEY_ESCAPE)) {
            mouseCaptured = !mouseCaptured;
            if (mouseCaptured) DisableCursor(); else EnableCursor();
        }
        if (IsKeyPressed(KEY_H)) showHitboxes = !showHitboxes;
        if (IsKeyPressed(KEY_G)) showRewards = !showRewards;
        if (IsKeyPressed(KEY_R)) {
            finalizeMatch();
            sim.reset();
            obs->reset(sim.client[1]);
            parser->reset();
            reward->reset(sim);
            std::fill(rewSum.begin(), rewSum.end(), 0.0);
            foeAim.reset();
            lastAgentIn = mc::Input{};
            liveYaw = sim.client[0].self.yaw;
            livePitch = sim.client[0].self.pitch;
            pendingYaw = pendingPitch = 0;
            pendingAttack = false;
            startMatch();
        }
        if (IsKeyPressed(KEY_MINUS)) ping0 = std::max(0, ping0 - 25);
        if (IsKeyPressed(KEY_EQUAL)) ping0 = std::min(1000, ping0 + 25);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) ping1 = std::max(0, ping1 - 25);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) ping1 = std::min(1000, ping1 + 25);
        sim.setPingMs(0, ping0);
        sim.setPingMs(1, ping1);

        if (mouseCaptured) {
            Vector2 md = GetMouseDelta();
            float scale = 0.15f;
            pendingYaw += md.x * scale;
            pendingPitch += md.y * scale;
            liveYaw += md.x * scale;
            livePitch = mc::MathHelper::clamp_float(livePitch + md.y * scale, -90.0f, 90.0f);
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) pendingAttack = true;
        }

        bool matchOver =
            sim.client[0].self.health <= 0.0f || sim.client[0].remote.health <= 0.0f;
        acc += dt;
        while (acc >= TICK) {
            acc -= TICK;
            if (matchOver) {
                pendingYaw = pendingPitch = 0;
                pendingAttack = false;
                continue;
            }
            mc::Input in;
            if (mouseCaptured) {
                in.forward = IsKeyDown(KEY_W);
                in.back = IsKeyDown(KEY_S);
                in.left = IsKeyDown(KEY_A);
                in.right = IsKeyDown(KEY_D);
                in.jump = IsKeyDown(KEY_SPACE);
                in.sneak = IsKeyDown(KEY_LEFT_SHIFT);
                in.sprintKey = IsKeyDown(KEY_LEFT_CONTROL);
                in.attack = pendingAttack;
            }
            in.yawDelta = pendingYaw;
            in.pitchDelta = pendingPitch;
            pendingYaw = pendingPitch = 0;
            pendingAttack = false;

            mc::Input agentIn;
            if (scripted) {
                agentIn = bot.act(sim.client[1], sim.tickCount);
            } else {
                torch::NoGradGuard ng;
                obs->build(sim.client[1], lastAgentIn, sim.tickCount, sim.pingMs(1),
                           obsBuf.data_ptr<float>());
                torch::Tensor logits = net->logitsOnly(obsBuf);
                torch::Tensor a = sampleActions(logits, spec, !stochastic).to(torch::kInt32);
                agentIn = parser->parse(a.data_ptr<int32_t>());
            }
            lastAgentIn = agentIn;

            if (recActive) {
                rec.ticks.push_back({in, agentIn});
                rec.pings.push_back({(int32_t)sim.pingMs(0), (int32_t)sim.pingMs(1)});
            }
            mc::StepResult r = sim.step(in, agentIn);
            reward->step(sim, r, 0);
            rewNow.clear();
            reward->collectStats(rewNow);
            if (rewSum.size() != rewNow.size()) rewSum.assign(rewNow.size(), 0.0);
            for (size_t i = 0; i < rewNow.size(); ++i) rewSum[i] += rewNow[i].second;
            liveYaw = sim.client[0].self.yaw;
            livePitch = sim.client[0].self.pitch;
            if (r.clientAttack[0]) hitFlash = 0.18;
            if (r.hurtFlash[0]) hurtFlash = 0.30;
        }
        hitFlash = std::max(0.0, hitFlash - dt);
        hurtFlash = std::max(0.0, hurtFlash - dt);

        double alpha = acc / TICK;
        const mc::Player& me = sim.client[0].self;
        const mc::RemoteView& foe = sim.client[0].remote;
        float fovTarget = me.sprinting ? 77.0f : 70.0f;
        fov += (fovTarget - fov) * (float)std::min(1.0, dt * 12.0);

        selfAnim.update((float)(me.posX - me.prevPosX), (float)(me.posZ - me.prevPosZ),
                        (float)dt);
        foeAnim.update((float)(foe.posX - foe.prevPosX), (float)(foe.posZ - foe.prevPosZ),
                       (float)dt);
        foeAim.update(foe.yaw, foe.pitch, (float)dt, aimSmoothTau);

        auto lerp = [&](double p, double c) { return (float)(p + (c - p) * alpha); };
        Vector3 eye{lerp(me.prevPosX, me.posX), lerp(me.prevPosY, me.posY) + me.eyeHeight(),
                    lerp(me.prevPosZ, me.posZ)};
        mc::Vec3 lookv = mc::getVectorForRotation(livePitch, liveYaw);
        Camera3D cam{};
        cam.position = eye;
        cam.target = Vector3{eye.x + (float)lookv.x, eye.y + (float)lookv.y,
                             eye.z + (float)lookv.z};
        cam.up = Vector3{0, 1, 0};
        cam.fovy = fov;
        cam.projection = CAMERA_PERSPECTIVE;

        BeginDrawing();
        ClearBackground(mcview::skyColor());
        BeginMode3D(cam);
        mcview::drawWorld(cam);

        Vector3 foePos{lerp(foe.prevPosX, foe.posX), lerp(foe.prevPosY, foe.posY),
                       lerp(foe.prevPosZ, foe.posZ)};
        bool foeCrouching = foe.pose == mc::Pose::Crouching;
        bool foeDead = foe.health <= 0.0f;
        if (!foeDead) {
            mcview::PlayerPose pose;
            pose.pos = foePos;
            pose.yaw = foeAim.yaw;
            pose.pitch = foeAim.pitch;
            pose.crouching = foeCrouching;
            pose.hurt = foe.hurtTime > 0;
            pose.swing = foe.swinging
                             ? std::min(1.0f, ((float)foe.swingTime + (float)alpha) / 6.0f)
                             : -1.0f;
            pose.anim = foeAnim;
            mcview::drawPlayer(pose);
        }
        if (showHitboxes && !foeDead) {
            mcview::drawBoxWires(foe.bb, WHITE);
            mcview::drawBoxWires(sim.client[1].self.bb, YELLOW);
        }
        float mySwing = me.swinging
                            ? std::min(1.0f, ((float)me.swingTime + (float)alpha) / 6.0f)
                            : -1.0f;
        mcview::drawFirstPerson(cam, mySwing, selfAnim.limbSwing, selfAnim.limbAmount);
        EndMode3D();

        const char* foeName =
            !nameTag.empty() ? nameTag.c_str() : (scripted ? "SimpleBot" : "Agent");
        if (!foeDead) mcview::drawNameTag(cam, foePos, foeCrouching, foeName, foe.health);

        int w = GetScreenWidth(), h = GetScreenHeight();
        int s = mcview::hudScale(h);
        mcview::drawDamageOverlay(w, h, (float)(hurtFlash / 0.30));
        mcview::drawCrosshair(w, h, hitFlash > 0);
        mcview::drawAttackIndicator(w, h, me.attackStrengthScale(0.5f));

        auto gfrac = [](const mc::ItemDur& it) {
            return it.broken ? 0.0f : 1.0f - (float)it.damage / (float)it.maxDamage;
        };
        int hotbarY = mcview::drawHotbar(w, h, gfrac(me.gear.sword));
        int rowX = w / 2 - 91 * s;
        mcview::drawHearts(rowX, hotbarY - 11 * s, me.health, s);
        mcview::drawArmorRow(rowX, hotbarY - 21 * s, me.gear.armorValue(), s);
        mcview::drawHunger(w / 2 + 91 * s, hotbarY - 11 * s, me.food.foodLevel, s);

        char buf[160];
        std::snprintf(buf, sizeof buf, "ping  you %d ms | %s %d ms", ping0, foeName, ping1);
        DrawText(buf, w - MeasureText(buf, 16) - 16, h - 24, 16, Color{220, 220, 220, 200});

        std::snprintf(buf, sizeof buf, "%s%s%s", me.sprinting ? "SPRINTING  " : "",
                      me.sneaking ? "SNEAKING  " : "",
                      me.food.foodLevel <= 6 ? "TOO HUNGRY TO SPRINT" : "");
        DrawText(buf, 16, h - 60, 18, Color{255, 235, 120, 230});
        DrawText("WASD move | mouse aim | LMB attack | SPACE jump | CTRL sprint | SHIFT sneak",
                 16, h - 40, 14, Color{220, 220, 220, 180});
        DrawText("-/= your ping | [/] agent ping | H hitboxes | G rewards | R restart | TAB release mouse",
                 16, h - 22, 14, Color{220, 220, 220, 180});
        if (showHitboxes)
            DrawText("white = remote view (your crosshair tests this) | yellow = true position",
                     16, h - 78, 14, Color{220, 220, 220, 200});

        if (showRewards && !rewNow.empty()) {
            const int fs = 16, rowH = 18, px = 14;
            int py = 14;
            int nrows = (int)rewNow.size() + 2;
            DrawRectangle(px - 6, py - 6, 250, nrows * rowH + 10, Color{0, 0, 0, 150});
            DrawText("REWARD (you)      now / episode", px, py, fs, Color{235, 235, 120, 255});
            py += rowH + 2;
            double totNow = 0, totSum = 0;
            char v[32];
            for (size_t i = 0; i < rewNow.size(); ++i) {
                std::string nm = rewNow[i].first;
                if (nm.rfind("rew_", 0) == 0) nm = nm.substr(4);
                double now = rewNow[i].second, sum = rewSum[i];
                totNow += now;
                totSum += sum;
                Color c = now > 1e-6f    ? Color{120, 230, 120, 255}
                          : now < -1e-6f ? Color{240, 130, 120, 255}
                                         : Color{160, 160, 172, 255};
                DrawText(nm.c_str(), px, py, fs, c);
                std::snprintf(v, sizeof v, "%+.3f", now);
                DrawText(v, px + 120, py, fs, c);
                std::snprintf(v, sizeof v, "%+.2f", sum);
                DrawText(v, px + 188, py, fs, c);
                py += rowH;
            }
            DrawText("TOTAL", px, py, fs, WHITE);
            std::snprintf(v, sizeof v, "%+.3f", totNow);
            DrawText(v, px + 120, py, fs, WHITE);
            std::snprintf(v, sizeof v, "%+.2f", totSum);
            DrawText(v, px + 188, py, fs, WHITE);
        }


        int knownWinner = me.health <= 0.0f ? 1 : foe.health <= 0.0f ? 0 : -1;
        if (knownWinner >= 0) {
            finalizeMatch();
            mcview::drawEndOverlay(w, h, knownWinner == 0, "press R to fight again");
        }
        EndDrawing();

        if (shotPath && ++frames == shotFrame) {
            TakeScreenshot(shotPath);
            break;
        }
    }
    finalizeMatch();
    mcview::unload();
    CloseWindow();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
