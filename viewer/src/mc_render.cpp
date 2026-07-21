#include "mc_render.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "raymath.h"
#include "rlgl.h"

#include "world.h"

namespace mcview {

namespace {

using mc1218::Arena;

constexpr float PXS = 1.8f / 32.0f;


inline uint32_t hash32(uint32_t x, uint32_t y, uint32_t salt) {
    uint32_t h = x * 374761393u + y * 668265263u + salt * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
inline float hash01(int x, int y, uint32_t salt) {
    return (float)(hash32((uint32_t)x, (uint32_t)y, salt) & 0xFFFF) / 65535.0f;
}

inline Color mul(Color c, float f) {
    return Color{(unsigned char)(c.r * f), (unsigned char)(c.g * f),
                 (unsigned char)(c.b * f), c.a};
}


struct MeshBuilder {
    std::vector<float> pos, uv;
    std::vector<unsigned char> col;

    void vert(Vector3 p, float u, float v, Color c) {
        pos.push_back(p.x); pos.push_back(p.y); pos.push_back(p.z);
        uv.push_back(u); uv.push_back(v);
        col.push_back(c.r); col.push_back(c.g); col.push_back(c.b); col.push_back(c.a);
    }
    void quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d,
              Color ca, Color cb, Color cc, Color cd,
              float u0 = 0, float v0 = 0, float u1 = 1, float v1 = 1) {
        vert(a, u0, v1, ca); vert(b, u1, v1, cb); vert(c, u1, v0, cc);
        vert(a, u0, v1, ca); vert(c, u1, v0, cc); vert(d, u0, v0, cd);
    }
    void quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Color cc,
              float u0 = 0, float v0 = 0, float u1 = 1, float v1 = 1) {
        quad(a, b, c, d, cc, cc, cc, cc, u0, v0, u1, v1);
    }

    Mesh upload() {
        Mesh m{};
        m.vertexCount = (int)pos.size() / 3;
        m.triangleCount = m.vertexCount / 3;
        m.vertices = (float*)MemAlloc((unsigned)(pos.size() * sizeof(float)));
        std::memcpy(m.vertices, pos.data(), pos.size() * sizeof(float));
        m.texcoords = (float*)MemAlloc((unsigned)(uv.size() * sizeof(float)));
        std::memcpy(m.texcoords, uv.data(), uv.size() * sizeof(float));
        m.colors = (unsigned char*)MemAlloc((unsigned)col.size());
        std::memcpy(m.colors, col.data(), col.size());
        UploadMesh(&m, false);
        return m;
    }
};

const Color SH_UP = WHITE;
const Color SH_DN = mul(WHITE, 0.55f);
const Color SH_NS = mul(WHITE, 0.80f);
const Color SH_EW = mul(WHITE, 0.68f);

Mesh makeBox(float wPx, float hPx, float dPx, float pivXPx, float pivYPx, float pivZPx) {
    MeshBuilder b;
    float x0 = -pivXPx * PXS, y0 = -pivYPx * PXS, z0 = -pivZPx * PXS;
    float x1 = x0 + wPx * PXS, y1 = y0 + hPx * PXS, z1 = z0 + dPx * PXS;
    b.quad({x0,y1,z0},{x0,y1,z1},{x1,y1,z1},{x1,y1,z0}, SH_UP);
    b.quad({x0,y0,z0},{x1,y0,z0},{x1,y0,z1},{x0,y0,z1}, SH_DN);
    b.quad({x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1}, SH_NS);
    b.quad({x1,y0,z0},{x0,y0,z0},{x0,y1,z0},{x1,y1,z0}, SH_NS);
    b.quad({x1,y0,z1},{x1,y0,z0},{x1,y1,z0},{x1,y1,z1}, SH_EW);
    b.quad({x0,y0,z0},{x0,y0,z1},{x0,y1,z1},{x0,y1,z0}, SH_EW);
    return b.upload();
}

Mesh makeHead(float sidePx, float inflate) {
    MeshBuilder b;
    float s = sidePx * PXS * inflate;
    float x0 = -s / 2, x1 = s / 2, z0 = -s / 2, z1 = s / 2;
    float yc = (sidePx * PXS) / 2;
    float y0 = yc - s / 2, y1 = yc + s / 2;
    auto U = [](int i) { return i / 6.0f; };
    b.quad({x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1}, SH_NS, U(0), 0, U(1), 1);
    b.quad({x0,y0,z0},{x0,y0,z1},{x0,y1,z1},{x0,y1,z0}, SH_EW, U(1), 0, U(2), 1);
    b.quad({x1,y0,z0},{x0,y0,z0},{x0,y1,z0},{x1,y1,z0}, SH_NS, U(2), 0, U(3), 1);
    b.quad({x1,y0,z1},{x1,y0,z0},{x1,y1,z0},{x1,y1,z1}, SH_EW, U(3), 0, U(4), 1);
    b.quad({x0,y1,z1},{x1,y1,z1},{x1,y1,z0},{x0,y1,z0}, SH_UP, U(4), 0, U(5), 1);
    b.quad({x0,y0,z0},{x1,y0,z0},{x1,y0,z1},{x0,y0,z1}, SH_DN, U(5), 0, U(6), 1);
    return b.upload();
}


Texture2D texFromImage(Image& img) {
    Texture2D t = LoadTextureFromImage(img);
    UnloadImage(img);
    SetTextureFilter(t, TEXTURE_FILTER_POINT);
    return t;
}

Texture2D spriteFromRows(const char* const* rows, int w, int h,
                         const char* keys, const Color* colors, int nColors) {
    Image img = GenImageColor(w, h, BLANK);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            char ch = rows[y][x];
            if (ch == '.') continue;
            for (int k = 0; k < nColors; ++k)
                if (keys[k] == ch) { ImageDrawPixel(&img, x, y, colors[k]); break; }
        }
    return texFromImage(img);
}

int noiseLevel(int x, int y, uint32_t salt) {
    float n = 0.6f * hash01(x / 2, y / 2, salt) + 0.4f * hash01(x, y, salt + 1);
    return n < 0.30f ? 0 : n < 0.62f ? 1 : n < 0.88f ? 2 : 3;
}
Texture2D noiseTexture(const Color shades[4], uint32_t salt) {
    Image img = GenImageColor(16, 16, BLANK);
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) ImageDrawPixel(&img, x, y, shades[noiseLevel(x, y, salt)]);
    return texFromImage(img);
}


struct State {
    bool ready = false;

    Texture2D stone{}, grassTop{}, grassSide{}, dirt{};
    Texture2D head{}, helmet{}, diamond{}, armLimb{}, legLimb{}, sword{};
    Texture2D shadow{};
    Texture2D heartBg{}, heartFull{}, heartHalf{};
    Texture2D shankBg{}, shankFull{}, shankHalf{};
    Texture2D armorIcon{}, armorIconEmpty{};

    Mesh arenaMesh{}, grassMesh{}, skirtMesh{}, dirtSkirtMesh{};
    Material matStone{}, matGrassTop{}, matGrassSide{}, matDirt{};

    Mesh headMesh{}, helmetMesh{}, torsoMesh{}, armMesh{}, legMesh{}, swordMesh{};
    Mesh shadowMesh{};
    Material matHead{}, matHelmet{}, matDiamond{}, matArm{}, matLeg{}, matSword{};
    Material matShadow{};
} S;

Material makeMat(Texture2D tex) {
    Material m = LoadMaterialDefault();
    SetMaterialTexture(&m, MATERIAL_MAP_DIFFUSE, tex);
    return m;
}

void drawPart(Mesh& mesh, Material& mat, Color tint) {
    Color prev = mat.maps[MATERIAL_MAP_DIFFUSE].color;
    mat.maps[MATERIAL_MAP_DIFFUSE].color = tint;
    DrawMesh(mesh, mat, MatrixIdentity());
    mat.maps[MATERIAL_MAP_DIFFUSE].color = prev;
}


bool solidAt(int x, int y, int z) {
    if (y == -1) return Arena::insideDisc(x, z) || Arena::wallRing(x, z);
    if (y >= 0 && y < Arena::WALL_HEIGHT) return Arena::wallRing(x, z);
    return false;
}

float floorAO(int x, int z, int cx, int cz) {
    int nx = x + (cx ? 1 : -1), nz = z + (cz ? 1 : -1);
    bool sideX = solidAt(nx, 0, z), sideZ = solidAt(x, 0, nz), diag = solidAt(nx, 0, nz);
    if (sideX && sideZ) return 0.55f;
    if (sideX || sideZ) return 0.70f;
    if (diag) return 0.80f;
    return 1.0f;
}

Mesh buildArenaMesh() {
    MeshBuilder b;
    const int W = Arena::WALL_HEIGHT;
    for (int x = -4; x < Arena::DIAMETER + 4; ++x)
        for (int z = -4; z < Arena::DIAMETER + 4; ++z) {
            float fx = (float)x, fz = (float)z;
            if (Arena::insideDisc(x, z)) {
                float a00 = floorAO(x, z, 0, 0), a10 = floorAO(x, z, 1, 0);
                float a11 = floorAO(x, z, 1, 1), a01 = floorAO(x, z, 0, 1);
                b.quad({fx, 0, fz + 1}, {fx + 1, 0, fz + 1}, {fx + 1, 0, fz}, {fx, 0, fz},
                       mul(SH_UP, a01), mul(SH_UP, a11), mul(SH_UP, a10), mul(SH_UP, a00));
            } else if (Arena::wallRing(x, z)) {
                b.quad({fx, (float)W, fz + 1}, {fx + 1, (float)W, fz + 1},
                       {fx + 1, (float)W, fz}, {fx, (float)W, fz}, SH_UP);
                for (int y = 0; y < W; ++y) {
                    float fy = (float)y, fy1 = fy + 1;
                    float bot = y == 0 ? 0.72f : 1.0f;
                    if (!solidAt(x, y, z + 1))
                        b.quad({fx, fy, fz + 1}, {fx + 1, fy, fz + 1},
                               {fx + 1, fy1, fz + 1}, {fx, fy1, fz + 1},
                               mul(SH_NS, bot), mul(SH_NS, bot), SH_NS, SH_NS);
                    if (!solidAt(x, y, z - 1))
                        b.quad({fx + 1, fy, fz}, {fx, fy, fz}, {fx, fy1, fz},
                               {fx + 1, fy1, fz},
                               mul(SH_NS, bot), mul(SH_NS, bot), SH_NS, SH_NS);
                    if (!solidAt(x + 1, y, z))
                        b.quad({fx + 1, fy, fz + 1}, {fx + 1, fy, fz},
                               {fx + 1, fy1, fz}, {fx + 1, fy1, fz + 1},
                               mul(SH_EW, bot), mul(SH_EW, bot), SH_EW, SH_EW);
                    if (!solidAt(x - 1, y, z))
                        b.quad({fx, fy, fz}, {fx, fy, fz + 1}, {fx, fy1, fz + 1},
                               {fx, fy1, fz},
                               mul(SH_EW, bot), mul(SH_EW, bot), SH_EW, SH_EW);
                }
            }
        }
    return b.upload();
}

constexpr float PLAIN_Y = (float)Arena::WALL_HEIGHT;
constexpr int NEAR_R = 52;
constexpr int FAR_R = 128;

Mesh buildGrassMesh() {
    MeshBuilder b;
    int c = (int)Arena::CENTER_X;
    for (int x = c - NEAR_R; x < c + NEAR_R; ++x)
        for (int z = c - NEAR_R; z < c + NEAR_R; ++z) {
            if (Arena::insideDisc(x, z) || Arena::wallRing(x, z)) continue;
            b.quad({(float)x, PLAIN_Y, (float)z + 1}, {(float)x + 1, PLAIN_Y, (float)z + 1},
                   {(float)x + 1, PLAIN_Y, (float)z}, {(float)x, PLAIN_Y, (float)z}, SH_UP);
        }
    auto big = [&](float x0, float z0, float x1, float z1) {
        b.quad({x0, PLAIN_Y, z1}, {x1, PLAIN_Y, z1}, {x1, PLAIN_Y, z0}, {x0, PLAIN_Y, z0},
               SH_UP, 0, 0, x1 - x0, z1 - z0);
    };
    float n0 = (float)(c - NEAR_R), n1 = (float)(c + NEAR_R);
    float f0 = (float)(c - FAR_R), f1 = (float)(c + FAR_R);
    big(f0, f0, f1, n0);
    big(f0, n1, f1, f1);
    big(f0, n0, n0, n1);
    big(n1, n0, f1, n1);
    return b.upload();
}

Mesh buildSkirtMesh() {
    MeshBuilder b;
    int c = (int)Arena::CENTER_X;
    float f0 = (float)(c - FAR_R), f1 = (float)(c + FAR_R);
    float yTop = PLAIN_Y, yBot = PLAIN_Y - 1.0f;
    float len = f1 - f0;
    b.quad({f0, yBot, f1}, {f1, yBot, f1}, {f1, yTop, f1}, {f0, yTop, f1}, SH_NS, 0, 0, len, 1);
    b.quad({f1, yBot, f0}, {f0, yBot, f0}, {f0, yTop, f0}, {f1, yTop, f0}, SH_NS, 0, 0, len, 1);
    b.quad({f1, yBot, f1}, {f1, yBot, f0}, {f1, yTop, f0}, {f1, yTop, f1}, SH_EW, 0, 0, len, 1);
    b.quad({f0, yBot, f0}, {f0, yBot, f1}, {f0, yTop, f1}, {f0, yTop, f0}, SH_EW, 0, 0, len, 1);
    return b.upload();
}

Mesh buildDirtSkirtMesh() {
    MeshBuilder b;
    int c = (int)Arena::CENTER_X;
    float f0 = (float)(c - FAR_R), f1 = (float)(c + FAR_R);
    float yTop = PLAIN_Y - 1.0f, yBot = PLAIN_Y - 9.0f;
    float len = f1 - f0, dep = yTop - yBot;
    Color top = mul(WHITE, 0.9f), bot = mul(WHITE, 0.45f);
    b.quad({f0, yBot, f1}, {f1, yBot, f1}, {f1, yTop, f1}, {f0, yTop, f1},
           bot, bot, top, top, 0, 0, len, dep);
    b.quad({f1, yBot, f0}, {f0, yBot, f0}, {f0, yTop, f0}, {f1, yTop, f0},
           bot, bot, top, top, 0, 0, len, dep);
    b.quad({f1, yBot, f1}, {f1, yBot, f0}, {f1, yTop, f0}, {f1, yTop, f1},
           bot, bot, top, top, 0, 0, len, dep);
    b.quad({f0, yBot, f0}, {f0, yBot, f1}, {f0, yTop, f1}, {f0, yTop, f0},
           bot, bot, top, top, 0, 0, len, dep);
    return b.upload();
}


const Color STONE_SHADES[4] = {{104,104,104,255}, {117,117,117,255},
                               {127,127,127,255}, {136,136,136,255}};
const Color GRASS_SHADES[4] = {{106,153,60,255}, {119,167,66,255},
                               {130,180,74,255}, {142,192,83,255}};
const Color DIRT_SHADES[4]  = {{110,79,54,255}, {123,89,61,255},
                               {134,96,67,255}, {148,108,76,255}};

void buildBlockTextures() {
    S.stone = noiseTexture(STONE_SHADES, 11);
    S.grassTop = noiseTexture(GRASS_SHADES, 22);
    S.dirt = noiseTexture(DIRT_SHADES, 33);

    Image img = GenImageColor(16, 16, BLANK);
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            Color c = DIRT_SHADES[noiseLevel(x, y, 33)];
            int fringe = 2 + (int)(hash01(x, 0, 35) * 2.99f);
            if (y < fringe) c = GRASS_SHADES[noiseLevel(x, y, 22)];
            ImageDrawPixel(&img, x, y, c);
        }
    S.grassSide = texFromImage(img);

    SetTextureWrap(S.stone, TEXTURE_WRAP_REPEAT);
    SetTextureWrap(S.grassTop, TEXTURE_WRAP_REPEAT);
    SetTextureWrap(S.grassSide, TEXTURE_WRAP_REPEAT);
    SetTextureWrap(S.dirt, TEXTURE_WRAP_REPEAT);
}


static const char* SWORD_ROWS[16] = {
    ".............aaa",
    "............abba",
    "...........abdea",
    "..........abdea.",
    ".........abdea..",
    "........abdea...",
    "..aa...aedea....",
    "..afa.aedea.....",
    "...agcegea......",
    "...aggfea.......",
    "....afac........",
    "...hicaac.......",
    "..hjk.ccac......",
    "aaik....cc......",
    "afc.............",
    "ccc.............",
};
static const char SWORD_KEYS[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'};
static const Color SWORD_COLORS[] = {
    {14, 63, 54, 255},
    {164, 253, 240, 255},
    {8, 37, 32, 255},
    {43, 199, 172, 255},
    {51, 235, 203, 255},
    {21, 99, 85, 255},
    {30, 138, 119, 255},
    {73, 54, 21, 255},
    {104, 78, 30, 255},
    {137, 103, 39, 255},
    {40, 30, 11, 255},
};

Color swordPixel(int x, int y) {
    if (x < 0 || x > 15 || y < 0 || y > 15) return BLANK;
    char ch = SWORD_ROWS[y][x];
    if (ch == '.') return BLANK;
    for (int k = 0; k < (int)sizeof SWORD_KEYS; ++k)
        if (SWORD_KEYS[k] == ch) return SWORD_COLORS[k];
    return BLANK;
}

void buildCharacterTextures() {
    static const char* F[8] = {
        "hhhhhhhh", "hhhhhhhh", "hssssssh", "ssssssss",
        "swissiws", "ssssssss", "sssddsss", "ssssssss"};
    static const char* SIDE[8] = {
        "hhhhhhhh", "hhhhhhhh", "hsssssss", "ssssssss",
        "ssssssss", "ssssssss", "ssssssss", "ssssssss"};
    static const char* TOP[8] = {
        "hhhhhhhh", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh",
        "hhhhhhhh", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh"};
    static const char* BOT[8] = {
        "ssssssss", "ssssssss", "ssssssss", "ssssssss",
        "ssssssss", "ssssssss", "ssssssss", "ssssssss"};
    const Color skin{198,140,101,255}, dark{154,101,69,255}, hair{57,41,25,255};
    const Color eyeW{255,255,255,255}, iris{75,58,133,255};
    auto pick = [&](char ch) {
        switch (ch) {
            case 's': return skin;
            case 'd': return dark;
            case 'h': return hair;
            case 'w': return eyeW;
            case 'i': return iris;
        }
        return BLANK;
    };
    Image img = GenImageColor(48, 8, BLANK);
    const char* const* faces[6] = {F, SIDE, SIDE, SIDE, TOP, BOT};
    for (int f = 0; f < 6; ++f)
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x)
                ImageDrawPixel(&img, f * 8 + x, y, pick(faces[f][y][x]));
    S.head = texFromImage(img);

    const Color base{109,213,202,255}, lite{160,238,229,255}, darkD{72,164,155,255};
    Image plate = GenImageColor(8, 8, BLANK);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            Color c = base;
            if (x == 0 || y == 0) c = lite;
            if (x == 7 || y == 7) c = darkD;
            if (x > 0 && y > 0 && x < 7 && y < 7) {
                if (hash01(x, y, 55) > 0.86f) c = lite;
                else if (hash01(x, y, 56) > 0.86f) c = darkD;
            }
            ImageDrawPixel(&plate, x, y, c);
        }
    Image helm = GenImageColor(48, 8, BLANK);
    for (int f = 0; f < 6; ++f)
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) {
                if (f == 0 && x >= 1 && x <= 6 && y >= 2) continue;
                ImageDrawPixel(&helm, f * 8 + x, y, GetImageColor(plate, x, y));
            }
    S.helmet = texFromImage(helm);

    Image plateDark = ImageCopy(plate);
    ImageColorBrightness(&plateDark, -32);
    const Color skinTone{198,140,101,255};
    Image armImg = GenImageColor(8, 12, BLANK);
    Image legImg = GenImageColor(8, 12, BLANK);
    for (int y = 0; y < 12; ++y)
        for (int x = 0; x < 8; ++x) {
            ImageDrawPixel(&armImg, x, y,
                           y < 9 ? GetImageColor(plate, x, y % 8) : skinTone);
            ImageDrawPixel(&legImg, x, y,
                           y < 7 ? GetImageColor(plateDark, x, y % 8)
                                 : GetImageColor(plate, x, (y + 3) % 8));
        }
    S.armLimb = texFromImage(armImg);
    S.legLimb = texFromImage(legImg);
    S.diamond = texFromImage(plate);
    UnloadImage(plateDark);

    Image sh = GenImageColor(32, 32, BLANK);
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x) {
            float dx = (x + 0.5f - 16.0f) / 14.0f, dy = (y + 0.5f - 16.0f) / 14.0f;
            float d = dx * dx + dy * dy;
            if (d < 1.0f) {
                unsigned char a = (unsigned char)(95.0f * (1.0f - d));
                ImageDrawPixel(&sh, x, y, Color{0, 0, 0, a});
            }
        }
    S.shadow = texFromImage(sh);

    S.sword = spriteFromRows(SWORD_ROWS, 16, 16, SWORD_KEYS, SWORD_COLORS,
                             (int)sizeof SWORD_KEYS);
}


void buildHudSprites() {
    static const char* heartMask[9] = {
        ".xx...xx.",
        "xxxx.xxxx",
        "xxxxxxxxx",
        "xxxxxxxxx",
        "xxxxxxxxx",
        ".xxxxxxx.",
        "..xxxxx..",
        "...xxx...",
        "....x....",
    };
    auto heart = [&](Color fill, Color hi, Color lo, bool halfOnly) {
        Image img = GenImageColor(9, 9, BLANK);
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 9; ++x) {
                if (heartMask[y][x] != 'x') continue;
                if (halfOnly && x > 4) continue;
                Color c = fill;
                if ((x == 1 || x == 2) && y == 1) c = hi;
                if (y >= 6) c = lo;
                ImageDrawPixel(&img, x, y, c);
            }
        return texFromImage(img);
    };
    S.heartBg = heart(Color{48,48,48,255}, Color{74,74,74,255}, Color{28,28,28,255}, false);
    S.heartFull = heart(Color{227,42,42,255}, Color{255,140,140,255}, Color{150,14,14,255}, false);
    S.heartHalf = heart(Color{227,42,42,255}, Color{255,140,140,255}, Color{150,14,14,255}, true);

    static const char* shankRows[9] = {
        ".....mmm.",
        "....mmmm.",
        "...mmmmm.",
        "...mmmm..",
        "..bmmm...",
        ".bbm.....",
        "bbb......",
        "bb.......",
        ".........",
    };
    auto shank = [&](bool bg, bool half) {
        Image img = GenImageColor(9, 9, BLANK);
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 9; ++x) {
                char ch = shankRows[y][x];
                if (ch == '.') continue;
                Color c;
                bool dimmed = bg || (half && x < 4);
                if (dimmed) c = ch == 'm' ? Color{48,48,48,255} : Color{36,36,36,255};
                else if (ch == 'b') c = Color{224,213,193,255};
                else c = y <= 1 ? Color{213,125,50,255} : Color{188,102,33,255};
                ImageDrawPixel(&img, x, y, c);
            }
        return texFromImage(img);
    };
    S.shankBg = shank(true, false);
    S.shankFull = shank(false, false);
    S.shankHalf = shank(false, true);

    static const char* armorRows[9] = {
        ".........",
        "xx.....xx",
        "xxx...xxx",
        "xxxx.xxxx",
        "xxxxxxxxx",
        ".xxxxxxx.",
        ".xxxxxxx.",
        ".xxxxxxx.",
        ".........",
    };
    auto armor = [&](Color fill, Color hi) {
        Image img = GenImageColor(9, 9, BLANK);
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 9; ++x)
                if (armorRows[y][x] == 'x') ImageDrawPixel(&img, x, y, y <= 2 ? hi : fill);
        return texFromImage(img);
    };
    S.armorIcon = armor(Color{200,200,205,255}, Color{240,240,245,255});
    S.armorIconEmpty = armor(Color{52,52,52,255}, Color{70,70,70,255});
}

void drawSprite(Texture2D t, int x, int y, int scale, Color tint = WHITE) {
    DrawTexturePro(t, {0, 0, (float)t.width, (float)t.height},
                   {(float)x, (float)y, (float)(t.width * scale), (float)(t.height * scale)},
                   {0, 0}, 0, tint);
}

void drawSpriteF(Texture2D t, float x, float y, float scale, Color tint = WHITE) {
    DrawTexturePro(t, {0, 0, (float)t.width, (float)t.height},
                   {x, y, t.width * scale, t.height * scale}, {0, 0}, 0, tint);
}

}


void init() {
    if (S.ready) return;
    buildBlockTextures();
    buildCharacterTextures();
    buildHudSprites();

    S.arenaMesh = buildArenaMesh();
    S.grassMesh = buildGrassMesh();
    S.skirtMesh = buildSkirtMesh();
    S.dirtSkirtMesh = buildDirtSkirtMesh();
    S.matStone = makeMat(S.stone);
    S.matGrassTop = makeMat(S.grassTop);
    S.matGrassSide = makeMat(S.grassSide);
    S.matDirt = makeMat(S.dirt);

    S.headMesh = makeHead(8, 1.0f);
    S.helmetMesh = makeHead(8, 1.14f);
    S.torsoMesh = makeBox(8, 12, 4, 4, 12, 2);
    S.armMesh = makeBox(4, 12, 4, 2, 10, 2);
    S.legMesh = makeBox(4, 12, 4, 2, 12, 2);
    S.matHead = makeMat(S.head);
    S.matHelmet = makeMat(S.helmet);
    S.matDiamond = makeMat(S.diamond);
    S.matArm = makeMat(S.armLimb);
    S.matLeg = makeMat(S.legLimb);
    S.matSword = LoadMaterialDefault();
    S.matShadow = makeMat(S.shadow);

    {
        MeshBuilder b;
        float r = 0.42f;
        b.quad({-r, 0, r}, {r, 0, r}, {r, 0, -r}, {-r, 0, -r}, WHITE);
        S.shadowMesh = b.upload();
    }

    {
        MeshBuilder b;
        const float SZ = 0.95f;
        const float gx = 3.0f / 16.0f;
        const float gy = 1.0f - 13.0f / 16.0f;
        const float P = SZ / 16.0f;
        auto solid = [](int x, int y) { return swordPixel(x, y).a != 0; };
        for (int py = 0; py < 16; ++py)
            for (int px = 0; px < 16; ++px) {
                Color c = swordPixel(px, py);
                if (!c.a) continue;
                float x0 = (px / 16.0f - gx) * SZ, x1 = x0 + P;
                float y1 = (1.0f - py / 16.0f - gy) * SZ, y0 = y1 - P;
                float z0 = -P / 2, z1 = P / 2;
                Color ce = mul(c, 0.76f), ct = mul(c, 0.90f), cb = mul(c, 0.58f);
                b.quad({x0,y0,z1}, {x1,y0,z1}, {x1,y1,z1}, {x0,y1,z1}, c);
                b.quad({x1,y0,z0}, {x0,y0,z0}, {x0,y1,z0}, {x1,y1,z0}, c);
                if (!solid(px + 1, py))
                    b.quad({x1,y0,z1}, {x1,y0,z0}, {x1,y1,z0}, {x1,y1,z1}, ce);
                if (!solid(px - 1, py))
                    b.quad({x0,y0,z0}, {x0,y0,z1}, {x0,y1,z1}, {x0,y1,z0}, ce);
                if (!solid(px, py - 1))
                    b.quad({x0,y1,z0}, {x0,y1,z1}, {x1,y1,z1}, {x1,y1,z0}, ct);
                if (!solid(px, py + 1))
                    b.quad({x0,y0,z0}, {x1,y0,z0}, {x1,y0,z1}, {x0,y0,z1}, cb);
            }
        S.swordMesh = b.upload();
    }

    S.ready = true;
}

void unload() {
    if (!S.ready) return;
    for (Texture2D* t : {&S.stone, &S.grassTop, &S.grassSide, &S.dirt, &S.head, &S.helmet,
                         &S.diamond, &S.armLimb, &S.legLimb, &S.sword, &S.shadow, &S.heartBg,
                         &S.heartFull, &S.heartHalf, &S.shankBg, &S.shankFull, &S.shankHalf,
                         &S.armorIcon, &S.armorIconEmpty})
        UnloadTexture(*t);
    for (Mesh* m : {&S.arenaMesh, &S.grassMesh, &S.skirtMesh, &S.dirtSkirtMesh,
                    &S.headMesh, &S.helmetMesh, &S.torsoMesh, &S.armMesh, &S.legMesh,
                    &S.swordMesh, &S.shadowMesh})
        UnloadMesh(*m);
    S.ready = false;
}


Color skyColor() { return Color{124, 168, 255, 255}; }

void drawWorld(const Camera3D& cam) {
    Matrix I = MatrixIdentity();
    DrawMesh(S.arenaMesh, S.matStone, I);
    DrawMesh(S.grassMesh, S.matGrassTop, I);
    DrawMesh(S.skirtMesh, S.matGrassSide, I);
    DrawMesh(S.dirtSkirtMesh, S.matDirt, I);

    {
        rlDisableBackfaceCulling();
        rlDisableDepthMask();
        Vector3 dir = Vector3Normalize(Vector3{0.55f, 0.75f, 0.20f});
        Vector3 c = Vector3Add(cam.position, Vector3Scale(dir, 380.0f));
        Vector3 r = Vector3Normalize(Vector3CrossProduct(dir, Vector3{0, 1, 0}));
        Vector3 u = Vector3Normalize(Vector3CrossProduct(r, dir));
        float s = 28.0f;
        Vector3 a = Vector3Add(Vector3Add(c, Vector3Scale(r, -s)), Vector3Scale(u, -s));
        Vector3 b = Vector3Add(Vector3Add(c, Vector3Scale(r, s)), Vector3Scale(u, -s));
        Vector3 d = Vector3Add(Vector3Add(c, Vector3Scale(r, s)), Vector3Scale(u, s));
        Vector3 e = Vector3Add(Vector3Add(c, Vector3Scale(r, -s)), Vector3Scale(u, s));
        DrawTriangle3D(a, b, d, Color{255, 255, 234, 255});
        DrawTriangle3D(a, d, e, Color{255, 255, 234, 255});
        rlEnableDepthMask();
        rlEnableBackfaceCulling();
    }
}


void AimSmoother::update(float targetYaw, float targetPitch, float dt, float tau) {
    if (!primed || tau <= 0.0f) {
        yaw = targetYaw;
        pitch = targetPitch;
        primed = true;
        return;
    }
    float k = 1.0f - std::exp(-dt / tau);
    yaw += mc1218::Mth::wrapDegrees(targetYaw - yaw) * k;
    pitch += (targetPitch - pitch) * k;
}

void PlayerAnim::update(float velX, float velZ, float dt) {
    float speed = std::sqrt(velX * velX + velZ * velZ) / 0.05f;
    float target = Clamp(speed / 5.6f, 0.0f, 1.0f);
    limbAmount += (target - limbAmount) * Clamp(dt * 10.0f, 0.0f, 1.0f);
    limbSwing += speed * dt * 3.4f;
}

void drawPlayer(const PlayerPose& p) {
    Color tint = p.hurt ? Color{255, 96, 96, 255} : WHITE;
    Color legTint = p.hurt ? Color{255, 96, 96, 255} : p.accent;

    float walk = std::sin(p.anim.limbSwing) * 42.0f * p.anim.limbAmount;

    rlPushMatrix();
    rlTranslatef(p.pos.x, p.pos.y, p.pos.z);
    {
        rlPushMatrix();
        rlTranslatef(0, 0.02f - Clamp(p.pos.y, 0.0f, 3.0f) * 0.0f, 0);
        drawPart(S.shadowMesh, S.matShadow, WHITE);
        rlPopMatrix();
    }
    rlRotatef(-p.yaw, 0, 1, 0);

    const float hipY = 12 * PXS, neckY = 24 * PXS;
    float bodyLean = 0.0f, bodyDrop = 0.0f;
    if (p.crouching) { bodyLean = 27.0f; bodyDrop = -0.14f; }

    for (int i = 0; i < 2; ++i) {
        float side = i == 0 ? -1.0f : 1.0f;
        rlPushMatrix();
        rlTranslatef(side * 2 * PXS, hipY + bodyDrop * 0.5f, 0);
        rlRotatef(side * walk, 1, 0, 0);
        drawPart(S.legMesh, S.matLeg, legTint);
        rlPopMatrix();
    }

    rlTranslatef(0, hipY + bodyDrop, 0);
    rlRotatef(bodyLean, 1, 0, 0);
    rlTranslatef(0, -hipY, 0);

    rlPushMatrix();
    rlTranslatef(0, neckY, 0);
    drawPart(S.torsoMesh, S.matDiamond, tint);
    rlPopMatrix();

    for (int i = 0; i < 2; ++i) {
        float side = i == 0 ? -1.0f : 1.0f;
        bool swordArm = side < 0;
        rlPushMatrix();
        rlTranslatef(side * 6 * PXS, neckY - 2 * PXS, 0);
        float ang = -side * walk * 0.75f;
        float inward = 0.0f;
        if (swordArm && p.swing >= 0.0f) {
            float s = std::sin(p.swing * PI);
            float s2 = std::sin(std::sqrt(p.swing) * PI);
            ang = -(s2 * 58.0f + s * 64.0f) - 12.0f;
            inward = side * -10.0f * s;
        } else if (swordArm) {
            ang += -16.0f;
        }
        rlRotatef(inward, 0, 0, 1);
        rlRotatef(ang, 1, 0, 0);
        drawPart(S.armMesh, S.matArm, tint);

        if (swordArm) {
            rlTranslatef(0, -8.5f * PXS, 2.0f * PXS);
            rlRotatef(45.0f, 1, 0, 0);
            rlRotatef(90.0f, 0, 1, 0);
            rlScalef(0.70f, 0.70f, 0.70f);
            drawPart(S.swordMesh, S.matSword, tint);
        }
        rlPopMatrix();
    }

    rlPushMatrix();
    rlTranslatef(0, neckY, 0);
    rlRotatef(-bodyLean, 1, 0, 0);
    rlRotatef(p.pitch, 1, 0, 0);
    drawPart(S.headMesh, S.matHead, tint);
    drawPart(S.helmetMesh, S.matHelmet, tint);
    rlPopMatrix();

    rlPopMatrix();
}

void drawFirstPerson(const Camera3D& cam, float swing, float bobPhase, float bobAmount) {
    Vector3 fwd = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(fwd, cam.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, fwd));

    float bobX = std::sin(bobPhase) * 0.02f * bobAmount;
    float bobY = -std::fabs(std::cos(bobPhase)) * 0.025f * bobAmount;

    Vector3 anchor = cam.position;
    anchor = Vector3Add(anchor, Vector3Scale(fwd, 0.80f));
    anchor = Vector3Add(anchor, Vector3Scale(right, 0.46f + bobX));
    anchor = Vector3Add(anchor, Vector3Scale(up, -0.48f + bobY));

    Matrix basis = {right.x, up.x, -fwd.x, anchor.x,
                    right.y, up.y, -fwd.y, anchor.y,
                    right.z, up.z, -fwd.z, anchor.z,
                    0, 0, 0, 1};

    rlDisableDepthTest();
    rlPushMatrix();
    rlMultMatrixf(MatrixToFloat(basis));

    float s = 0.0f, s2 = 0.0f;
    if (swing >= 0.0f) {
        s = std::sin(swing * PI);
        s2 = std::sin(std::sqrt(swing) * PI);
    }

    rlPushMatrix();
    rlTranslatef(0.12f - 0.08f * s2, -0.34f - 0.03f * s, 0.24f - 0.16f * s);
    rlRotatef(-62.0f + 22.0f * s, 1, 0, 0);
    rlRotatef(16.0f, 0, 0, 1);
    rlScalef(0.85f, 0.85f, 0.85f);
    drawPart(S.armMesh, S.matArm, WHITE);
    rlPopMatrix();

    rlTranslatef(-0.42f * s2, 0.06f * s, -0.24f * s);
    rlRotatef(-70.0f * s, 1, 0, 0);
    rlRotatef(-36.0f * s2, 0, 1, 0);

    rlScalef(0.48f, 0.48f, 0.48f);
    rlRotatef(-18.0f, 0, 1, 0);
    rlRotatef(28.0f, 0, 0, 1);
    drawPart(S.swordMesh, S.matSword, WHITE);

    rlPopMatrix();
    rlEnableDepthTest();
}

void drawBoxWires(const mc1218::AABB& bb, Color c) {
    DrawCubeWires(Vector3{(float)((bb.minX + bb.maxX) / 2), (float)((bb.minY + bb.maxY) / 2),
                          (float)((bb.minZ + bb.maxZ) / 2)},
                  (float)(bb.maxX - bb.minX), (float)(bb.maxY - bb.minY),
                  (float)(bb.maxZ - bb.minZ), c);
}


int hudScale(int screenH) { return screenH >= 880 ? 3 : 2; }

void drawCrosshair(int w, int h, bool hitFlash) {
    Color c = hitFlash ? Color{255, 80, 80, 235} : Color{255, 255, 255, 200};
    int s = hudScale(h);
    DrawRectangle(w / 2 - 5 * s, h / 2 - s / 2, 10 * s, s, c);
    DrawRectangle(w / 2 - s / 2, h / 2 - 5 * s, s, 10 * s, c);
}

void drawAttackIndicator(int w, int h, float charge) {
    if (charge >= 1.0f) return;
    int s = hudScale(h);
    int bw = 16 * s, bh = 2 * s;
    int x = w / 2 - bw / 2, y = h / 2 + 8 * s;
    DrawRectangle(x - s, y - s, bw + 2 * s, bh + 2 * s, Color{0, 0, 0, 140});
    DrawRectangle(x, y, bw, bh, Color{255, 255, 255, 90});
    DrawRectangle(x, y, (int)(bw * charge), bh, Color{255, 255, 255, 235});
}

int drawHotbar(int w, int h, float swordDurFrac) {
    int s = hudScale(h);
    int slot = 20 * s, bw = slot * 9 + 2 * s, bh = 22 * s;
    int x = w / 2 - bw / 2, y = h - bh - 2 * s;
    DrawRectangle(x, y, bw, bh, Color{16, 16, 16, 175});
    DrawRectangleLinesEx({(float)x, (float)y, (float)bw, (float)bh}, (float)s,
                         Color{58, 58, 58, 230});
    for (int i = 0; i < 9; ++i)
        DrawRectangleLinesEx({(float)(x + s + i * slot), (float)(y + s), (float)slot,
                              (float)(bh - 2 * s)}, (float)s, Color{92, 92, 92, 150});
    DrawRectangleLinesEx({(float)(x - s), (float)(y - s), (float)(slot + 3 * s),
                          (float)(bh + 2 * s)}, (float)s, Color{245, 245, 245, 255});
    drawSprite(S.sword, x + s + 2 * s, y + 3 * s, s);
    if (swordDurFrac < 1.0f) {
        int bx = x + 3 * s, by = y + bh - 5 * s, bwid = 13 * s;
        float f = Clamp(swordDurFrac, 0.0f, 1.0f);
        DrawRectangle(bx, by, bwid, 2 * s, Color{0, 0, 0, 255});
        Color dc{(unsigned char)(255 * (1.0f - f)), (unsigned char)(255 * f), 40, 255};
        DrawRectangle(bx, by, (int)(bwid * f), s, dc);
    }
    return y;
}

void drawHearts(int x, int y, float health, int scale) {
    for (int i = 0; i < 10; ++i) {
        int px = x + i * 8 * scale;
        drawSprite(S.heartBg, px, y, scale);
        float v = health - 2.0f * i;
        if (v >= 2.0f) drawSprite(S.heartFull, px, y, scale);
        else if (v >= 1.0f) drawSprite(S.heartHalf, px, y, scale);
    }
}

void drawHunger(int xRight, int y, int food, int scale, unsigned char alpha) {
    Color tint{255, 255, 255, alpha};
    for (int i = 0; i < 10; ++i) {
        int px = xRight - (i + 1) * 8 * scale;
        drawSprite(S.shankBg, px, y, scale, tint);
        int v = food - 2 * i;
        if (v >= 2) drawSprite(S.shankFull, px, y, scale, tint);
        else if (v >= 1) drawSprite(S.shankHalf, px, y, scale, tint);
    }
}

void drawArmorRow(int x, int y, float armorPoints, int scale) {
    for (int i = 0; i < 10; ++i) {
        float v = armorPoints - 2.0f * i;
        drawSprite(v >= 1.0f ? S.armorIcon : S.armorIconEmpty, x + i * 8 * scale, y, scale);
    }
}

void drawBar(int x, int y, int w, int h, float frac, Color fill) {
    frac = Clamp(frac, 0.0f, 1.0f);
    DrawRectangle(x, y, w, h, Color{0, 0, 0, 160});
    DrawRectangle(x + 1, y + 1, (int)((w - 2) * frac), h - 2, fill);
}

void drawDamageOverlay(int w, int h, float a) {
    if (a <= 0.0f) return;
    unsigned char alpha = (unsigned char)Clamp(a * 115.0f, 0.0f, 115.0f);
    Color edge{200, 0, 0, alpha}, none{200, 0, 0, 0};
    int band = h / 4;
    DrawRectangleGradientV(0, 0, w, band, edge, none);
    DrawRectangleGradientV(0, h - band, w, band, none, edge);
    DrawRectangleGradientH(0, 0, band, h, edge, none);
    DrawRectangleGradientH(w - band, 0, band, h, none, edge);
}

void drawEndOverlay(int w, int h, bool won, const char* subtitle) {
    DrawRectangle(0, 0, w, h, won ? Color{20, 20, 40, 90} : Color{120, 0, 0, 110});
    const char* msg = won ? "Victory!" : "You died!";
    int fs = 12 * hudScale(h);
    int mw = MeasureText(msg, fs);
    DrawText(msg, (w - mw) / 2 + fs / 10, h / 3 + fs / 10, fs, Color{40, 10, 10, 190});
    DrawText(msg, (w - mw) / 2, h / 3, fs, won ? Color{255, 215, 80, 255} : WHITE);
    if (subtitle) {
        int ss = fs / 3;
        DrawText(subtitle, (w - MeasureText(subtitle, ss)) / 2, h / 3 + fs + ss, ss,
                 Color{235, 235, 235, 255});
    }
}

void drawNameTag(const Camera3D& cam, Vector3 feetPos, bool crouching, const char* name,
                 float health) {
    if (!name || !*name) return;
    Vector3 at{feetPos.x, feetPos.y + (crouching ? 1.5f : 1.8f) + 0.5f, feetPos.z};

    Vector3 fwd{cam.target.x - cam.position.x, cam.target.y - cam.position.y,
                cam.target.z - cam.position.z};
    Vector3 rel{at.x - cam.position.x, at.y - cam.position.y, at.z - cam.position.z};
    float dot = fwd.x * rel.x + fwd.y * rel.y + fwd.z * rel.z;
    if (dot <= 0.0f) return;

    float dist = std::sqrt(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z);
    if (dist < 1e-3f) return;

    int fs = (int)(250.0f / dist);
    if (fs < 16) fs = 16;
    if (fs > 44) fs = 44;

    Vector2 p = GetWorldToScreen(at, cam);
    int tw = MeasureText(name, fs);
    int padX = fs / 3, padY = fs / 6;
    int x = (int)p.x - tw / 2, y = (int)p.y - fs / 2;
    DrawRectangle(x - padX, y - padY, tw + 2 * padX, fs + 2 * padY, Color{0, 0, 0, 64});
    DrawText(name, x, y, fs, Color{255, 255, 255, 255});

    char buf[16];
    float hs = fs / 16.0f;
    int nfs = fs * 2 / 3 < 12 ? 12 : fs * 2 / 3;
    std::snprintf(buf, sizeof buf, "%.1f", health);
    float rowW = 8 * 10 * hs + hs;
    float totW = rowW + nfs / 3.0f + MeasureText(buf, nfs);
    float hx = p.x - totW / 2.0f;
    float hy = y - padY - 9 * hs - fs / 5.0f;
    for (int i = 0; i < 10; ++i) {
        float px = hx + i * 8 * hs;
        drawSpriteF(S.heartBg, px, hy, hs);
        float v = health - 2.0f * i;
        if (v >= 2.0f) drawSpriteF(S.heartFull, px, hy, hs);
        else if (v >= 1.0f) drawSpriteF(S.heartHalf, px, hy, hs);
    }
    DrawText(buf, (int)(hx + rowW + nfs / 3.0f), (int)(hy + 4.5f * hs - nfs / 2.0f), nfs,
             WHITE);
}

}
