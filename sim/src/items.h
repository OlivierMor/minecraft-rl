#pragma once
#include <cstdint>
#include "mc_math.h"

namespace mc1218 {

struct ItemDur {
    int maxDamage = 0;
    int damage = 0;
    bool broken = false;

    int remaining() const { return broken ? 0 : maxDamage - damage; }
};

enum ArmorSlot : int { FEET = 0, LEGS = 1, CHEST = 2, HEAD = 3 };

struct Loadout {
    static constexpr float SWORD_ATTACK_DAMAGE = 7.0f;
    static constexpr double ATTACK_SPEED = 1.6;
    static constexpr int SWORD_DURABILITY = 1561;
    static constexpr int SWORD_DAMAGE_PER_ATTACK = 1;

    static constexpr float UNBREAKING_IGNORE_ARMOR = 6.0f / 20.0f;
    static constexpr float UNBREAKING_IGNORE_TOOL = 3.0f / 4.0f;

    static constexpr int ARMOR_DURABILITY[4] = {429, 495, 528, 363};
    static constexpr int ARMOR_DEFENSE[4] = {3, 6, 8, 3};
    static constexpr float ARMOR_TOUGHNESS_EACH = 2.0f;
    static constexpr float PROT4_EPF_EACH = 4.0f;

    static constexpr float MAX_HEALTH = 20.0f;
    static constexpr double MOVEMENT_SPEED_BASE = (double)0.1f;
    static constexpr double SPRINT_SPEED_MODIFIER = 0.3;
    static constexpr double ENTITY_INTERACTION_RANGE = 3.0;
    static constexpr double BLOCK_INTERACTION_RANGE = 4.5;
    static constexpr double GRAVITY = 0.08;
    static constexpr float JUMP_STRENGTH = 0.42f;
    static constexpr double KNOCKBACK_RESISTANCE = 0.0;
    static constexpr float STEP_HEIGHT = 0.6f;
    static constexpr double SNEAKING_SPEED = 0.3;
    static constexpr double SAFE_FALL_DISTANCE = 3.0;
    static constexpr double FALL_DAMAGE_MULTIPLIER = 1.0;

    static float attackStrengthDelay() { return (float)(1.0 / ATTACK_SPEED * 20.0); }
};

struct Equipment {
    ItemDur sword{Loadout::SWORD_DURABILITY, 0, false};
    ItemDur armor[4] = {
        {Loadout::ARMOR_DURABILITY[FEET], 0, false},
        {Loadout::ARMOR_DURABILITY[LEGS], 0, false},
        {Loadout::ARMOR_DURABILITY[CHEST], 0, false},
        {Loadout::ARMOR_DURABILITY[HEAD], 0, false},
    };

    float armorValue() const {
        float v = 0.0f;
        for (int s = 0; s < 4; ++s)
            if (!armor[s].broken) v += (float)Loadout::ARMOR_DEFENSE[s];
        return v;
    }
    float toughness() const {
        float v = 0.0f;
        for (int s = 0; s < 4; ++s)
            if (!armor[s].broken) v += Loadout::ARMOR_TOUGHNESS_EACH;
        return v;
    }
    float protectionEpf() const {
        float v = 0.0f;
        for (int s = 0; s < 4; ++s)
            if (!armor[s].broken) v += Loadout::PROT4_EPF_EACH;
        return v;
    }
};

}
