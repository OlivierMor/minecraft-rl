# Disclaimer & Acceptable Use

## No affiliation

This project is an independent, fan-made research effort. It is **not**
affiliated with, endorsed by, sponsored by, or in any way associated with
Mojang Studios, Mojang Synergies AB, Microsoft Corporation, or any of their
subsidiaries. "Minecraft" is a trademark of Mojang Synergies AB. The name is
used here only to describe what the software interoperates with
(nominative use), not to suggest any official status.

## No Minecraft code or assets

This repository contains **no source code, no assets, no textures, no sounds,
and no other copyrighted material from Minecraft**. The simulator in `sim/`
is an original, independently written implementation of game mechanics 
(movement, combat timing, knockback), authored from scratch in
C++. Game *mechanics and rules* are not protected expression; no Minecraft
program code has been copied into this project. This project does not
distribute the game, does not modify the game's files, and does not bypass
any technical protection measures. Nothing here lets anyone play Minecraft
without buying it.

## Intended purpose

This is a research and education project about reinforcement learning:
building fast headless simulators, PPO self-play training, and studying
emergent competitive behavior. That is its only intended purpose.

## Acceptable use

By using this software you agree that you are solely responsible for how you
use it and anything you produce with it. In particular, you must **not** use
this software or models trained with it to:

- gain an unfair advantage over other players on any multiplayer server
  (including ranked, competitive, or matchmade play);
- automate gameplay on any server whose rules or operator does not explicitly
  permit bots;
- evade, test, or defeat anti-cheat systems;
- violate the [Minecraft EULA](https://www.minecraft.net/en-us/eula) or the
  [Minecraft Usage Guidelines](https://www.minecraft.net/en-us/usage-guidelines),
  or the terms of service of any server you connect to.

This repository deliberately ships **no component that connects to or
controls the actual game client**, precisely so that it cannot be dropped
into online play as-is.

## Model outputs: licensing

The code is licensed under the
[GNU General Public License v3.0](LICENSE). Trained models are covered by
the following additional terms:

1. Policy checkpoints and exported bundles produced by this software embed
   the software's own serialization format, data structures, constants and
   configuration values. To the extent such an artifact constitutes a
   covered work under section 0 of the GPL, it is covered by the GPL
   automatically.
2. Independently of point 1, permission to use this software for training
   is granted **on the condition** that any trained model, checkpoint,
   policy, or derivative thereof that you *distribute* (publish, sell,
   share, or embed in other software) is licensed under the GNU GPL v3.0
   or a compatible license, and is never distributed under closed-source
   or proprietary terms.
3. Models you train and keep private carry no obligation.

If any part of these output terms is held unenforceable, the remainder — and
the GPL itself for the code — still applies to the fullest extent permitted
by law.

## Model outputs: responsibility

Trained policies, checkpoints, replays, and any other artifacts you generate
with this software are produced by *you*, on your hardware, from your
configuration. The authors and contributors do not control, endorse, or
accept any responsibility or liability for such artifacts or for how they are
used. No trained weights are distributed with this repository.

## No warranty, no liability

The software is provided "AS IS", without warranty of any kind, as set out
in sections 15 and 16 of the [GPL v3.0](LICENSE). To the maximum extent
permitted by law, the authors and contributors shall not be liable for any
claim, damages, or other liability arising from the software, its outputs,
or their use.
