# minecraft-rl

A reinforcement learning framework for Minecraft-style combat, built for
research. Today it does 1.9+ sword duels; the architecture is designed
to grow into other Minecraft-like tasks.

It consists of a headless C++ reimplementation of the relevant combat
physics (written from scratch — it contains no Minecraft code or assets), a
PPO self-play trainer built on LibTorch, and viewers to inspect what the
policies learn.

Everything in the training loop is C++: there is no Python in the hot path,
and no game client involved anywhere. The simulator reproduces 1.9+ style
movement, attack-cooldown, knockback, sprint-reset and hit-detection rules
closely enough that the learned behavior is meaningful, at billions of steps per hour on one machine.

> **Scope:** this is a simulator-only research project. It ships nothing
> that connects to or controls the actual game, and it must not be used to
> automate play on servers or against other players. See
> [DISCLAIMER.md](DISCLAIMER.md) before using it.

## Layout

| Path | What it is |
|---|---|
| `sim/` | Headless combat sim — player physics, item/attack rules, hit registration, network delay model, scripted reference bot |
| `rl/` | Environment plumbing, observation/action/reward components, PPO learner, self-play pool with PFSP opponent sampling and Elo tracking |
| `apps/` | Executables: `train`, `export_policy`, `play`, `watch` |
| `viewer/` | raylib renderer used by `play` and `watch` |
| `cloud/` | One-shot GPU-box provisioning script and a training dashboard |
| `configs/` | Training configs (TOML) |

Observations, actions, rewards, terminal conditions and state setters are
registry-backed components selected by name from the config, so a new reward
term or observation layout is a new component plus a line of TOML rather than
a fork of the training loop.

## Build

Requires a C++17 compiler, CMake ≥ 3.18, and
[LibTorch](https://pytorch.org/get-started/locally/) (CUDA build for training,
CPU build is fine for the viewers). raylib is optional — without it, `play`
and `watch` are skipped and the rest still builds.

```bash
# LibTorch goes in third_party/libtorch (or point CMAKE_PREFIX_PATH elsewhere)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMCRL_NATIVE=ON \
      -DCMAKE_PREFIX_PATH="$PWD/third_party/libtorch"
cmake --build build -j
```

On a fresh Linux GPU box, `cloud/setup.sh` does all of the above — installs
packages, downloads the CUDA LibTorch, builds, and prints the commands to
start a run.

## Train

```bash
./build/train configs/sword.toml            # start a run
./build/train configs/sword.toml --resume   # continue a stopped one
```

Checkpoints, logs and metrics land in `runs/<run.name>/`. Serving that
directory over HTTP and opening `cloud/dashboard.html` gives live curves for
reward components, Elo, KL, entropy and throughput.

The default config trains 24576 parallel environments against a self-play
pool: a fraction of matches are mirror games, a decaying fraction are against
the scripted bot, and the rest are drawn from a capped pool of past
checkpoints weighted by PFSP so the policy keeps facing opponents it does not
already beat.

## Inspect

```bash
./build/play  runs/sword                  # play against a checkpoint in the sim
./build/watch runs/sword runs/sword       # watch two policies fight
./build/watch --replay replays/foo.mcrp   # replay a recorded match
```

`play` takes `--ping0/--ping1` to set simulated latency per side,
`--stochastic` to sample instead of argmax, and `--record/--no-record` to
control replay capture. Passing `bot` instead of a run directory uses the
scripted reference opponent. Both viewers run entirely inside the simulator.

`export_policy` serializes a trained policy plus the constants it was
trained against into a single self-describing bundle, for analysis or for
loading from other tooling.

## Legal

Copyright (C) 2026 OlivierMor.

Licensed under the [GNU General Public License v3.0](LICENSE): you may use,
study, modify and redistribute this software, but anything you distribute
that is built from it must be released under the GPL as well, with source.
Closed-source redistribution — including inside commercial "clients" — is
not permitted.

**Trained models are share-alike too.** Policies, checkpoints and exported
bundles produced with this framework embed the framework's own data
structures, constants and configuration; this project treats them as covered
works. If you distribute a trained model (or anything containing one), you
must license it under the GPL v3 and must not distribute it closed-source.
See [DISCLAIMER.md](DISCLAIMER.md) for the exact terms.

This project is not affiliated with or endorsed by Mojang Studios or
Microsoft. "Minecraft" is a trademark of Mojang Synergies AB. The repository
contains no Minecraft code or assets, and it is not permitted to use it to
cheat, to automate play against other people, or to violate the
[Minecraft EULA](https://www.minecraft.net/en-us/eula) or
[Usage Guidelines](https://www.minecraft.net/en-us/usage-guidelines).
Trained models and other outputs remain the sole responsibility of whoever
produces them. Full details in [DISCLAIMER.md](DISCLAIMER.md).
