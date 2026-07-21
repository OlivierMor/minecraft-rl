#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LIBTORCH_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip}"

echo "== network preflight =="
if ! getent hosts download.pytorch.org >/dev/null 2>&1; then
    echo "DNS is broken on this instance - installing public resolvers"
    printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\n" > /etc/resolv.conf
    if ! getent hosts download.pytorch.org >/dev/null 2>&1; then
        echo "ERROR: still cannot resolve hostnames - this host's networking is"
        echo "broken. Destroy the instance and rent a different offer."
        exit 1
    fi
fi

echo "== packages =="
if command -v apt-get >/dev/null; then
    apt-get update -y >/dev/null
    apt-get install -y --no-install-recommends \
        cmake g++ make unzip curl ca-certificates tmux rsync python3 >/dev/null
fi

echo "== libtorch (CUDA) =="
if [ ! -d third_party/libtorch ]; then
    mkdir -p third_party
    df -h . | tail -1 | awk '{print "disk free on this volume: " $4 " (need ~8 GB)"}'
    echo "downloading libtorch (~2.5 GB)..."
    curl -fSL -C - --progress-bar "$LIBTORCH_URL" -o third_party/libtorch.zip
    echo "unpacking..."
    unzip -q third_party/libtorch.zip -d third_party
    rm third_party/libtorch.zip
fi

echo "== build (sim + rl + tools, -march=native) =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMCRL_NATIVE=ON \
      -DCMAKE_PREFIX_PATH="$(pwd)/third_party/libtorch"
cmake --build build -j"$(nproc)"

echo "== dashboard =="
mkdir -p runs
cp -f cloud/dashboard.html runs/

echo
echo "ready. start training on this box:"
echo "  tmux new -s train"
echo "  ./build/train configs/sword.toml 2>&1 | tee -a runs/train.out"
echo "  (Ctrl-B then D detaches; --resume continues a stopped run)"
echo "dashboard: tmux new -d -s dash 'cd runs && python3 -m http.server 8080'"
echo "  then tunnel from your laptop: ssh -p <port> -N -L 8080:localhost:8080 root@<host>"
