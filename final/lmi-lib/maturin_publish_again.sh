#!/usr/bin/env bash
set -euo pipefail

PYTHON_VER="3.11"
IMAGE="quay.io/pypa/manylinux_2_34_x86_64"

docker run --rm -it -v "$(pwd)":/io "${IMAGE}" bash -c '
  set -euo pipefail

  PYTHON_VER="3.11"

  ## 0. system libraries for -sys crates
  yum -y install openssl-devel hdf5-devel pkgconfig

  ## 1. Rust tool-chain
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain nightly
  source "$HOME/.cargo/env"

  ## 2. Python tooling + libtorch via PyPI
  PY_BIN="/opt/python/cp${PYTHON_VER/./}-cp${PYTHON_VER/./}/bin"
  python${PYTHON_VER} -m pip install --upgrade pip maturin \
      torch==2.5.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

  # Make sure *that* python is the default one
  export PATH="${PY_BIN}:$PATH"

  rm -f /io/target/wheels/*.whl
  ## 3. build the wheel
  export RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+avx2"
  export LIBTORCH_USE_PYTORCH=1       # still needed so torch-sys reuses the wheel
  cd /io/lmi-lib
  python${PYTHON_VER} -m maturin build \
      --release \
      --manylinux 2_34 \
      --interpreter python${PYTHON_VER} \
      --skip-auditwheel


  ## 4. repair & copy out
  auditwheel repair \
    --exclude libtorch.so \
    --exclude libtorch_cpu.so \
    --exclude libtorch_python.so \
    --exclude libc10.so \
    /io/target/wheels/*.whl \
    -w /io/dist
'
