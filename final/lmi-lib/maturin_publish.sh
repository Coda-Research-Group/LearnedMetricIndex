#! /bin/bash
RUSTFLAGS="-C linker=gcc" LIBTORCH_USE_PYTORCH=1 maturin build --release  --skip-auditwheel --manylinux 2_34

# Get the newest wheel in ../target/wheels
WHEEL=$(ls ../target/wheels/*.whl | tail -n 1)
maturin upload $WHEEL --username __token__ --password $PYPI_TOKEN


