#! /bin/bash
# RUSTFLAGS="-C linker=gcc" LIBTORCH_USE_PYTORCH=1 maturin build --release  --skip-auditwheel --manylinux 2_34

WHEEL=$(ls -t ./dist/*.whl | head -n 1)
maturin upload $WHEEL --username __token__ --password $PYPI_TOKEN


