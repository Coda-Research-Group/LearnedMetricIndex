# Learned Metric Index (LMI)

This project implements a Learned Metric Index using PyTorch and Rust.

## Prerequisites

-   Python 3.11 (will likely work with other versions, but only tested with this one)
-   Rust toolchain
-   GCC compiler
-   pipenv

## Setup Instructions

1. Install pipenv if you haven't already:

```bash
pip install pipenv
```

2. Install all dependencies specified in Pipfile

```bash
pipenv install
```

3. Activate the virtual environment:

```bash
pipenv shell
```

4. Build the Rust component:

```bash
RUSTFLAGS="-C linker=gcc" LIBTORCH_USE_PYTORCH=1 cargo build --release
cp -f target/release/liblmi.so lmi.so
```

## Running the Project

After setting up the environment, you can run the Python file:

```bash
python your_script.py
```
