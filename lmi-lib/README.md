# Learned Metric Index (LMI)

This project implements a Learned Metric Index using PyTorch and Rust.

## Prerequisites

-   Python 3.11 (will likely work with other versions, but only tested with this one)
-   Rust toolchain (nightly)
-   GCC compiler

## Setup Instructions

1. Install pipenv if you haven't already

```bash
pip install pipenv
```

2. Install all dependencies specified in Pipfile.lock

```bash
pipenv --python python3.11 sync
```

3. Activate the virtual environment

```bash
pipenv shell
```

4. Build the Rust component and install the package using maturin

```bash
RUSTFLAGS="-C linker=gcc" LIBTORCH_USE_PYTORCH=1 maturin develop --release
```

## Downloading the data
From this folder run
```bash
cd .. && mkdir data2024 && cd data2024
wget https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=300K.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5
cd ../lmi
```
The first wget will probably not work, so you need to get the dataset from elsewhere :)

## Running the Project

After setting up the environment, you can run the Python file:

```bash
python3 test.py
```

This will train the LMI on the 300k SISAP24 dataset and evaluate it on 10k public queries.

You can get the evaluation with
```bash
python3 eval.py --results result res.csv
```

and then
```bash
cat res.csv
```