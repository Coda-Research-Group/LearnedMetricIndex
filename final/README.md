If you want to compile from source, you can go into ./lmi-lib and run
RUSTFLAGS="-C linker=gcc" LIBTORCH_USE_PYTORCH=1 maturin develop --profile=dev
This will compile lmi-rs and install it into your current python environment.

If you just want to test out lmi-rs without compiling, you can just
pip install lmi-rs

You need to be in a Python environment that has torch2.5.1 . Other versions might not work.

you can then cd into lmi-lib and run ./run_task1_local.sh and so on with other tasks. You need to downlaod the 300K Laion-2B dataset for this
you can then evaluate the result using
python eval.py --results results_rust res_rust.csv

The original python implementation is available at
https://github.com/Coda-Research-Group/LearnedMetricIndex/tree/paper-sisap24-indexing-challenge

./rust_lmi contains the base rust implementation
./lmi-lib contains the python wrapper.
./benchmarks contain code for running benchmarks on metacentrum with larger datasets and also scripts for the analysis.

