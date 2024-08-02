# SISAP 2024 Indexing Challenge

This branch contains the code for our submission to the SISAP 2024 Indexing Challenge.

**Members:**

- [David Procházka](https://github.com/ProchazkaDavid), Masaryk University
- [Terézia Slanináková](https://github.com/TerkaSlan), Masaryk University
- Jozef Čerňanský, Masaryk University
- [Jaroslav Oľha](https://github.com/JaroOlha), Masaryk University
- Matej Antol, Masaryk University
- [Vlastislav Dohnal](https://github.com/dohnal), Masaryk University

## Setup

See also `.github/workflows/ci.yml`. Note the different parameters for 300K and 100M datasets when running the experiments. We encourage Windows users to use Docker.

### Using Docker

```shell
docker build -t sisap24 -f Dockerfile .
docker run -it --rm sisap24 bash
```

### Using Conda

```shell
conda create -n lmi -y python=3.11
conda activate lmi
conda install -c pytorch -y faiss-cpu=1.8.0
conda install -y h5py=3.11.0
pip install --no-cache-dir numpy==1.26.4 tqdm==4.66.4 loguru==0.7.2 scikit-learn==1.5.1
pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

## Running Experiments

### 300K dataset

Note that experiments with this dataset size will only visit a single bucket in each task. This is done to speed up the CI pipeline.

```shell
DBSIZE=300K

# Download data
mkdir data2024 && cd data2024
wget https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=${DBSIZE}.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=${DBSIZE}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5
cd ..

# Run experiments on 300K dataset
python3 task1.py --dataset-size ${DBSIZE} --sample-size 100000 --chunk-size 100000 &>task1.log
python3 task2.py --dataset-size ${DBSIZE} --sample-size 100000 --chunk-size 100000 &>task2.log
python3 task3.py --dataset-size ${DBSIZE} --sample-size 100000 --chunk-size 100000 &>task3.log
```

### 100M dataset

```shell
DBSIZE=100M

# Download data
mkdir data2024 && cd data2024
wget https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=${DBSIZE}.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=${DBSIZE}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5
cd ..

# Run experiments on 100M dataset
python3 task1.py &>task1.log
python3 task2.py &>task2.log
python3 task3.py &>task3.log
```

### Evaluating Results

```shell
# Calculate recall
python3 eval.py --results result res.csv

# Show the results
cat res.csv
```
