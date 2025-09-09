# Learned Metric Index on OOD Datasets from VIBE

## Setup

### Using Docker

```shell
docker build -t sisap24 -f Dockerfile .
docker run -it -d --rm --name sisap24 -v $(pwd)/data:/app/data -v $(pwd)/result:/app/result sisap24
docker exec -it sisap24 bash
nohup bash run.sh &
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

Download OOD datasets from `https://github.com/vector-index-bench/vibe#datasets` and place them in the `data` directory.

## Evaluating Results

```shell
python eval.py out.csv
```

## Plotting Results

`pip install matplotlib seaborn`

```shell
python plot.py
```
