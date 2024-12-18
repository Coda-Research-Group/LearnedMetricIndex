# Enhancing Performance of Learned Metric Index for Indexing Large Datasets

### Preparing enviroment

```shell
conda create -n learnedmetric-bp -y python=3.11
conda activate learnedmetric-bp
conda install -c pytorch -y faiss-cpu=1.8.0
conda install -y h5py=3.11.0
pip install --no-cache-dir numpy==1.26.4 tqdm==4.66.4 loguru==0.7.2 scikit-learn==1.5.1
pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```
### Enviroment for GPU

```shell
conda create -n lmi-gpu -y python=3.11
conda activate lmi-gpu
conda install faiss-gpu=1.8.0 pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y h5py=3.11.0
pip install --no-cache-dir numpy==1.26.4 tqdm==4.66.4 loguru==0.7.2 scikit-learn==1.5.1
```

### 100M dataset

```shell
DBSIZE=100M

# Download data
wget https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=${DBSIZE}.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5
wget http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=${DBSIZE}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5
```

### 1B dataset

```shell

# Download data
wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin
```


### Evaluating Results

```shell
# Calculate recall
python3 eval.py --results result res.csv

# Show the results
cat res.csv
```

## Results and Figures

The results of experiments are stored in 'data' directory. All of the experiments were conducted on Metacentrum, launched with the scripts in 'metacentrum'. In 'experiments' is the Python source code used in the experiments. LMI 2023 implementation was adopted and modified from  
https://github.com/TerkaSlan/sisap23-laion-challenge-learned-index