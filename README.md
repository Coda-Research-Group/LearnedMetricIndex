# Introduction

Learned Metric Index (LMI) is an index for approximate nearest neighbor search on complex data using machine learning and probability-based navigation. 


# Getting started

See examples of how to index and search in a dataset in: [01_Introduction.ipynb](01_Introduction.ipynb) notebook.

## Installation

See also ``.github/workflows/ci.yml``

### Using conda
```bash
conda create -n env python=3.8
conda activate env
conda install matplotlib pandas scikit-learn jupyterlab
pip install h5py flake8 setuptools tqdm faiss-cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --editable .
```

## Running

```bash
jupyter-lab
# and open 01_Introduction.ipynb

# or
python3 search/search.py
```

## Evaluation

```bash
python3 eval/eval.py
python3 eval/plot.py res.csv
```

## Performance

**LMI comprised of 1 ML model**
- Recall: 91.421%
- Search runtime (for 10k queries): ~220s
- Build time: 20828s
- Dataset: LAION1B, 10M subset
- Hardware used:
    - CPU Intel Xeon Gold 6130
    - 42gb RAM
    - 1 CPU core
- Hyperparameters:
    - 120 leaf nodes
    - 200 epochs
    - 1 hidden layer with 512 neurons
    - 0.01 learning rate
    - 4 leaf nodes stop condition

## Hardware requirements

**10M:**
- 42gb RAM
- 1 CPU core
- ~6h of runtime (waries depending on the hardware)

# LMI in action
🌐 [**Similarity search in 214M protein structures (AlphaFold DB)**](https://alphafind.fi.muni.cz/search)

# Publications

**"LMI Proposition" (2021):**
> M. Antol, J. Ol'ha, T. Slanináková, V. Dohnal: [Learned Metric Index—Proposition of learned indexing for unstructured data](https://www.sciencedirect.com/science/article/pii/S0306437921000326?casa_token=EvG8iaWkqQUAAAAA:xgfbutrsNGcBXnTN-U4MQ65hgmPE3fAyzwqtijzGC-JRrkO1IYNmcN3A8yMsSOT3CCoHpqVtMA). Information Systems, 2021 - Elsevier (2021)

**"Data-driven LMI" (2021):**
> T. Slanináková, M. Antol, J. Ol'ha, V. Kaňa, V. Dohnal: [Learned Metric Index—Proposition of learned indexing for unstructured data](https://link.springer.com/chapter/10.1007/978-3-030-89657-7_7). SISAP 2021 - Similarity Search and Applications pp 81-94 (2021)

**"LMI in Proteins" (2022):**
> J. Ol'ha, T. Slanináková, M. Gendiar, M. Antol, V. Dohnal: [Learned Indexing in Proteins: Extended Work on Substituting Complex Distance Calculations with Embedding and Clustering Techniques](https://arxiv.org/abs/2208.08910), and [Learned Indexing in Proteins: Substituting Complex Distance Calculations with Embedding and Clustering Techniques](https://link.springer.com/chapter/10.1007/978-3-031-17849-8_22) SISAP 2022 - Similarity Search and Applications pp 274-282 (2022)

**"Reproducible LMI" (2023):**
- [**Repository**](https://github.com/TerkaSlan/LMIF)
- [**Mendeley data**](https://data.mendeley.com/datasets/8wp73zxr47/12)
> T. Slanináková, M. Antol, J. Ol'ha, V. Kaňa, V. Dohnal, S. Ladra, M. A. Martinez-Prieto: [Reproducible experiments with Learned Metric Index Framework](https://www.sciencedirect.com/science/article/pii/S0306437923000911). Information Systems, Volume 118, September 2023, 102255 (2023)

**"LMI in a large (214M) protein database" (2024):**
- [**Web**](https://alphafind.fi.muni.cz/search)
- [**Repository**](https://github.com/Coda-Research-Group/AlphaFind)
- [**Data**](https://data.narodni-repozitar.cz/general/datasets/d35zf-1ja47)
> PROCHÁZKA, David, Terézia SLANINÁKOVÁ, Jaroslav OĽHA, Adrián ROŠINEC, Katarína GREŠOVÁ, Miriama JÁNOŠOVÁ, Jakub ČILLÍK, Jana PORUBSKÁ, Radka SVOBODOVÁ, Vlastislav DOHNAL a Matej ANTOL.: [AlphaFind: Discover structure similarity across the entire known proteome](https://www.biorxiv.org/content/10.1101/2024.02.15.580465v1). BioRxiv (pre-print version)


## Team
🔎[**Complex data analysis research group**](https://disa.fi.muni.cz/complex-data-analysis)
- [Terézia Slanináková](https://github.com/TerkaSlan), Masaryk University
- [David Procházka](https://github.com/ProchazkaDavid), Masaryk University
- [Jaroslav Oľha](https://github.com/JaroOlha), Masaryk University
- Matej Antol, Masaryk University
- [Vlastislav Dohnal](https://github.com/dohnal), Masaryk University
