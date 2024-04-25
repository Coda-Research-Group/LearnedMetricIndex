# Learned Metric Index Starterpack

This branch contains a starter code to learn about the Learned Metric Index (LMI), its internals, and how to use it. The code is written in Python and uses PyTorch (neural network) and FAISS (k-means).

The `lmi.py` script does the following
- Reads the entire dataset into memory from a local file (see below for download link)
- Constructs a Learned Metric Index with a single model (768-512-384-320) and 320 buckets under it
    - The construction is done by
        - Running k-means clustering on the whole dataset with the given number of clusters (=number of buckets)
        - Training a neural network with the above architecture (the network predicts the bucket ID for a given input)
        - After training the network, the entire dataset is run through the network, obtaining the bucket ID for each object, which indicates in which bucket each object should be placed
        - Create buckets by assigning the points to the appropriate buckets
- Issue a query to the index to find the ten nearest neighbors for that query
  - The LMI passes the query through the neural network to find the bucket ID for the query
  - The LMI internally visits only a single bucket (in most cases, visiting a single bucket is not enough to find all ten nearest neighbors)
  - The LMI then retrieves the points from the bucket and computes the ten nearest neighbors in a brute-force manner

In addition, the script includes an evaluation of the obtained result
- The script computes the ground truth (the actual ten nearest neighbors for the query over the entire dataset)
- The script computes the recall (the number of nearest neighbors found divided by the number of nearest neighbors to be found)
  - One means that all nearest neighbors were found
  - Zero means that none of the nearest neighbors were found

## Running the code

```shell
# Download the dataset to your local machine
wget 'https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=100K.h5'

# Setup the environment and install the dependencies
conda create -n lmi-starterpack python=3.12
conda activate lmi-starterpack
conda install -c pytorch faiss-cpu=1.8.0
pip install h5py
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run the code
python3 lmi.py
```
