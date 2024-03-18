import pytest
from search import download, prepare
from li.BuildConfiguration import BuildConfiguration
from li.clustering import algorithms
from li.LearnedIndexBuilder import LearnedIndexBuilder

from sklearn import preprocessing
import pandas as pd

def get_data(data_part, **config):
    return np.array(
        h5py.File(
            os.path.join(
                'data',
                config['dataset'],
                config['size'],
                data_part
            ),
            'r'
        )[config['emb']]
    )


# Fixture for common setup tasks
@pytest.fixture(scope="session")
def setup():
    # Perform setup tasks here
    print("\nSetting up tests...")

    config = {
        # get the smallest version of the LAION dataset
        'dataset': 'pca32v2',
        'emb': 'pca32',
        'size': '100K',
        # n. of nearest neighbors
        'k': 10,
        # normalize the data to be able to use K-Means
        'preprocess': True
    }
    prepare(config['dataset'], config['size'])
    data = get_data("dataset.h5", **config)
    queries = get_data("query.h5", **config)
    if config['preprocess']:
        data = preprocessing.normalize(data)
        queries = preprocessing.normalize(queries)
    yield data, queries

# Example tests using the setup fixture
def test_one_level(setup):
    data, queries = setup
    n_categories = [10]

    build_config = BuildConfiguration(
        # which clustering algorithm to use
        algorithms['faiss_kmeans'],
        # how many epochs to train for
        100,
        # what model to use (see li/model.py
        'MLP',
        # what learning rate to use
        0.01,
        # how many categories at what level to build LMI for
        # 10, 10 results in 100 buckets in total
        n_categories
    )
    dimensionality = data.shape[1]
    sample = 1_000
    increment = 100

    first_build_data = data.iloc[:sample]
    builder = LearnedIndexBuilder(first_build_data, build_config)
    li, data_prediction, n_buckets_in_index, build_t, cluster_t = builder.build()
    assert data_prediction.shape == (sample, 1), "Data prediction shape is not correct: " + str(data_prediction.shape)
    assert builder.data.shape == (sample, dimensionality), "Data shape is not correct: " + str(builder.data.shape)
    
    insert_data = data.iloc[sample:sample+increment]
    data_prediction_2, n_buckets_in_index_2, insert_t = builder.insert(insert_data)
    assert data_prediction_2.shape == (increment, 1), "Data prediction shape is not correct: " + str(data_prediction.shape)
    # 32
    assert builder.data.shape == (sample+increment, dimensionality+len(n_categories)), "Data shape is not correct: " + str(builder.data.shape)

    data_prediction_all = np.vstack((data_prediction, data_prediction_2))

    n_queries=5
    k=10
    dists, nns, measured_time = li.search(
        data_navigation=builder.data,
        queries_navigation=queries[:n_queries],
        data_search=builder.data[[col for col in builder.data.columns if type(col) is int]],
        queries_search=queries[:n_queries],
        data_prediction=data_prediction_all,
        n_categories=n_categories,
        n_buckets=1,
        k=k,
    )
    assert dists.shape == (n_queries, k), "Dists shape is not correct: " + str(dists.shape)