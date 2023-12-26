import numpy as np
import pandas as pd
import sys

from chromadb.li_index.search import ChromaIndex
from chromadb.li_index.search.li.LearnedIndexBuilder import LearnedIndexBuilder
from chromadb.li_index.search.li.BuildConfiguration import BuildConfiguration
from chromadb.li_index.search.li.clustering import ClusteringAlgorithm
from sklearn import preprocessing
from typing import List, Optional, Union, Sequence, Dict


class LMI(ChromaIndex):
    _internal_index = None
    _build_config = None
    _dataset = None
    # maps every object from the dataset to the bucket from the tree leafs
    _data_prediction = None
    _n_categories = None

    def __init__(self):
        self._dataset = np.array([])

    def add_items(self, data: List[Union[Sequence[float], Sequence[int]]], ids=None, num_threads=-1,
                  replace_deleted=False):
        data_converted = np.array(data)
        if self._dataset.size == 0:
            data_appended = data_converted
        else:
            data_appended = np.concatenate((self._dataset, data_converted))
        self._dataset = data_appended

    def init_index(self,
                   max_elements,
                   clustering_algorithms: Optional[List[ClusteringAlgorithm]],
                   epochs: Optional[List[int]],
                   model_types: [str],
                   learning_rate: Optional[List[int]],
                   n_categories: Optional[List[int]],
                   kmeans: Optional[Dict],
                   is_persistent_index=False,
                   persistence_location=None
                   ):
        print(f"""
            LMI Build Config:
            {{
                clustering_algorithms: {clustering_algorithms},
                epochs: {epochs},
                model_types: {model_types},
                learning_rate: {learning_rate},
                n_categories: {n_categories},
            }}
             """)
        self._n_categories = n_categories
        self._build_config = BuildConfiguration(
            clustering_algorithms,
            epochs,
            model_types,
            learning_rate,
            n_categories,
            kmeans=kmeans
        )

    def build_index(self):
        # normalize dataset
        self._dataset = preprocessing.normalize(self._dataset)

        # Convert dataset to pandas and shift its index by one if it starts at zero
        data_for_build = pd.DataFrame(self._dataset)
        if data_for_build.index.start == 0:
            data_for_build.index += 1

        builder = LearnedIndexBuilder(data_for_build, self._build_config)

        li, data_prediction, n_buckets_in_index, build_t, cluster_t = builder.build()
        self._data_prediction = data_prediction
        self._internal_index = li
        print(f"LMI built with n_buckets_in_index: {n_buckets_in_index}")
        print(f"Time taken to build: {build_t}; Time taken to cluster: {cluster_t}")
        return data_prediction

    def knn_query(self, data, k=1, n_buckets=1, use_threshold=False, constraint_weight=0.0, num_threads=-1, filter=None, *args, **kwargs)\
            -> (np.ndarray, np.ndarray, np.ndarray):
        data_for_build = pd.DataFrame(self._dataset)

        # TODO: this operation should be performed in the build_index method
        if data_for_build.index.start == 0:
            data_for_build.index += 1

        data_converted = np.array(data)
        data_converted = preprocessing.normalize(data_converted)

        if filter is not None:
            filter = np.array([filter])

        dists, nns, bucket_order, measured_time = self._internal_index.search(
            data_navigation=data_for_build,
            queries_navigation=data_converted,
            data_search=data_for_build,
            queries_search=data_converted,
            data_prediction=self._data_prediction,
            n_categories=self._n_categories,
            n_buckets=n_buckets,
            k=k,
            attribute_filter=filter,
            constraint_weight=constraint_weight
        )

        return nns, dists, bucket_order

    def get_items(self, ids=None):
        return self._dataset

    def get_current_count(self):
        "Return size of the dataset that LMI currently operates on."
        return self._dataset.size

    def resize_index(self, new_size):
        # Resizing in the context of LMI does not make sense currently
        pass

    def close_file_handles(self):
        # Was Related to HNSW implementation not sure how this translates to LMI
        pass

    def get_max_elements(self):
        # LMI currently does not have notion of maximum number of elements
        # Return maxint to prevent resizing logic from executing
        return sys.maxsize

    def load_index(self, path_to_index, max_elements=0, allow_replace_deleted=False, is_persistent_index=False):
        # TODO: Loading from pickle file trained LMI index
        pass

    def mark_deleted(self, label):
        # TODO: implement grave marking into LMI so it does not return elements that should be perceived as deleted
        pass

    def open_file_handles(self):
        # Probably related to HNSW implementation
        pass

    def persist_dirty(self):
        # Not sure what this is for?
        pass

    def save_index(self, path_to_index):
        # TODO: implement saving triugh pickle file
        pass

    def set_num_threads(self, num_threads):
        # LMI currently works only one single thread so no need for this
        pass

    def get_ids_list(self):
        # Not used by LMISegment or by chroma in general
        pass

    def unmark_deleted(self, label):
        # TODO: same as mark_deleted
        pass

    # TODO: find out if these are necessary
    def __getstate__(self):
        pass

    def __repr__(self):
        pass

    def __setstate__(self, arg0):
        pass
