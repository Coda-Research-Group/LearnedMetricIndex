from abc import ABC, abstractmethod
from chromadb.li_index.search.li.clustering import ClusteringAlgorithm
from typing import List

class ChromaIndex(ABC):

    @abstractmethod
    def add_items(self, data, ids=None, num_threads=-1, replace_deleted=False):
        pass

    @abstractmethod
    def close_file_handles(self):
        pass

    @abstractmethod
    def get_current_count(self):
        pass

    @abstractmethod
    def get_ids_list(self):
        pass

    @abstractmethod
    def get_items(self, ids=None):
        pass

    @abstractmethod
    def get_max_elements(self):
        pass

    @abstractmethod
    def init_index(self, max_elements, algorithms: List[ClusteringAlgorithm], epochs: [int], model: [str], learning_rate: [int], n_categories: [int],  is_persistent_index=False, persistence_location=None):
        pass

    @abstractmethod
    def knn_query(self, data, k=1, num_threads=-1, filter=None, *args, **kwargs):
        pass

    @abstractmethod
    def load_index(self, path_to_index, max_elements=0, allow_replace_deleted=False, is_persistent_index=False):
        pass

    @abstractmethod
    def mark_deleted(self, label):
        pass

    @abstractmethod
    def open_file_handles(self):
        pass

    @abstractmethod
    def persist_dirty(self):
        pass

    @abstractmethod
    def resize_index(self, new_size):
        pass

    @abstractmethod
    def save_index(self, path_to_index):
        pass

    @abstractmethod
    def set_num_threads(self, num_threads):
        pass

    @abstractmethod
    def unmark_deleted(self, label):
        pass

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __setstate__(self, arg0):
        pass

    # Properties
    dim = property()
    max_elements = property()
    num_threads = property()
    space = property()
