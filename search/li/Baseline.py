from chromadb.li_index.search.li.Logger import Logger
from chromadb.li_index.search.li.utils import pairwise_cosine
import time
import numpy as np


class Baseline(Logger):
    def __init__(self):
        pass

    def search(self, queries, data, k=10):
        s = time.time()
        anns = pairwise_cosine(data, queries).T
        nns = anns.argsort()[:, :k] + 1
        dists = np.sort(anns)[:, :k]
        return dists, nns, time.time() - s

    def build(self, data):
        s = time.time()
        self.logger.info("No build method implemented for baseline.")
        return time.time() - s
